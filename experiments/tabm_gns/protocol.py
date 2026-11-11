"""
Data preparation, TabM training, the TabGNS search loop for TabM, and evaluation.

The search loop is `tabgns.tabgns.TabGNS.fit` with the MLP swapped for a gated TabM:
  per training batch: one weight step on the training batch; one no-grad evaluation of the loss on the
  next (cycled) validation batch in eval mode; one gate step on that validation batch, still in eval
  mode (TabGNS never switches back to train mode before its gate step).
Early stopping watches the epoch's mean validation-batch loss with patience `stop_patience`. TabGNS
hands its own net to `EarlyStopper`, so the "best" model it returns is the net from the last epoch;
`checkpoint: last` reproduces that, `checkpoint: best` keeps a real snapshot of the best epoch.

TabM-specific choices (TabM's own conventions):
  * training loss = mean of the k members' losses (each member is trained on its own prediction);
  * prediction = mean of the members' probabilities (classification) or outputs (regression);
  * validation loss for early stopping = loss of that ensemble prediction.
The gate step uses the training loss (mean over members) on the validation batch.
"""
from __future__ import annotations

import copy
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from torch import Tensor

from experiments.tabm_gns.model import GatedTabM, TabM

Split = Tuple[np.ndarray, np.ndarray]


# ----------------------------------------------------------------------------------------------- data

@dataclass
class Prepared:
  name: str
  task: str                      # 'regression' | 'classification'
  train: Split
  valid: Split
  test: Split
  out_dim: int
  feature_chunks: List[int]      # input columns per original feature (TabM's per-feature random signs)
  target_mean: float = 0.0       # regression targets are standardised on train; metrics undo it
  target_std: float = 1.0
  info: Dict = field(default_factory=dict)

  @property
  def inp_dim(self) -> int:
    return self.train[0].shape[1]


def _openml_frame(did: int):
  import openml
  # Read the ARFF (cached as a pickle) rather than the parquet file. OpenML did not serve parquet for these
  # datasets when the study started, so every run so far read the ARFF; parquet also needs a working pyarrow.
  os.environ['OPENML_SKIP_PARQUET'] = 'true'
  d = openml.datasets.get_dataset(did, download_data=True, download_qualities=False,
                                  download_features_meta_data=False)
  X, y, cat_mask, names = d.get_data(target=d.default_target_attribute, dataset_format='dataframe')
  return X, y, list(cat_mask), list(names)


def prepare(cfg: dict, seed: int) -> Prepared:
  """60/20/20 split (as experiments.datasets.split), then preprocessing fitted on the training part only:
  quantile-normal numeric features, one-hot categorical features (missing values as their own level,
  unseen levels ignored), standardised regression targets."""
  X, y, cat_mask, names = _openml_frame(cfg['openml_id'])
  task = cfg['task']
  cat_cols = [n for n, c in zip(names, cat_mask) if c]
  num_cols = [n for n, c in zip(names, cat_mask) if not c]
  idx = np.arange(len(X))
  i_tr, i_te = train_test_split(idx, test_size=0.2, random_state=seed)
  i_tr, i_va = train_test_split(i_tr, test_size=0.25, random_state=seed)

  parts = {}
  if num_cols:
    num = X[num_cols].astype(np.float64).to_numpy()
    # Numeric NaNs (none in the chosen datasets) would be mean-filled from train.
    if np.isnan(num).any():
      num = np.where(np.isnan(num), np.nanmean(num[i_tr], axis=0), num)
    qt = QuantileTransformer(output_distribution='normal', random_state=0,
                             n_quantiles=min(1000, len(i_tr)), subsample=10**9)
    qt.fit(num[i_tr])
    parts['num'] = qt.transform(num).astype(np.float32)
  chunks = [1] * len(num_cols)
  if cat_cols:
    cat = X[cat_cols].astype(object).where(X[cat_cols].notna(), '__missing__').astype(str).to_numpy()
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)
    enc.fit(cat[i_tr])
    parts['cat'] = enc.transform(cat)
    chunks += [len(c) for c in enc.categories_]
  Xp = np.concatenate([parts[k] for k in ('num', 'cat') if k in parts], axis=1).astype(np.float32)

  info = {'n_rows': int(len(X)), 'n_num': len(num_cols), 'n_cat': len(cat_cols), 'inp_dim': int(Xp.shape[1])}
  if task == 'classification':
    classes = sorted(pd.unique(y.astype(str)))
    yp = pd.Categorical(y.astype(str), categories=classes).codes.astype(np.int64)
    out_dim, mean, std = len(classes), 0.0, 1.0
    info['classes'] = classes
  else:
    yv = y.astype(np.float64).to_numpy()
    mean, std = float(yv[i_tr].mean()), float(yv[i_tr].std())
    yp = ((yv - mean) / std).astype(np.float32).reshape(-1, 1)
    out_dim = 1
  return Prepared(cfg['name'], task, (Xp[i_tr], yp[i_tr]), (Xp[i_va], yp[i_va]), (Xp[i_te], yp[i_te]),
                  out_dim, chunks, mean, std, info)


def batches(split_: Split, batch_size: int, gen: torch.Generator, shuffle: bool = True):
  X, y = torch.as_tensor(split_[0]), torch.as_tensor(split_[1])
  n = len(X)
  order = torch.randperm(n, generator=gen) if shuffle else torch.arange(n)
  for i in range(0, n, batch_size):
    j = order[i:i + batch_size]
    yield X[j], y[j]


def n_batches(split_: Split, batch_size: int) -> int:
  return math.ceil(len(split_[0]) / batch_size)


# ----------------------------------------------------------------------------------------------- losses

def member_loss(task: str, out: Tensor, y: Tensor) -> Tensor:
  """TabM training loss: mean over the k members of each member's loss. out: (B, k, d_out)."""
  k = out.shape[1]
  if task == 'classification':
    return F.cross_entropy(out.flatten(0, 1), y.repeat_interleave(k))
  return F.mse_loss(out, y.unsqueeze(1).expand_as(out))


def ensemble_output(task: str, out: Tensor) -> Tensor:
  """Log of the mean member probability (classification) or the mean output (regression). (B, d_out)."""
  if task == 'classification':
    return torch.logsumexp(F.log_softmax(out, dim=-1), dim=1) - math.log(out.shape[1])
  return out.mean(dim=1)


def ensemble_loss(task: str, out: Tensor, y: Tensor) -> Tensor:
  e = ensemble_output(task, out)
  return F.nll_loss(e, y) if task == 'classification' else F.mse_loss(e, y)


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, task: str, batch_size: int = 4096) -> np.ndarray:
  model.eval()
  outs = [ensemble_output(task, model(torch.as_tensor(X[i:i + batch_size]))) for i in range(0, len(X), batch_size)]
  return torch.cat(outs).numpy()


def metrics(task: str, pred: np.ndarray, y: np.ndarray, data: Prepared) -> Dict[str, float]:
  """Accuracy (%) and log-loss for classification; RMSE and MSE in the target's original units for regression.
  `pred` is log-probabilities (TabM) or logits (MLP) for classification."""
  if task == 'classification':
    logp = torch.log_softmax(torch.as_tensor(pred, dtype=torch.float64), dim=1)
    return {'accuracy': float((logp.argmax(1).numpy() == y).mean() * 100),
            'logloss': float(F.nll_loss(logp, torch.as_tensor(y)).item())}
  err = (pred.reshape(-1) - y.reshape(-1)) * data.target_std
  mse = float(np.mean(err ** 2))
  return {'rmse': math.sqrt(mse), 'mse': mse}


# ----------------------------------------------------------------------------------------------- training

def make_optimizer(params, cfg: dict) -> torch.optim.Optimizer:
  return torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['weight_decay'])


def _log(msg: str) -> None:
  print(msg, file=sys.stderr, flush=True)


def train_tabm(model: TabM, data: Prepared, cfg: dict, gen: torch.Generator, tag: str = '') -> Dict:
  """AdamW, mean-of-members loss, early stopping on the ensemble's validation loss, best snapshot kept."""
  opt = make_optimizer(model.parameters(), cfg)
  Xv, yv = torch.as_tensor(data.valid[0]), torch.as_tensor(data.valid[1])
  best, best_state, bad, hist = math.inf, copy.deepcopy(model.state_dict()), 0, []
  start = time.perf_counter()
  epoch = 0
  for epoch in range(1, cfg['epochs'] + 1):
    model.train()
    tl = 0.
    for xb, yb in batches(data.train, cfg['batch_size'], gen):
      opt.zero_grad(set_to_none=True)
      loss = member_loss(data.task, model(xb), yb)
      loss.backward()
      opt.step()
      tl += loss.item()
    model.eval()
    with torch.no_grad():
      vl = sum(ensemble_loss(data.task, model(Xv[i:i + 4096]), yv[i:i + 4096]).item() * len(Xv[i:i + 4096])
               for i in range(0, len(Xv), 4096)) / len(Xv)
    hist.append((round(tl / n_batches(data.train, cfg['batch_size']), 6), round(vl, 6)))
    _log(f'[{tag} train] epoch {epoch} train {hist[-1][0]:.4f} valid {vl:.4f} widths {model.widths} '
         f'{time.perf_counter() - start:.0f}s')
    if not math.isfinite(vl):
      break
    if vl < best:
      best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
    else:
      bad += 1
      if bad >= cfg['patience']:
        break
  model.load_state_dict(best_state)
  return {'epochs': epoch, 'best_valid_loss': best, 'time_s': time.perf_counter() - start, 'loss_history': hist}


def search(net: GatedTabM, data: Prepared, cfg: dict, train_cfg: dict, gen: torch.Generator, tag: str = '') -> Dict:
  """TabGNS search on a gated TabM. See the module docstring for the step order."""
  wopt = make_optimizer(net.weight_parameters(), train_cfg)
  aopt = torch.optim.Adam(net.arch_parameters(), lr=cfg['arch_lr'], weight_decay=0.)
  bs = train_cfg['batch_size']
  nb = n_batches(data.train, bs)
  valid_iter = batches(data.valid, bs, gen)
  best, bad, best_epoch = math.inf, 0, 0
  best_state = copy.deepcopy(net.state_dict())
  arch_hist, p_hist, loss_hist = [], [], []
  start = time.perf_counter()
  epoch = 0
  for epoch in range(1, cfg['epochs'] + 1):
    tl = vl = 0.
    for xb, yb in batches(data.train, bs, gen):
      try:
        xv, yv = next(valid_iter)
      except StopIteration:
        valid_iter = batches(data.valid, bs, gen)
        xv, yv = next(valid_iter)
      net.train()
      wopt.zero_grad(set_to_none=True)
      loss = member_loss(data.task, net(xb), yb)
      loss.backward()
      wopt.step()
      tl += loss.item()

      net.eval()
      with torch.no_grad():
        vl += ensemble_loss(data.task, net(xv), yv).item()

      aopt.zero_grad(set_to_none=True)
      aloss = member_loss(data.task, net(xv), yv)
      aloss.backward(inputs=net.arch_parameters())  # gate gradients only; weight grads are not needed
      aopt.step()
    vl /= nb
    arch_hist.append(net.curr_architecture())
    p_hist.append([round(float(g.p_open().mean()), 4) for g in net.gates])
    loss_hist.append((round(tl / nb, 6), round(vl, 6)))
    _log(f'[{tag} search] epoch {epoch} train {tl / nb:.4f} valid {vl:.4f} arch {arch_hist[-1]} '
         f'p_open {p_hist[-1]} {time.perf_counter() - start:.0f}s')
    if not math.isfinite(vl) or not all(torch.isfinite(p).all() for p in net.arch_parameters()):
      break
    if vl < best:
      best, bad, best_epoch = vl, 0, epoch
      if cfg['checkpoint'] == 'best':
        best_state = copy.deepcopy(net.state_dict())
    else:
      bad += 1
      if bad >= cfg['patience']:
        break
  if cfg['checkpoint'] == 'best':
    net.load_state_dict(best_state)
  return {'epochs': epoch, 'best_epoch': best_epoch, 'best_valid_loss': best,
          'time_s': time.perf_counter() - start, 'arch_history': arch_hist, 'p_open_history': p_hist,
          'loss_history': loss_hist}
