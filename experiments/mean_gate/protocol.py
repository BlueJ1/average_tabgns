"""
Data preparation, search, extraction and fine-tuning for the mean-gate study. Both arms run exactly
this code; only `gate_mode` differs.

Random streams (all derived from the run seed, none from torch's global generator except weight init):
  weight init      torch.manual_seed(seed) right before building each network
  train order      generator seed + 1   (search and fine-tuning use fresh generators with the same seed)
  valid order      generator seed + 2   (batches for the architecture step)
  gate noise       generator seed + 3
  stochastic eval  generator seed + 4
Mean bookkeeping draws no random numbers, so it cannot shift any of these streams.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_diabetes, make_classification, make_friedman1, make_regression
from sklearn.preprocessing import QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset

from experiments.datasets import Dataset, split
from experiments.mean_gate.model import ControlledGatedNet

STREAM_OFFSETS = {'train_order': 1, 'valid_order': 2, 'gate_noise': 3, 'stochastic_eval': 4}


def generator(seed: int, stream: str) -> torch.Generator:
  return torch.Generator().manual_seed(seed * 1000 + STREAM_OFFSETS[stream])


# ---------------------------------------------------------------- data

def make_raw_dataset(cfg: dict) -> Dataset:
  """Unscaled data; synthetic generation uses cfg['data_seed'], independent of the run seed."""
  gen, rs = cfg['generator'], cfg.get('data_seed')
  if gen == 'friedman1':
    X, y = make_friedman1(n_samples=cfg['n_samples'], n_features=cfg['n_features'], noise=cfg['noise'], random_state=rs)
  elif gen == 'regression':
    X, y = make_regression(n_samples=cfg['n_samples'], n_features=cfg['n_features'], n_targets=cfg['n_targets'],
                           noise=cfg['noise'], random_state=rs)
  elif gen == 'classification':
    X, y = make_classification(n_samples=cfg['n_samples'], n_features=cfg['n_features'],
                               n_informative=cfg['n_informative'], n_classes=cfg['n_classes'], random_state=rs)
  elif gen == 'sklearn_diabetes':  # real data bundled with scikit-learn, no download
    X, y = load_diabetes(return_X_y=True, scaled=False)
  else:
    raise ValueError(f'Unknown generator {gen}')
  if cfg['task'] == 'classification':
    y = np.asarray(y, dtype=np.int64)
  else:
    y = np.asarray(y, dtype=np.float32).reshape(len(y), -1)
  return Dataset(name=cfg['name'], task=cfg['task'], X=np.asarray(X, dtype=np.float64), y=y)


@dataclass
class Prepared:
  task: str
  inp_dim: int
  out_dim: int
  train: Tuple[np.ndarray, np.ndarray]
  valid: Tuple[np.ndarray, np.ndarray]
  test: Tuple[np.ndarray, np.ndarray]
  target_mean: Optional[np.ndarray]
  target_std: Optional[np.ndarray]


def prepare(ds: Dataset, split_seed: int) -> Prepared:
  """60/20/20 split first, then quantile-normal features and (regression) standardised targets fitted on train only."""
  (xtr, ytr), (xva, yva), (xte, yte) = split(ds, seed=split_seed)
  scaler = QuantileTransformer(output_distribution='normal', random_state=0, n_quantiles=min(1000, len(xtr)))
  xtr, xva, xte = (scaler.fit_transform(xtr).astype(np.float32), scaler.transform(xva).astype(np.float32),
                   scaler.transform(xte).astype(np.float32))
  mu = sd = None
  if ds.task == 'regression':
    mu, sd = ytr.mean(axis=0), ytr.std(axis=0)
    ytr, yva, yte = [((y - mu) / sd).astype(np.float32) for y in (ytr, yva, yte)]
  out_dim = int(ds.y.max()) + 1 if ds.task == 'classification' else ds.y.shape[1]
  return Prepared(ds.task, xtr.shape[1], out_dim, (xtr, ytr), (xva, yva), (xte, yte), mu, sd)


def _tensors(split_: Tuple[np.ndarray, np.ndarray], task: str):
  X = torch.tensor(split_[0], dtype=torch.float32)
  y = torch.tensor(split_[1], dtype=torch.long if task == 'classification' else torch.float32)
  return X, y


def loader(split_, task: str, batch_size: int, gen: torch.Generator) -> DataLoader:
  return DataLoader(TensorDataset(*_tensors(split_, task)), batch_size=batch_size, shuffle=True, generator=gen)


def criterion(task: str) -> nn.Module:
  return nn.MSELoss() if task == 'regression' else nn.CrossEntropyLoss()


# ---------------------------------------------------------------- evaluation

@torch.no_grad()
def metrics(task: str, out: torch.Tensor, y: torch.Tensor, target_std: Optional[np.ndarray] = None) -> Dict[str, float]:
  if task == 'regression':
    res = {'mse': float(F.mse_loss(out, y))}
    if target_std is not None:  # MSE in the original target units
      res['mse_orig'] = float((((out - y) ** 2) * torch.tensor(target_std ** 2, dtype=out.dtype)).mean())
    return res
  return {'cross_entropy': float(F.cross_entropy(out, y)), 'accuracy': float((out.argmax(1) == y).float().mean() * 100)}


@torch.no_grad()
def evaluate(model_fn, split_, task: str, target_std=None) -> Dict[str, float]:
  X, y = _tensors(split_, task)
  return metrics(task, model_fn(X), y, target_std)


@torch.no_grad()
def evaluate_stochastic(net: ControlledGatedNet, split_, task: str, n_samples: int, gen: torch.Generator,
                        target_std=None) -> Dict[str, float]:
  """Mean metric over `n_samples` sampled masks (one mask per full-split pass), no statistics updates."""
  net.eval()
  X, y = _tensors(split_, task)
  runs = [metrics(task, net(X, gate_generator=gen), y, target_std) for _ in range(n_samples)]
  return {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}


# ---------------------------------------------------------------- search

def search(net: ControlledGatedNet, data: Prepared, cfg: dict, seed: int) -> Dict:
  """TabGNS search loop (tabgns.tabgns.TabGNS.fit) with a real best-checkpoint snapshot.

  Per training batch: weight step on the train batch (sampled gates, the only pass that updates the
  means), a no-grad logging pass on a validation batch, then the gate step on that validation batch.
  After each epoch the fixed-mask validation loss selects the checkpoint and drives early stopping.
  """
  task, crit = data.task, criterion(data.task)
  w_opt = torch.optim.Adam(net.weight_parameters(), lr=cfg['weight_lr'], weight_decay=0.)
  a_opt = torch.optim.Adam(net.arch_parameters(), lr=cfg['arch_lr'], weight_decay=0.)
  train_dl = loader(data.train, task, cfg['batch_size'], generator(seed, 'train_order'))
  valid_dl = loader(data.valid, task, cfg['batch_size'], generator(seed, 'valid_order'))
  gate_gen = generator(seed, 'gate_noise')
  valid_iter = iter(valid_dl)

  best, best_state, best_epoch, counter = np.inf, None, -1, 0
  history = []
  start = time.perf_counter()
  for epoch in range(cfg['epochs']):
    train_loss = stoch_valid_loss = 0.
    for xb, yb in train_dl:
      try:
        xs, ys = next(valid_iter)
      except StopIteration:
        valid_iter = iter(valid_dl)
        xs, ys = next(valid_iter)

      net.train()
      w_opt.zero_grad()
      out = net(xb, gate_generator=gate_gen, update_stats=True)
      loss = crit(out, yb)
      loss.backward()
      w_opt.step()
      train_loss += loss.item()

      with torch.no_grad():
        net.eval()
        stoch_valid_loss += crit(net(xs, gate_generator=gate_gen), ys).item()

      a_opt.zero_grad()
      loss = crit(net(xs, gate_generator=gate_gen), ys)
      loss.backward()
      a_opt.step()

    net.eval()
    fixed_valid = evaluate(net, data.valid, task)
    fixed_valid_loss = fixed_valid['mse'] if task == 'regression' else fixed_valid['cross_entropy']
    finite = bool(np.isfinite(fixed_valid_loss) and np.isfinite(train_loss))
    p_open = torch.cat(net.p_open())
    history.append({
      'epoch': epoch,
      'widths': net.widths(),
      'train_loss': train_loss / len(train_dl),
      'stochastic_valid_loss': stoch_valid_loss / len(train_dl),
      'fixed_mask_valid_loss': fixed_valid_loss,
      'mean_p_open': float(p_open.mean()),
      'frac_p_open_uncertain': float(((p_open > 0.1) & (p_open < 0.9)).float().mean()),
    })
    if not finite:
      history[-1]['nonfinite'] = True
      break
    if fixed_valid_loss < best:
      best, best_epoch, counter = fixed_valid_loss, epoch, 0
      best_state = copy.deepcopy(net.state_dict())  # weights, gates and mean buffers
    else:
      counter += 1
      if counter >= cfg['patience']:
        break
  search_time = time.perf_counter() - start
  if best_state is not None:
    net.load_state_dict(best_state)
  return {'history': history, 'best_epoch': best_epoch, 'best_fixed_mask_valid_loss': float(best),
          'epochs_run': len(history), 'search_time_s': search_time,
          'stats_time_s': sum(b.stats_time_s for b in net.blocks),
          'n_stat_updates': [int(b.num_stat_updates) for b in net.blocks]}


# ---------------------------------------------------------------- fine-tuning

def finetune(model: nn.Module, data: Prepared, cfg: dict, seed: int) -> Dict:
  """AdamW (as tabgns.mlp.MLP.fit) with early stopping on validation loss and a deep-copied best snapshot."""
  task, crit = data.task, criterion(data.task)
  opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'])
  train_dl = loader(data.train, task, cfg['batch_size'], generator(seed, 'train_order'))
  Xv, yv = _tensors(data.valid, task)
  with torch.no_grad():
    model.eval()
    best = float(crit(model(Xv), yv))
  best_state, best_epoch, counter, epochs_run = copy.deepcopy(model.state_dict()), -1, 0, 0
  start = time.perf_counter()
  for epoch in range(cfg['epochs']):
    model.train()
    for xb, yb in train_dl:
      opt.zero_grad()
      loss = crit(model(xb), yb)
      loss.backward()
      opt.step()
    epochs_run += 1
    model.eval()
    with torch.no_grad():
      vl = float(crit(model(Xv), yv))
    if not np.isfinite(vl):
      break
    if vl < best:
      best, best_epoch, counter = vl, epoch, 0
      best_state = copy.deepcopy(model.state_dict())
    else:
      counter += 1
      if counter >= cfg['patience']:
        break
  model.load_state_dict(best_state)
  return {'finetune_time_s': time.perf_counter() - start, 'finetune_epochs_run': epochs_run,
          'finetune_best_epoch': best_epoch, 'finetune_best_valid_loss': best}
