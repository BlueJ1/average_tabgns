"""
TabNAS (Yang et al., NeurIPS 2022) re-implementation. The paper's authors also re-implemented it,
since no public code exists, so the details below follow the TabNAS paper and Sec. VI-A:

- One-shot SuperNet with weight sharing: a layer of width w uses the first w units of a 512-wide layer.
- A factorised categorical policy (one distribution per layer over the discrete width options),
  trained with REINFORCE and a moving-average baseline (momentum 0.9), RL learning rate 1e-3.
- The reward is the negative validation loss (MSE, or cross-entropy for classification): TabNAS uses
  logistic loss for both weight and RL updates (appendix C.2, footnote 4).
- The first quarter of the epochs only trains the shared weights, with architectures sampled from
  the (still uniform) policy.
- Rejection sampling with a Monte-Carlo estimate (2048 samples) of the feasible-set probability P(V).
  The paper disables the parameter constraint, so with `max_params: null` every architecture is
  feasible, P(V) = 1 and the correction term vanishes.
- Final architecture: the most likely architecture under the policy that fits the parameter budget
  (the most likely width of each layer when there is no budget).

Assumption (not specified in either paper): after the warm-up, the search stops early when the
validation loss of that most likely architecture has not improved for `stop_patience` epochs.
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset

from experiments.common import RunContext, Split, hidden_params, width_options

# Above this many width combinations the most likely feasible architecture is approximated by sampling.
_MAX_ENUMERATED = 50_000_000


class SharedMLP(nn.Module):
  """SuperNet whose sub-networks are the leading units of every layer."""
  def __init__(self, inp_dim: int, max_widths: List[int], out_dim: int):
    super().__init__()
    dims = [inp_dim, *max_widths]
    self.hidden = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(max_widths)))
    self.output = nn.Linear(max_widths[-1], out_dim)

  def forward(self, x: torch.Tensor, widths: List[int]) -> torch.Tensor:
    prev = x.shape[1]
    for layer, w in zip(self.hidden, widths):
      x = F.relu(F.linear(x, layer.weight[:w, :prev], layer.bias[:w]))
      prev = w
    return F.linear(x, self.output.weight[:, :prev], self.output.bias)


class Policy(nn.Module):
  def __init__(self, n_layers: int, options: List[int]):
    super().__init__()
    self.options = torch.tensor(options)
    self.logits = nn.Parameter(torch.zeros(n_layers, len(options)))

  def dist(self) -> Categorical:
    return Categorical(logits=self.logits)

  def widths(self, idx: torch.Tensor) -> List[int]:
    return self.options[idx].tolist()

  def argmax(self) -> List[int]:
    return self.widths(self.logits.argmax(dim=1))

  @torch.no_grad()
  def most_likely_feasible(self, max_params: Optional[int]) -> List[int]:
    """Most probable architecture with at most `max_params` hidden-to-hidden weights."""
    best = self.argmax()
    if max_params is None or hidden_params(best) <= max_params:
      return best
    logp = torch.log_softmax(self.logits, dim=1).numpy()
    opts = self.options.numpy().astype(np.int64)
    n_layers, k = logp.shape
    if k ** n_layers > _MAX_ENUMERATED:
      return self._most_likely_sampled(max_params)
    # Build the joint log-probability and parameter count of every combination one layer at a time.
    score, params = logp[0], np.zeros(k, dtype=np.int64)
    for l in range(1, n_layers):
      prev = opts.reshape((1,) * (l - 1) + (k, 1))
      score = score[..., None] + logp[l]
      params = params[..., None] + prev * opts
    score = np.where(params <= max_params, score, -np.inf)
    flat = int(np.argmax(score))
    if not np.isfinite(score.flat[flat]):
      raise RuntimeError(f'No architecture in the search space fits max_params={max_params}')
    return self.widths(torch.tensor(np.unravel_index(flat, score.shape)))

  def _most_likely_sampled(self, max_params: int, n_samples: int = 100_000) -> List[int]:
    dist = self.dist()
    idx = dist.sample((n_samples,))
    feasible = torch.tensor([hidden_params(self.widths(i)) <= max_params for i in idx])
    if not feasible.any():
      raise RuntimeError(f'No sampled architecture fits max_params={max_params}')
    logp = dist.log_prob(idx).sum(-1).masked_fill(~feasible, -np.inf)
    return self.widths(idx[logp.argmax()])


def _sample_feasible(policy: Policy, max_params: Optional[int], max_tries: int = 10_000) -> torch.Tensor:
  for _ in range(max_tries):
    idx = policy.dist().sample()
    if max_params is None or hidden_params(policy.widths(idx)) <= max_params:
      return idx
  raise RuntimeError('Could not sample an architecture within the parameter budget')


def _log_feasible_prob(policy: Policy, max_params: Optional[int], mc_samples: int) -> torch.Tensor:
  """Differentiable Monte-Carlo estimate of log P(V) used by TabNAS' rejection-sampling gradient."""
  if max_params is None:
    return torch.zeros(())
  dist = policy.dist()
  idx = dist.sample((mc_samples,))
  logp = dist.log_prob(idx).sum(-1)
  feasible = torch.tensor([hidden_params(policy.widths(i)) <= max_params for i in idx], dtype=torch.float32)
  return torch.log(torch.mean(torch.exp(logp - logp.detach()) * feasible).clamp_min(1e-12))


def _loss_fn(task: str):
  return nn.MSELoss() if task == 'regression' else nn.CrossEntropyLoss()


@torch.no_grad()
def _valid_loss(net: SharedMLP, widths: List[int], X: torch.Tensor, y: torch.Tensor, criterion: nn.Module) -> float:
  net.eval()
  out = torch.cat([net(X[i:i + 8192], widths) for i in range(0, len(X), 8192)])
  net.train()
  return criterion(out, y).item()


def search_tabnas(ctx: RunContext, train: Split, valid: Split, mcfg: dict) -> Tuple[List[int], float, list]:
  device = ctx.device
  options = width_options(mcfg['width_options'], max(ctx.search_widths))
  y_dtype = torch.float32 if ctx.task == 'regression' else torch.long
  bs = ctx.general['batch_size']
  train_loader = DataLoader(TensorDataset(torch.tensor(train[0]), torch.tensor(train[1], dtype=y_dtype)),
                            batch_size=bs, shuffle=True)
  valid_loader = DataLoader(TensorDataset(torch.tensor(valid[0]), torch.tensor(valid[1], dtype=y_dtype)),
                            batch_size=bs, shuffle=True)
  X_valid_full = torch.tensor(valid[0], device=device)
  y_valid_full = torch.tensor(valid[1], dtype=y_dtype, device=device)

  net = SharedMLP(ctx.inp_dim, list(ctx.search_widths), ctx.out_dim).to(device)
  policy = Policy(len(ctx.search_widths), options)
  weight_opt = torch.optim.Adam(net.parameters(), lr=ctx.general['weight_lr'])
  policy_opt = torch.optim.Adam(policy.parameters(), lr=mcfg['rl_lr'])
  criterion = _loss_fn(ctx.task)
  max_params = mcfg.get('max_params')

  epochs = mcfg.get('search_epochs') or ctx.general['iters']
  warmup = int(epochs * mcfg['warmup_fraction'])
  patience = ctx.general['stop_patience']
  baseline: Optional[float] = None
  best, best_widths, bad_epochs = np.inf, policy.most_likely_feasible(max_params), 0
  history = []

  start = time.perf_counter()
  valid_iter = iter(valid_loader)
  for epoch in range(epochs):
    for xb, yb in train_loader:
      xb, yb = xb.to(device), yb.to(device)
      # Shared-weight update on a sampled sub-network.
      widths = policy.widths(_sample_feasible(policy, max_params))
      weight_opt.zero_grad()
      criterion(net(xb, widths), yb).backward()
      weight_opt.step()

      if epoch < warmup:
        continue
      # Policy update with REINFORCE on a validation batch.
      try:
        xv, yv = next(valid_iter)
      except StopIteration:
        valid_iter = iter(valid_loader)
        xv, yv = next(valid_iter)
      idx = _sample_feasible(policy, max_params)
      with torch.no_grad():
        reward = -criterion(net(xv.to(device), policy.widths(idx)), yv.to(device)).item()
      baseline = reward if baseline is None else mcfg['baseline_momentum'] * baseline + (1 - mcfg['baseline_momentum']) * reward
      log_prob = policy.dist().log_prob(idx).sum() - _log_feasible_prob(policy, max_params, mcfg['mc_samples'])
      policy_opt.zero_grad()
      (-(reward - baseline) * log_prob).backward()
      policy_opt.step()

    current = policy.most_likely_feasible(max_params)
    history.append(current)
    if epoch < warmup:
      continue
    val = _valid_loss(net, current, X_valid_full, y_valid_full, criterion)
    if val < best:
      best, best_widths, bad_epochs = val, current, 0
    else:
      bad_epochs += 1
      if bad_epochs >= patience:
        break

  return list(best_widths), time.perf_counter() - start, history
