"""
One entry point per compared method. Each returns the final model, its widths, and timing/history.
"""
from __future__ import annotations

import copy
import time
from typing import Callable, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.common import RunContext, Split, train_mlp
from tabgns.tabgns import TabGNS


def run_large_mlp(ctx: RunContext, train: Split, valid: Split, mcfg: dict) -> dict:
  """Naive baseline: the largest architecture in the search space, trained directly."""
  model, train_time = train_mlp(ctx, ctx.search_widths, train, valid)
  return {'model': model, 'widths': list(ctx.search_widths), 'search_time_s': 0.0, 'train_time_s': train_time}


def _tabgns_loaders(ctx: RunContext, train: Split, valid: Split):
  def to_tensors(X, y):
    if ctx.task == 'classification':
      # TabGNS.fit and utils.accuracy expect one-hot targets for classification.
      y = np.eye(ctx.out_dim, dtype=np.float32)[y]
    return TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
  bs = ctx.general['batch_size']
  return (DataLoader(to_tensors(*train), batch_size=bs, shuffle=True),
          DataLoader(to_tensors(*valid), batch_size=bs, shuffle=True))


def run_tabgns(ctx: RunContext, train: Split, valid: Split, mcfg: dict) -> dict:
  """Search with gated neurons, extract gates with p_open >= 0.5, warm-start and fine-tune (Sec. VI-A)."""
  config = {
    'general': {**ctx.general,
                'problem_type': ctx.task,
                'criterion': 'mse' if ctx.task == 'regression' else 'cross_entropy',
                'inp_dim': ctx.inp_dim,
                'out_dim': ctx.out_dim,
                'device': ctx.device},
    'search_space': {'layer_widths': list(ctx.search_widths)},
    'tabgns': {k: mcfg[k] for k in ('temperature', 'arch_lr', 'gate_initialisation_val', 'scale')},
  }
  train_loader, valid_loader = _tabgns_loaders(ctx, train, valid)

  start = time.perf_counter()
  search = TabGNS(copy.deepcopy(config))
  subnet = search.fit(train_loader, valid_loader)
  search_time = time.perf_counter() - start
  widths = list(subnet.layer_widths)

  # The paper fine-tunes "for a small number of additional epochs"; the exact number is not given.
  # Default: same epoch cap and early stopping as every other model (what TabGNSTrainer does).
  model, train_time = train_mlp(ctx, widths, train, valid, init=subnet, epochs=mcfg.get('finetune_epochs'))
  return {'model': model, 'widths': widths, 'search_time_s': search_time, 'train_time_s': train_time,
          'history': search.arch_hist}


def run_tabnas(ctx: RunContext, train: Split, valid: Split, mcfg: dict) -> dict:
  from experiments.baselines.tabnas import search_tabnas
  widths, search_time, history = search_tabnas(ctx, train, valid, mcfg)
  model, train_time = train_mlp(ctx, widths, train, valid)  # retrained from random init
  return {'model': model, 'widths': widths, 'search_time_s': search_time, 'train_time_s': train_time,
          'history': history}


def run_age(ctx: RunContext, train: Split, valid: Split, mcfg: dict) -> dict:
  from experiments.baselines.age import search_age
  widths, search_time, history = search_age(ctx, train, valid, mcfg)
  model, train_time = train_mlp(ctx, widths, train, valid)  # retrained from random init
  return {'model': model, 'widths': widths, 'search_time_s': search_time, 'train_time_s': train_time,
          'history': history}


METHODS: Dict[str, Callable[..., dict]] = {
  'large_mlp': run_large_mlp,
  'tabgns': run_tabgns,
  'tabnas': run_tabnas,
  'age': run_age,
}
