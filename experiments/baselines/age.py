"""
Aging Evolution (AgE), the NAS half of AgEBO-tabular (Egele et al., 2021), without the Bayesian
hyperparameter optimisation, as in the paper.

The paper ran DeepHyper's implementation on 20 parallel CPU workers. This is a self-contained
asynchronous version with the same settings (population 100, sample size 10, 300 evaluations) on
the paper's MLP search space: every layer picks one of the discrete widths, and a mutation
resamples the width of one random layer. Every candidate is trained fully (lr 1e-3, max 300 epochs,
early stopping patience 20) and scored on the validation split.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import List, Optional, Tuple

import numpy as np
import torch

from experiments.common import RunContext, Split, objective, predict, train_mlp, width_options

_WORKER_STATE: Optional[tuple] = None


def _init_worker(ctx: RunContext, train: Split, valid: Split, threads: int) -> None:
  global _WORKER_STATE
  torch.set_num_threads(threads)
  _WORKER_STATE = (ctx, train, valid)


def _evaluate(widths: List[int], seed: int) -> Tuple[List[int], float, float]:
  ctx, train, valid = _WORKER_STATE
  torch.manual_seed(seed)
  model, train_time = train_mlp(ctx, widths, train, valid, device='cpu')
  return widths, objective(ctx.task, valid[1], predict(model, valid[0], 'cpu')), train_time


def search_age(ctx: RunContext, train: Split, valid: Split, mcfg: dict) -> Tuple[List[int], float, list]:
  rng = np.random.default_rng(ctx.seed)
  options = width_options(mcfg['width_options'], max(ctx.search_widths))
  n_layers = len(ctx.search_widths)
  population_size, sample_size = mcfg['population_size'], mcfg['sample_size']
  iterations, n_workers = mcfg['iterations'], mcfg['n_workers']

  population: deque = deque(maxlen=population_size)  # oldest member drops out first
  history: list = []

  def propose(n_in_flight: int) -> List[int]:
    if len(history) + n_in_flight < population_size or len(population) < sample_size:
      return [int(rng.choice(options)) for _ in range(n_layers)]
    candidates = [population[i] for i in rng.choice(len(population), size=sample_size, replace=False)]
    child = list(max(candidates, key=lambda c: c['objective'])['widths'])
    layer = int(rng.integers(n_layers))
    child[layer] = int(rng.choice([o for o in options if o != child[layer]]))
    return child

  def record(result) -> None:
    widths, obj, train_time = result
    entry = {'widths': list(widths), 'objective': obj, 'train_time_s': train_time}
    history.append(entry)
    population.append(entry)

  start = time.perf_counter()
  if n_workers <= 1:
    _init_worker(ctx, train, valid, torch.get_num_threads())
    for i in range(iterations):
      record(_evaluate(propose(0), ctx.seed * 100_003 + i))
  else:
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context('spawn'),
                             initializer=_init_worker, initargs=(ctx, train, valid, 1)) as pool:
      pending, submitted = set(), 0
      while submitted < min(n_workers, iterations):
        pending.add(pool.submit(_evaluate, propose(len(pending)), ctx.seed * 100_003 + submitted))
        submitted += 1
      while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
          record(future.result())
          if submitted < iterations:
            pending.add(pool.submit(_evaluate, propose(len(pending)), ctx.seed * 100_003 + submitted))
            submitted += 1
  search_time = time.perf_counter() - start

  best = max(history, key=lambda h: h['objective'])
  return best['widths'], search_time, history
