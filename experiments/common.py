"""
Shared helpers: train an MLP of given widths with the paper's protocol, predict, score, count parameters.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

from tabgns.mlp import MLP
from tabgns.utils import param_count

Split = Tuple[np.ndarray, np.ndarray]


@dataclass
class RunContext:
  task: str
  inp_dim: int
  out_dim: int
  device: str
  seed: int
  search_widths: List[int]
  # Training protocol from Sec. VI-A: lr 1e-3, max 300 epochs, early stopping patience 20.
  general: dict = field(default_factory=dict)

  def mlp_config(self, widths: List[int], epochs: Optional[int] = None, device: Optional[str] = None) -> dict:
    general = {**self.general,
               'problem_type': self.task,
               'inp_dim': self.inp_dim,
               'out_dim': self.out_dim,
               'device': device or self.device}
    if epochs is not None:
      general['iters'] = epochs
    return {'general': general, 'layer_widths': list(widths)}


def hidden_params(widths: List[int]) -> int:
  """Parameter count as reported in the paper: hidden-to-hidden weights only (5x512 -> 1,048,576)."""
  return param_count(list(widths))


def full_params(inp_dim: int, widths: List[int], out_dim: int) -> int:
  """All weights and biases, including input and output layers."""
  dims = [inp_dim, *widths, out_dim]
  return sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))


def train_mlp(ctx: RunContext, widths: List[int], train: Split, valid: Split,
              init: Optional[MLP] = None, epochs: Optional[int] = None,
              device: Optional[str] = None) -> Tuple[MLP, float]:
  """Trains an MLP (from scratch, or warm-started from `init`) and returns it with its training time."""
  config = ctx.mlp_config(widths, epochs=epochs, device=device)
  if init is not None:
    model = init
    model.config = copy.deepcopy(config)
  else:
    model = MLP(config)
  start = time.perf_counter()
  model.fit(train[0], train[1], valid[0], valid[1])
  return model, time.perf_counter() - start


@torch.no_grad()
def predict(model: torch.nn.Module, X: np.ndarray, device: str, batch_size: int = 8192) -> np.ndarray:
  """Raw outputs (logits for classification). MLP.predict only supports classification."""
  model = model.to(device).eval()
  outs = []
  for i in range(0, len(X), batch_size):
    xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
    outs.append(model(xb).cpu())
  return torch.cat(outs).numpy()


def score(task: str, y_true: np.ndarray, outputs: np.ndarray) -> float:
  """Test metric of Table II: MSE for regression, accuracy in percent for classification."""
  if task == 'regression':
    return float(np.mean((outputs - y_true.reshape(outputs.shape)) ** 2))
  return float(np.mean(np.argmax(outputs, axis=1) == y_true) * 100)


def objective(task: str, y_true: np.ndarray, outputs: np.ndarray) -> float:
  """Validation objective that the NAS baselines maximise: -MSE or accuracy."""
  s = score(task, y_true, outputs)
  return -s if task == 'regression' else s


def width_options(options: List[int], max_width: int) -> List[int]:
  """Discrete widths for TabNAS/AgE, clipped to the search space (smaller spaces in smoke tests)."""
  opts = sorted({int(o) for o in options if o <= max_width} | {max_width})
  return opts
