"""
Gated network for the controlled zero-gating vs. mean-replacement comparison, and its extraction.

Differences from tabgns/gates.py + tabgns/tabgns.py (both arms use these, the original stays as is):
- A gate is open with probability sigmoid(s) during the search (binary Concrete / Gumbel-softmax with
  logits [s, 0]), and the deterministic mask keeps s >= 0, i.e. sigmoid(s) >= 0.5. The original samples
  with logits [s, 1] (p_open = sigmoid(s - 1)) but extracts with sigmoid(s) >= 0.5.
- Gate noise comes from an explicit torch.Generator, so nothing else (data order, mean bookkeeping)
  shifts the gate samples.
- Extraction never scales weights by gate probabilities (the original's `scale=true`).

gate_mode='zero':  h_tilde = g * h
gate_mode='mean':  h_tilde = g * h + (1 - g) * mu,  mu = detached running mean of the post-ReLU,
                   pre-gate activation h, updated only by passes called with update_stats=True.
"""
from __future__ import annotations

import time
import warnings
from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from tabgns.gates import GatedBlock
from tabgns.mlp import MLP

GATE_MODES = ('zero', 'mean')


def sample_gates(strengths: Tensor, temperature: float, generator: torch.Generator) -> Tensor:
  """Hard straight-through binary Concrete sample, P(open) = sigmoid(strengths).

  Same distribution and gradient as F.gumbel_softmax(stack([s, 0]), hard=True)[:, 0]: the difference
  of two Gumbels is logistic noise. One mask per call, shared by the whole minibatch, as in the original.
  """
  u = torch.rand(strengths.shape, generator=generator, device=generator.device)
  u = u.clamp(1e-6, 1 - 1e-6).to(strengths.device)
  soft = torch.sigmoid((strengths + torch.log(u) - torch.log1p(-u)) / temperature)
  hard = (soft > 0.5).to(soft.dtype)
  return hard + soft - soft.detach()


class ControlledGatedBlock(GatedBlock):
  """GatedBlock (same parameters, same creation order) with a fixed-mask path and optional mean replacement."""

  def __init__(self, in_features: int, out_features: int, gate_initial_value: float, temperature: float,
               gate_mode: str, mean_momentum: float):
    super().__init__(in_features, out_features, gate_initial_value, temperature)
    assert gate_mode in GATE_MODES, gate_mode
    self.gate_mode = gate_mode
    self.mean_momentum = mean_momentum
    # Both arms carry the buffers so checkpoints have the same keys; only the mean arm updates or uses them.
    self.register_buffer('running_mean', torch.zeros(out_features))
    self.register_buffer('num_stat_updates', torch.zeros((), dtype=torch.long))
    self.stats_time_s = 0.0  # wall time spent in _update_stats (not saved)

  def fixed_mask(self) -> Tensor:
    return (self.strengths >= 0).to(self.strengths.dtype)

  @torch.no_grad()
  def _update_stats(self, h: Tensor) -> None:
    # EMA with BatchNorm's default momentum, initialised from the first training batch instead of zeros.
    t0 = time.perf_counter()
    batch_mean = h.detach().mean(dim=0)
    if int(self.num_stat_updates) == 0:
      self.running_mean.copy_(batch_mean)
    else:
      self.running_mean.mul_(1 - self.mean_momentum).add_(self.mean_momentum * batch_mean)
    self.num_stat_updates += 1
    self.stats_time_s += time.perf_counter() - t0

  def forward(self, x: Tensor, gates: Optional[Tensor] = None, update_stats: bool = False) -> Tensor:
    """`gates`: a (sampled or fixed) gate vector; None means the deterministic fixed mask."""
    h = self.activation(self.linear(x))
    if update_stats and self.gate_mode == 'mean':
      self._update_stats(h)
    g = self.fixed_mask() if gates is None else gates
    if self.gate_mode == 'mean':
      return g * h + (1 - g) * self.running_mean
    return g * h


class ControlledGatedNet(nn.Module):
  """Mirrors tabgns.tabgns.GatedNet (blocks, then output layer) so a seed gives the same initial weights."""

  def __init__(self, inp_dim: int, out_dim: int, layer_widths: List[int], gate_initial_value: float,
               temperature: float, gate_mode: str, mean_momentum: float = 0.1):
    super().__init__()
    self.gate_mode = gate_mode
    self.temperature = temperature
    blocks, d = [], inp_dim
    for w in layer_widths:
      blocks.append(ControlledGatedBlock(d, w, gate_initial_value, temperature, gate_mode, mean_momentum))
      d = w
    self.blocks = nn.ModuleList(blocks)
    self.output_layer = nn.Linear(d, out_dim)

  def forward(self, x: Tensor, gate_generator: Optional[torch.Generator] = None,
              update_stats: bool = False, masks: Optional[List[Tensor]] = None) -> Tensor:
    """Stochastic gates if `gate_generator` is given, explicit `masks` if given, else the fixed mask."""
    for i, block in enumerate(self.blocks):
      if masks is not None:
        gates = masks[i]
      elif gate_generator is not None:
        gates = sample_gates(block.strengths, self.temperature, gate_generator)
      else:
        gates = None
      x = block(x, gates=gates, update_stats=update_stats)
    return self.output_layer(x)

  def arch_parameters(self) -> List[nn.Parameter]:
    return [b.strengths for b in self.blocks]

  def weight_parameters(self) -> List[nn.Parameter]:
    params = []
    for b in self.blocks:
      params.extend(b.weight_parameters())
    return params + list(self.output_layer.parameters())

  def fixed_masks(self) -> List[Tensor]:
    return [b.fixed_mask().detach() for b in self.blocks]

  def widths(self) -> List[int]:
    return [int(m.sum()) for m in self.fixed_masks()]

  def p_open(self) -> List[Tensor]:
    return [torch.sigmoid(b.strengths).detach() for b in self.blocks]


@torch.no_grad()
def extract_mlp(net: ControlledGatedNet, inp_dim: int, out_dim: int, problem_type: str = 'regression',
                masks: Optional[List[Tensor]] = None) -> MLP:
  """Copies the open neurons of `net` (fixed mask unless `masks` is given) into a tabgns.mlp.MLP.

  For the mean arm, each removed neuron j of layer l contributes the constant mu_j to layer l+1, so
  it is folded into the next layer's bias before the column is dropped:
      b_next' = b_next + W_next[:, D] @ mu_D.
  This reproduces the fixed-mask, mean-replaced network up to floating point. No probability scaling.
  """
  net = net.cpu()
  masks = [m.cpu() for m in (masks if masks is not None else net.fixed_masks())]
  widths = [int(m.sum()) for m in masks]
  with warnings.catch_warnings():  # zero-width layers are allowed; torch warns that their init is a no-op
    warnings.filterwarnings('ignore', message='Initializing zero-element tensors')
    subnet = MLP({'general': {'inp_dim': inp_dim, 'out_dim': out_dim, 'problem_type': problem_type, 'device': 'cpu'},
                  'layer_widths': widths})
  targets = [m for m in subnet.net if isinstance(m, nn.Linear)]
  sources = [b.linear for b in net.blocks] + [net.output_layer]

  prev_keep = torch.arange(inp_dim)
  prev_drop = torch.empty(0, dtype=torch.long)
  prev_mean: Optional[Tensor] = None
  for li, (src, trg) in enumerate(zip(sources, targets)):
    is_output = li == len(net.blocks)
    rows = torch.arange(src.out_features) if is_output else torch.nonzero(masks[li]).flatten()
    weight = src.weight.detach()[rows]
    bias = src.bias.detach()[rows].clone()
    if net.gate_mode == 'mean' and prev_mean is not None and len(prev_drop):
      bias += weight[:, prev_drop].double() @ prev_mean[prev_drop].double()
    trg.weight.data = weight[:, prev_keep].clone()
    trg.bias.data = bias.to(trg.weight.dtype)
    if not is_output:
      block = net.blocks[li]
      prev_keep = rows
      prev_drop = torch.nonzero(masks[li] == 0).flatten()
      prev_mean = block.running_mean.detach()
  return subnet
