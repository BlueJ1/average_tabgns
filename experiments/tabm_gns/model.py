"""
TabM (Gorishniy et al., ICLR 2025) and a gated TabM supernet for TabGNS.

TabM here is the `arch_type='tabm'` variant of the reference package (tabm 0.0.3), written out so
that every layer can be gated and pruned:

  x (B, d_in) -> (B, k, d_in) -> [BatchEnsemble linear -> ReLU -> Dropout] x n_blocks
              -> k independent linear heads -> (B, k, d_out)

A BatchEnsemble linear layer shares one weight matrix W between the k members and gives each member
its own input scaling r_i, output scaling s_i and bias b_i:  y_i = s_i * (W (r_i * x_i)) + b_i.
As in TabM, the first layer's r is initialised with random signs (one sign per input feature, so the
one-hot columns of a categorical feature share a sign) and every other r and s with ones; the k biases
start equal.

The gated supernet puts one TabGNS gate on every hidden neuron, after ReLU and dropout. A gate is
shared by all k members, because pruning a neuron removes a row of the shared W: the architecture of
a TabM is one list of widths for the whole ensemble. The gate is the same as in `tabgns/gates.py`
(hard Gumbel softmax over the logits [g, 1]). `noise='shared'` draws one mask per forward pass for the
whole ensemble, like TabGNS draws one per pass for its MLP; `noise='member'` draws an independent mask
for each member, so one step sees k different sub-networks of the same gate distribution.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _init_rsqrt_uniform_(t: Tensor, d: int) -> Tensor:
  b = d ** -0.5
  return nn.init.uniform_(t, -b, b)


@torch.no_grad()
def _random_signs_(t: Tensor, chunks: Optional[Sequence[int]] = None) -> Tensor:
  """Random +-1, one value per chunk of the last dimension (per input feature, as in TabM)."""
  if chunks is None:
    return t.bernoulli_(0.5).mul_(2).add_(-1)
  assert sum(chunks) == t.shape[-1], (chunks, t.shape)
  start = 0
  for c in chunks:
    t[..., start:start + c] = torch.empty(*t.shape[:-1], 1).bernoulli_(0.5).mul_(2).add_(-1)
    start += c
  return t


class LinearBatchEnsemble(nn.Module):
  """k linear layers that share W:  y_i = s_i * (W (r_i * x_i)) + b_i.  (B, k, d_in) -> (B, k, d_out)."""

  def __init__(self, d_in: int, d_out: int, k: int, r_init: str = 'ones',
               r_chunks: Optional[Sequence[int]] = None):
    super().__init__()
    self.in_features, self.out_features, self.k = d_in, d_out, k
    self.weight = nn.Parameter(torch.empty(d_out, d_in))
    self.r = nn.Parameter(torch.empty(k, d_in))
    self.s = nn.Parameter(torch.empty(k, d_out))
    self.bias = nn.Parameter(torch.empty(k, d_out))
    self.r_init, self.r_chunks = r_init, r_chunks
    self.reset_parameters()

  def reset_parameters(self) -> None:
    _init_rsqrt_uniform_(self.weight, max(self.in_features, 1))
    if self.r_init == 'random-signs':
      _random_signs_(self.r.data, self.r_chunks)
    else:
      nn.init.ones_(self.r)
    nn.init.ones_(self.s)
    b = _init_rsqrt_uniform_(torch.empty(self.out_features), max(self.in_features, 1))
    with torch.no_grad():
      self.bias.copy_(b.expand(self.k, -1))  # all k biases start equal, as in TabM

  def forward(self, x: Tensor) -> Tensor:
    return (x * self.r) @ self.weight.T * self.s + self.bias


class LinearEnsemble(nn.Module):
  """k fully independent linear layers (TabM's output heads).  (B, k, d_in) -> (B, k, d_out)."""

  def __init__(self, d_in: int, d_out: int, k: int):
    super().__init__()
    self.in_features, self.out_features, self.k = d_in, d_out, k
    self.weight = nn.Parameter(torch.empty(k, d_in, d_out))
    self.bias = nn.Parameter(torch.empty(k, d_out))
    _init_rsqrt_uniform_(self.weight, max(d_in, 1))
    _init_rsqrt_uniform_(self.bias, max(d_in, 1))

  def forward(self, x: Tensor) -> Tensor:
    return torch.einsum('bki,kio->bko', x, self.weight) + self.bias


class TabM(nn.Module):
  """Plain TabM with given hidden widths. Returns per-member outputs (B, k, d_out)."""

  def __init__(self, d_in: int, widths: Sequence[int], d_out: int, k: int = 32, dropout: float = 0.0,
               first_r_chunks: Optional[Sequence[int]] = None):
    super().__init__()
    self.d_in, self.widths, self.d_out, self.k = d_in, list(widths), d_out, k
    self.dropout = dropout
    self.first_r_chunks = list(first_r_chunks) if first_r_chunks is not None else None
    dims = [d_in, *self.widths]
    self.linears = nn.ModuleList([
      LinearBatchEnsemble(dims[i], dims[i + 1], k, r_init='random-signs' if i == 0 else 'ones',
                          r_chunks=self.first_r_chunks if i == 0 else None)
      for i in range(len(self.widths))])
    self.drop = nn.Dropout(dropout)
    self.head = LinearEnsemble(dims[-1], d_out, k)

  def forward(self, x: Tensor) -> Tensor:
    x = x.unsqueeze(1).expand(-1, self.k, -1)
    for lin in self.linears:
      x = self.drop(F.relu(lin(x)))
    return self.head(x)


class TabMGate(nn.Module):
  """TabGNS gate (tabgns.gates.BinaryGumbelSoftmax) for one layer of a TabM."""

  def __init__(self, d: int, k: int, init: float, temperature: float, noise: str):
    super().__init__()
    assert noise in ('shared', 'member')
    self.k, self.temperature, self.noise = k, temperature, noise
    self.logits = nn.Parameter(init * torch.ones(d))
    # When set, forward uses this fixed 0/1 mask instead of sampling (for tests and evaluation).
    self.fixed_mask: Optional[Tensor] = None

  def sample(self) -> Tensor:
    g = self.logits if self.noise == 'shared' else self.logits.expand(self.k, -1)
    two = torch.stack([g, torch.ones_like(g)], dim=-1)  # P(open) = sigmoid(g - 1), as in tabgns
    return F.gumbel_softmax(two, tau=self.temperature, hard=True)[..., 0]

  def forward(self, x: Tensor) -> Tensor:  # x: (B, k, d); mask (d,) or (k, d) broadcasts
    if self.fixed_mask is not None:
      return x * self.fixed_mask
    return x * self.sample()

  def p_open(self) -> Tensor:
    """sigmoid(g): the quantity that extraction thresholds at 0.5 (as in tabgns)."""
    return torch.sigmoid(self.logits)


class GatedTabM(TabM):
  """TabM supernet with a TabGNS gate on every hidden neuron."""

  def __init__(self, d_in: int, widths: Sequence[int], d_out: int, k: int, dropout: float,
               gate_init: float, temperature: float, noise: str = 'shared',
               first_r_chunks: Optional[Sequence[int]] = None):
    super().__init__(d_in, widths, d_out, k, dropout, first_r_chunks)
    self.gates = nn.ModuleList([TabMGate(w, k, gate_init, temperature, noise) for w in self.widths])

  def forward(self, x: Tensor) -> Tensor:
    x = x.unsqueeze(1).expand(-1, self.k, -1)
    for lin, gate in zip(self.linears, self.gates):
      x = gate(self.drop(F.relu(lin(x))))
    return self.head(x)

  def arch_parameters(self) -> List[nn.Parameter]:
    return [g.logits for g in self.gates]

  def weight_parameters(self) -> List[nn.Parameter]:
    return [p for n, p in self.named_parameters() if not n.startswith('gates.')]

  def keep_masks(self) -> List[Tensor]:
    return [(g.p_open() >= 0.5).detach().cpu() for g in self.gates]

  def curr_architecture(self) -> List[int]:
    return [int(m.sum()) for m in self.keep_masks()]

  @torch.no_grad()
  def extract(self, scale: bool) -> TabM:
    """The TabM of the kept neurons (sigmoid(g) >= 0.5), with the supernet's weights.

    With `scale`, a kept neuron's outgoing scale s and bias are multiplied by sigmoid(g), like
    `TabGNS.extract_net` scales the rows of W and b; since sigmoid(g) > 0 this is the same as scaling
    the neuron's output after ReLU.
    """
    masks = self.keep_masks()
    sub = TabM(self.d_in, [int(m.sum()) for m in masks], self.d_out, self.k, self.dropout,
               self.first_r_chunks)
    cols = torch.arange(self.d_in)
    for src, trg, gate, keep in zip(self.linears, sub.linears, self.gates, masks):
      rows = torch.nonzero(keep).flatten()
      p = gate.p_open().detach().cpu()[rows] if scale else torch.ones(len(rows))
      trg.weight.copy_(src.weight.detach().cpu()[rows][:, cols])
      trg.r.copy_(src.r.detach().cpu()[:, cols])
      trg.s.copy_(src.s.detach().cpu()[:, rows] * p)
      trg.bias.copy_(src.bias.detach().cpu()[:, rows] * p)
      cols = rows
    sub.head.weight.copy_(self.head.weight.detach().cpu()[:, cols, :])
    sub.head.bias.copy_(self.head.bias.detach().cpu())
    return sub


def tabm_params(d_in: int, widths: Sequence[int], d_out: int, k: int) -> dict:
  """Parameter counts of a TabM.

  hidden: shared hidden-to-hidden weights only, the paper's MLP count (5x512 MLP -> 1,048,576).
  shared: all shared weight matrices (input layer included).
  full:   every trainable parameter: shared W, the k members' r, s, biases, and the k heads.
  """
  dims = [d_in, *widths]
  hidden = sum(widths[i] * widths[i + 1] for i in range(len(widths) - 1))
  shared = sum(dims[i] * dims[i + 1] for i in range(len(widths)))
  adapters = sum(k * (dims[i] + 2 * dims[i + 1]) for i in range(len(widths)))
  heads = k * (dims[-1] * d_out + d_out)
  return {'hidden': hidden, 'shared': shared, 'full': shared + adapters + heads}


def mlp_params(d_in: int, widths: Sequence[int], d_out: int) -> dict:
  dims = [d_in, *widths, d_out]
  hidden = sum(widths[i] * widths[i + 1] for i in range(len(widths) - 1))
  full = sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))
  return {'hidden': hidden, 'shared': full, 'full': full}
