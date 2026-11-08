"""
Correctness checks for the mean-gate experiment path.

  python -m unittest -v experiments.mean_gate.test_mean_gate
"""
from __future__ import annotations

import copy
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

from experiments.datasets import Dataset
from experiments.mean_gate.model import ControlledGatedNet, extract_mlp, sample_gates
from experiments.mean_gate import report
from experiments.mean_gate.protocol import generator, prepare, search

RESULTS = Path(__file__).resolve().parents[2] / 'results' / 'mean_gate'


def make_net(mode: str, widths=(7, 5, 6), inp=4, out=3, seed=0) -> ControlledGatedNet:
  torch.manual_seed(seed)
  return ControlledGatedNet(inp, out, list(widths), gate_initial_value=0.5, temperature=1.0, gate_mode=mode)


def randomise(net: ControlledGatedNet, masks, seed=1) -> None:
  """Arbitrary biases, means and gate logits whose sign gives `masks` (dropout and scaling are not used)."""
  g = torch.Generator().manual_seed(seed)
  with torch.no_grad():
    for block, m in zip(net.blocks, masks):
      block.linear.bias.normal_(generator=g)
      block.running_mean.copy_(torch.rand(block.running_mean.shape, generator=g) * 2)
      block.strengths.copy_(torch.where(m > 0, 1.0, -1.0) * (0.1 + torch.rand(m.shape, generator=g)))
    net.output_layer.bias.normal_(generator=g)


class GateSemantics(unittest.TestCase):

  def test_open_and_closed_gates(self):
    for mode in ('zero', 'mean'):
      net = make_net(mode)
      block = net.blocks[0]
      with torch.no_grad():
        block.strengths.copy_(torch.tensor([5., -5., 0., -1e-4, 3., -3., 2.]))
        block.running_mean.copy_(torch.arange(7.) + 1)
      x = torch.randn(9, 4)
      h = torch.relu(block.linear(x))
      out = block(x)  # fixed mask: s >= 0 open
      is_open = torch.tensor([1, 0, 1, 0, 1, 0, 1], dtype=torch.bool)
      torch.testing.assert_close(out[:, is_open], h[:, is_open])
      closed = block.running_mean[~is_open].expand(9, -1) if mode == 'mean' else torch.zeros(9, 3)
      torch.testing.assert_close(out[:, ~is_open], closed)
      # Saturated logits make the stochastic gate deterministic, and the same as the fixed mask.
      with torch.no_grad():
        block.strengths.copy_(torch.where(is_open, 50., -50.))
      gates = sample_gates(block.strengths, 1.0, torch.Generator().manual_seed(0))
      torch.testing.assert_close(block(x, gates=gates), block(x))

  def test_sampling_threshold_matches_extraction_threshold(self):
    s = torch.tensor([-2., -0.5, 0., 0.5, 2.])
    gen = torch.Generator().manual_seed(0)
    freq = torch.stack([sample_gates(s, 1.0, gen).detach() for _ in range(20000)]).mean(0)
    torch.testing.assert_close(freq, torch.sigmoid(s), atol=0.015, rtol=0)
    net = make_net('zero', widths=(5,))
    with torch.no_grad():
      net.blocks[0].strengths.copy_(s)
    self.assertEqual(net.fixed_masks()[0].tolist(), [0., 0., 1., 1., 1.])  # sigmoid(s) >= 0.5

  def test_mean_with_zero_mu_equals_zero_gating(self):
    zero, mean = make_net('zero'), make_net('mean')
    x = torch.randn(11, 4)
    for gz, gm in ((torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)), (None, None)):
      torch.testing.assert_close(zero(x, gate_generator=gz), mean(x, gate_generator=gm))

  def test_gate_gradient_uses_centred_activation(self):
    net = make_net('mean', widths=(3,))
    block = net.blocks[0]
    with torch.no_grad():
      block.running_mean.copy_(torch.tensor([0.3, 0.7, 1.1]))
    x = torch.randn(5, 4)
    gates = sample_gates(block.strengths, 1.0, torch.Generator().manual_seed(0))
    gates.retain_grad()
    block(x, gates=gates).sum().backward()
    h = torch.relu(block.linear(x)).detach()
    torch.testing.assert_close(gates.grad, (h - block.running_mean).sum(0))


class MeanStatistics(unittest.TestCase):

  def test_update_rule(self):
    net = make_net('mean', widths=(6,))
    block = net.blocks[0]
    x1, x2 = torch.randn(8, 4), torch.randn(8, 4)
    net(x1, update_stats=True)
    m1 = torch.relu(block.linear(x1)).mean(0).detach()
    torch.testing.assert_close(block.running_mean, m1)  # initialised from the first batch
    net(x2, update_stats=True)
    m2 = torch.relu(block.linear(x2)).mean(0).detach()
    torch.testing.assert_close(block.running_mean, 0.9 * m1 + 0.1 * m2)
    self.assertEqual(int(block.num_stat_updates), 2)
    self.assertFalse(block.running_mean.requires_grad)

  def test_eval_and_validation_passes_leave_means_unchanged(self):
    net = make_net('mean')
    net(torch.randn(8, 4), update_stats=True)
    before = copy.deepcopy(dict(net.named_buffers()))
    x = torch.randn(16, 4)
    for train_mode in (True, False):
      net.train(train_mode)
      net(x)
      net(x, gate_generator=torch.Generator().manual_seed(0))
    for k, v in net.named_buffers():
      torch.testing.assert_close(v, before[k], rtol=0, atol=0)

  def test_search_updates_means_on_training_steps_only(self):
    rng = np.random.default_rng(0)
    ds = Dataset('toy', 'regression', rng.normal(size=(200, 4)), rng.normal(size=(200, 1)).astype(np.float32))
    data = prepare(ds, split_seed=0)
    net = make_net('mean', widths=(8, 8), inp=4, out=1)
    cfg = {'weight_lr': 1e-3, 'arch_lr': 0.05, 'batch_size': 32, 'epochs': 3, 'patience': 10}
    res = search(net, data, cfg, seed=0)
    steps_per_epoch = -(-len(data.train[0]) // 32)
    # The selected checkpoint's counter: one update per training batch up to the best epoch.
    self.assertEqual(res['n_stat_updates'], [steps_per_epoch * (res['best_epoch'] + 1)] * 2)

  def test_checkpoint_round_trip(self):
    net = make_net('mean')
    net(torch.randn(8, 4), update_stats=True)
    buf = io.BytesIO()
    torch.save(net.state_dict(), buf)
    buf.seek(0)
    other = make_net('mean', seed=5)
    other.load_state_dict(torch.load(buf, weights_only=True))
    for (k, v), (_, w) in zip(net.state_dict().items(), other.state_dict().items()):
      torch.testing.assert_close(v, w, rtol=0, atol=0, msg=k)


class Pairing(unittest.TestCase):

  def test_arms_share_initial_weights_and_gate_samples(self):
    zero, mean = make_net('zero'), make_net('mean')
    for (k, a), (_, b) in zip(zero.named_parameters(), mean.named_parameters()):
      torch.testing.assert_close(a, b, rtol=0, atol=0, msg=k)
    gz, gm = generator(0, 'gate_noise'), generator(0, 'gate_noise')
    x = torch.randn(8, 4)
    zero(x, gate_generator=gz)
    mean(x, gate_generator=gm, update_stats=True)  # bookkeeping must not consume gate noise
    self.assertTrue(torch.equal(torch.rand(5, generator=gz), torch.rand(5, generator=gm)))


class Extraction(unittest.TestCase):

  def check(self, mode, masks, widths=(7, 5, 6)):
    net = make_net(mode, widths=widths)
    randomise(net, masks)
    net.eval()
    sub = extract_mlp(net, 4, 3).eval()
    x = torch.randn(64, 4)
    with torch.no_grad():
      ref, out = net(x), sub(x)
    self.assertEqual(sub.layer_widths, [int(m.sum()) for m in masks])
    self.assertLess(float((ref - out).abs().max()), 1e-5)
    return net, sub, x

  def test_mixed_masks_three_layers(self):
    masks = [torch.tensor([1, 0, 1, 1, 0, 0, 1.]), torch.tensor([0, 1, 1, 0, 1.]), torch.tensor([1, 1, 0, 0, 0, 1.])]
    for mode in ('zero', 'mean'):
      with self.subTest(mode=mode):
        self.check(mode, masks)

  def test_bias_fold_is_needed(self):
    masks = [torch.tensor([1, 0, 1, 1, 0, 0, 1.]), torch.tensor([0, 1, 1, 0, 1.]), torch.tensor([1, 1, 0, 0, 0, 1.])]
    net, _, x = self.check('mean', masks)
    naive = copy.deepcopy(net)
    naive.gate_mode = 'zero'
    with torch.no_grad():
      self.assertGreater(float((net(x) - extract_mlp(naive, 4, 3)(x)).abs().max()), 1e-2)

  def test_fully_removed_hidden_layer(self):
    for closed_layer in range(3):
      masks = [torch.tensor([1, 0, 1, 1, 0, 0, 1.]), torch.tensor([0, 1, 1, 0, 1.]), torch.tensor([1, 1, 0, 0, 0, 1.])]
      masks[closed_layer] = torch.zeros_like(masks[closed_layer])
      for mode in ('zero', 'mean'):
        with self.subTest(mode=mode, layer=closed_layer):
          _, sub, x = self.check(mode, masks)
          with torch.no_grad():  # nothing downstream of an empty layer depends on the input
            self.assertLess(float(sub(x).std(0).max()), 1e-6)

  def test_all_open(self):
    masks = [torch.ones(7), torch.ones(5), torch.ones(6)]
    for mode in ('zero', 'mean'):
      self.check(mode, masks)


@unittest.skipUnless((RESULTS / 'pilot' / 'friedman_reg' / 'seed0.json').exists(), 'needs the saved pilot results')
class ReportGrouping(unittest.TestCase):
  """A second protocol next to `pilot` must never be pooled with it (PR #3 review)."""

  def setUp(self):
    self.root = Path(tempfile.mkdtemp())
    self.addCleanup(shutil.rmtree, self.root)
    for seed in range(3):
      self.write('pilot', seed)

  def write(self, experiment: str, seed: int, finetune_epochs: int = 100, name: str = None):
    r = json.loads((RESULTS / 'pilot' / 'friedman_reg' / f'seed{seed}.json').read_text())
    r['experiment'] = experiment
    r['config']['experiment'] = experiment
    r['config']['finetune']['epochs'] = finetune_epochs
    out = self.root / experiment / 'friedman_reg' / (name or f'seed{seed}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r))

  def test_protocols_stay_separate(self):
    for seed in range(3):
      self.write('pilot_long_ft', seed, finetune_epochs=300)
    rows = [l for l in report.compact(report.load(self.root)) if 'friedman_reg' in l]
    self.assertEqual(len(rows), 2)
    for row in rows:
      self.assertIn('| 0–2 |', row)
      self.assertRegex(row, r'\| \d/3 \|')

  def test_mixed_protocol_under_one_name_is_rejected(self):
    self.write('pilot', 3, finetune_epochs=300)
    with self.assertRaisesRegex(ValueError, 'Different protocols'):
      report.load(self.root)

  def test_duplicate_seed_is_rejected(self):
    self.write('pilot', 0, name='seed0_copy.json')
    with self.assertRaisesRegex(ValueError, 'Duplicate'):
      report.load(self.root)


if __name__ == '__main__':
  unittest.main()
