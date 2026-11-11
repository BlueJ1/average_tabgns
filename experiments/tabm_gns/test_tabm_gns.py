"""
Correctness checks for the TabGNS -> TabM transfer:  python -m unittest -v experiments.tabm_gns.test_tabm_gns
"""
from __future__ import annotations

import copy
import math
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from experiments.tabm_gns.model import GatedTabM, TabM, tabm_params
from experiments.tabm_gns.protocol import (Prepared, ensemble_loss, ensemble_output, member_loss, metrics,
                                           search)


def _net(noise='shared', k=4, widths=(16, 12, 8), d_in=7, d_out=3, seed=0, gate_init=0.3):
  torch.manual_seed(seed)
  net = GatedTabM(d_in, list(widths), d_out, k, 0.0, gate_init, 1.0, noise)
  with torch.no_grad():  # spread the gate logits so that some neurons are kept and some are dropped
    for g in net.gates:
      g.logits.normal_(0, 1.5)
    for lin in net.linears:  # non-trivial adapters
      lin.r.normal_(1, 0.3)
      lin.s.normal_(1, 0.3)
      lin.bias.normal_(0, 0.3)
  return net


class TestTabM(unittest.TestCase):

  def test_matches_reference_package(self):
    try:
      import tabm
    except ImportError:
      self.skipTest('tabm not installed')
    torch.manual_seed(0)
    ref = tabm.TabM.make(n_num_features=7, d_out=3, n_blocks=3, d_block=16, k=4, dropout=0.0)
    ours = TabM(7, [16, 16, 16], 3, k=4)
    with torch.no_grad():
      for i, lin in enumerate(ours.linears):
        src = ref.backbone.blocks[i][0]
        lin.weight.copy_(src.weight); lin.r.copy_(src.r); lin.s.copy_(src.s); lin.bias.copy_(src.bias)
      ours.head.weight.copy_(ref.output.weight); ours.head.bias.copy_(ref.output.bias)
    ref.eval(); ours.eval()
    x = torch.randn(5, 7)
    torch.testing.assert_close(ours(x), ref(x_num=x), rtol=1e-5, atol=1e-6)
    self.assertEqual(sum(p.numel() for p in ours.parameters()), sum(p.numel() for p in ref.parameters()))

  def test_param_count(self):
    m = TabM(10, [8, 6], 3, k=4)
    self.assertEqual(tabm_params(10, [8, 6], 3, 4)['full'], sum(p.numel() for p in m.parameters()))
    self.assertEqual(tabm_params(10, [8, 6], 3, 4)['hidden'], 48)

  def test_first_layer_signs_per_feature_chunk(self):
    torch.manual_seed(1)
    m = TabM(6, [8], 2, k=16, first_r_chunks=[1, 2, 3])
    r = m.linears[0].r.detach()
    self.assertTrue(set(r.unique().tolist()) <= {-1.0, 1.0})
    self.assertTrue(torch.equal(r[:, 1], r[:, 2]) and torch.equal(r[:, 3], r[:, 5]))

  def test_supernet_starts_from_large_tabm_weights(self):
    torch.manual_seed(3)
    large = TabM(7, [16, 12], 3, k=4, dropout=0.1)
    torch.manual_seed(3)
    sup = GatedTabM(7, [16, 12], 3, 4, 0.1, -3, 1.0)
    for (n, a), (_, b) in zip(large.named_parameters(), sup.weight_parameters() and
                              [(n, p) for n, p in sup.named_parameters() if not n.startswith('gates.')]):
      self.assertTrue(torch.equal(a, b), n)


class TestExtraction(unittest.TestCase):

  def _check(self, noise, scale):
    net = _net(noise)
    masks = net.keep_masks()
    self.assertTrue(all(0 < m.sum() < len(m) for m in masks), 'test needs partially open layers')
    for g, m in zip(net.gates, masks):
      g.fixed_mask = m.float() * (g.p_open().detach() if scale else 1.)
    net.eval()
    x = torch.randn(9, 7)
    sub = net.extract(scale=scale).eval()
    self.assertEqual(sub.widths, [int(m.sum()) for m in masks])
    torch.testing.assert_close(sub(x), net(x), rtol=1e-5, atol=1e-6)

  def test_extract_exact_unscaled(self):
    self._check('shared', False)

  def test_extract_exact_scaled(self):
    self._check('shared', True)

  def test_extract_exact_member_noise(self):
    self._check('member', True)


class TestGates(unittest.TestCase):

  def test_mask_shapes(self):
    torch.manual_seed(0)
    shared, member = _net('shared', k=8), _net('member', k=8)
    self.assertEqual(tuple(shared.gates[0].sample().shape), (16,))
    m = member.gates[0].sample()
    self.assertEqual(tuple(m.shape), (8, 16))
    self.assertFalse(all(torch.equal(m[0], m[i]) for i in range(1, 8)), 'members should differ')

  def test_open_probability_follows_tabgns(self):
    # tabgns.gates.BinaryGumbelSoftmax: logits [g, 1] -> P(open) = sigmoid(g - 1).
    torch.manual_seed(0)
    net = GatedTabM(3, [2000], 2, 4, 0.0, 0.5, 1.0, 'member')
    p = net.gates[0].sample().mean().item()
    self.assertAlmostEqual(p, 1 / (1 + math.exp(0.5)), delta=0.01)

  def test_gate_only_backward_matches_full_backward(self):
    net = _net('shared')
    x, y = torch.randn(11, 7), torch.randint(0, 3, (11,))
    torch.manual_seed(5)
    member_loss('classification', net(x), y).backward()
    full = [g.logits.grad.clone() for g in net.gates]
    net.zero_grad()
    torch.manual_seed(5)
    member_loss('classification', net(x), y).backward(inputs=net.arch_parameters())
    for a, g in zip(full, net.gates):
      torch.testing.assert_close(a, g.logits.grad)
    self.assertTrue(all(p.grad is None or not p.grad.any() for p in net.weight_parameters()))


class TestControls(unittest.TestCase):

  def test_rewind_takes_initial_weights_of_kept_neurons(self):
    torch.manual_seed(7)
    init = GatedTabM(7, [16, 12], 3, 4, 0.0, -3, 1.0)
    trained = copy.deepcopy(init)
    with torch.no_grad():
      for p in trained.parameters():
        p.add_(1.0)  # "training" moves every weight
    masks = [torch.rand(16) > 0.5, torch.rand(12) > 0.5]
    torch.manual_seed(7)
    fresh = GatedTabM(7, [16, 12], 3, 4, 0.0, -3, 1.0)
    sub = fresh.extract(scale=False, masks=masks)
    rows0, rows1 = torch.nonzero(masks[0]).flatten(), torch.nonzero(masks[1]).flatten()
    self.assertEqual(sub.widths, [len(rows0), len(rows1)])
    torch.testing.assert_close(sub.linears[0].weight, init.linears[0].weight[rows0])
    torch.testing.assert_close(sub.linears[1].weight, init.linears[1].weight[rows1][:, rows0])
    torch.testing.assert_close(sub.linears[1].r, init.linears[1].r[:, rows0])
    torch.testing.assert_close(sub.head.weight, init.head.weight[:, rows1, :])

  def test_uniform_width_matches_params(self):
    from experiments.tabm_gns.run import uniform_width
    target = tabm_params(93, [300, 310, 170], 9, 32)['full']
    w = uniform_width(93, 3, 9, 32, target)
    best = abs(tabm_params(93, [w] * 3, 9, 32)['full'] - target)
    self.assertTrue(all(best <= abs(tabm_params(93, [v] * 3, 9, 32)['full'] - target) for v in (w - 1, w + 1)))

  def test_search_is_deterministic(self):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 5)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    data = Prepared('toy', 'classification', (X[:240], y[:240]), (X[240:320], y[240:320]), (X[320:], y[320:]), 2, [1] * 5)
    runs = []
    for _ in range(2):
      torch.manual_seed(0)
      net = GatedTabM(5, [16, 16], 2, 4, 0.1, -1, 1.0, 'member')
      torch.manual_seed(1)
      s = search(net, data, {'arch_lr': 0.05, 'epochs': 4, 'patience': 5, 'checkpoint': 'last'},
                 {'lr': 2e-3, 'weight_decay': 3e-4, 'batch_size': 32}, torch.Generator().manual_seed(2))
      runs.append((s['arch_history'], s['loss_history'], [g.logits.detach().clone() for g in net.gates]))
    self.assertEqual(runs[0][0], runs[1][0])
    self.assertEqual(runs[0][1], runs[1][1])
    for a, b in zip(runs[0][2], runs[1][2]):
      self.assertTrue(torch.equal(a, b))


class TestLossesAndSearch(unittest.TestCase):

  def test_ensemble_output(self):
    out = torch.randn(5, 4, 3)
    probs = F.softmax(out, -1).mean(1)
    torch.testing.assert_close(ensemble_output('classification', out).exp(), probs)
    torch.testing.assert_close(ensemble_output('regression', out), out.mean(1))
    y = torch.randint(0, 3, (5,))
    torch.testing.assert_close(ensemble_loss('classification', out, y), F.nll_loss(probs.log(), y))

  def test_search_smoke(self):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 5)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] ** 2 > 0.5).astype(np.int64)
    data = Prepared('toy', 'classification', (X[:360], y[:360]), (X[360:480], y[360:480]),
                    (X[480:], y[480:]), 2, [1] * 5)
    torch.manual_seed(0)
    net = GatedTabM(5, [32, 32], 2, 4, 0.0, -3, 1.0, 'shared')
    train_cfg = {'lr': 2e-3, 'weight_decay': 3e-4, 'batch_size': 64}
    s = search(net, data, {'arch_lr': 0.05, 'epochs': 15, 'patience': 5, 'checkpoint': 'last'}, train_cfg,
               torch.Generator().manual_seed(0))
    self.assertEqual(len(s['arch_history']), s['epochs'])
    self.assertTrue(all(np.isfinite(l[1]) for l in s['loss_history']))
    m = metrics('classification', np.log(np.full((120, 2), 0.5)), y[480:], data)
    self.assertAlmostEqual(m['logloss'], math.log(2), places=6)


if __name__ == '__main__':
  unittest.main()
