"""
One (dataset, seed) cell of the mean-gate study: both arms (zero gating, mean replacement) on the same
split, initial weights, gate initialisation and random streams, plus a full-width dense reference.
Writes <results_dir>/<experiment>/<dataset>/seed<seed>.json. Hydra multirun gives the grid, e.g.

  python -m experiments.mean_gate.run -m experiment=mean_gate_pilot dataset=friedman_reg,synthetic_cls_4k seed=0,1,2
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, cast

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from experiments.common import full_params, hidden_params
from experiments.mean_gate.model import ControlledGatedNet, extract_mlp
from experiments.mean_gate.protocol import (STREAM_OFFSETS, evaluate, evaluate_stochastic, finetune, generator,
                                            make_raw_dataset, prepare, search)
from tabgns.mlp import MLP
from tabgns.utils import set_seed

ROOT = Path(__file__).resolve().parents[2]


def provenance() -> Dict[str, Any]:
  def git(*args):
    try:
      return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except Exception as e:  # not a git checkout, git missing, ...
      return f'<unavailable: {e}>'
  import hydra as hydra_mod, sklearn
  src = sorted((ROOT / 'experiments' / 'mean_gate').glob('*.py')) + [ROOT / 'tabgns' / 'gates.py', ROOT / 'tabgns' / 'mlp.py']
  return {
    'git_head': git('rev-parse', 'HEAD').strip(),
    'git_status_porcelain': git('status', '--porcelain'),
    'git_diff_head_sha256': hashlib.sha256(git('diff', 'HEAD').encode()).hexdigest(),
    'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in src},
    'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
    'cpu_count': os.cpu_count(), 'torch_threads': torch.get_num_threads(),
    'versions': {'torch': torch.__version__, 'numpy': np.__version__, 'sklearn': sklearn.__version__,
                 'hydra': hydra_mod.__version__},
  }


def _param_digest(module: torch.nn.Module) -> str:
  h = hashlib.sha256()
  for name, p in module.named_parameters():
    h.update(name.encode())
    h.update(p.detach().cpu().numpy().tobytes())
  return h.hexdigest()


def _all_inputs(data) -> torch.Tensor:
  return torch.tensor(np.concatenate([data.train[0], data.valid[0], data.test[0]]), dtype=torch.float32)


def run_arm(arm: str, data, cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
  s, task, sd = cfg['search'], data.task, data.target_std
  torch.manual_seed(seed)
  net = ControlledGatedNet(data.inp_dim, data.out_dim, list(s['layer_widths']), s['gate_initialisation_val'],
                           s['temperature'], gate_mode=arm, mean_momentum=s['mean_momentum'])
  rec: Dict[str, Any] = {'arm': arm, 'init_param_sha256': _param_digest(net)}

  rec['search'] = search(net, data, {**s, 'batch_size': cfg['general']['batch_size'],
                                     'weight_lr': cfg['general']['weight_lr']}, seed)
  net.eval()
  widths = net.widths()
  rec['widths'] = widths
  rec['zero_width_layers'] = [i for i, w in enumerate(widths) if w == 0]
  rec['n_params'] = hidden_params(widths)
  rec['n_params_full'] = full_params(data.inp_dim, widths, data.out_dim)
  rec['p_open'] = [p.tolist() for p in net.p_open()]
  rec['running_mean'] = [b.running_mean.tolist() for b in net.blocks]

  # Deterministic fixed-mask network (what extraction should reproduce) and the stochastic network.
  rec['fixed_mask'] = {k: evaluate(net, getattr(data, k), task, sd) for k in ('valid', 'test')}
  stoch_gen = generator(seed, 'stochastic_eval')
  rec['stochastic'] = {k: evaluate_stochastic(net, getattr(data, k), task, cfg['eval']['stochastic_samples'],
                                              stoch_gen, sd) for k in ('valid', 'test')}

  t0 = time.perf_counter()
  subnet = extract_mlp(net, data.inp_dim, data.out_dim, task)
  rec['extraction_time_s'] = time.perf_counter() - t0
  X = _all_inputs(data)
  with torch.no_grad():
    subnet.eval()
    ref, ext = net(X), subnet(X)
    rec['extraction_max_abs_diff'] = float((ref - ext).abs().max())
    rec['extraction_max_rel_diff'] = float(((ref - ext).abs() / ref.abs().clamp_min(1e-6)).max())
  rec['extracted'] = {k: evaluate(subnet, getattr(data, k), task, sd) for k in ('valid', 'test')}
  if arm == 'mean':
    # Diagnostic only: delete closed neurons without folding mu into the next bias.
    naive_net = copy.deepcopy(net)
    naive_net.gate_mode = 'zero'
    naive = extract_mlp(naive_net, data.inp_dim, data.out_dim, task)
    rec['extracted_without_bias_fold'] = {k: evaluate(naive, getattr(data, k), task, sd) for k in ('valid', 'test')}

  ft_cfg = {**cfg['finetune'], 'batch_size': cfg['general']['batch_size']}
  rec['finetune'] = finetune(subnet, data, ft_cfg, seed)
  rec['finetuned'] = {k: evaluate(subnet, getattr(data, k), task, sd) for k in ('valid', 'test')}
  rec['nonfinite'] = any(not np.isfinite(v) for part in ('fixed_mask', 'extracted', 'finetuned')
                         for split_ in rec[part].values() for v in split_.values())
  return rec


def run_dense_reference(data, cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
  widths = list(cfg['search']['layer_widths'])
  torch.manual_seed(seed)
  model = MLP({'general': {'inp_dim': data.inp_dim, 'out_dim': data.out_dim, 'problem_type': data.task,
                           'device': 'cpu'}, 'layer_widths': widths})
  ft = finetune(model, data, {**cfg['finetune'], 'batch_size': cfg['general']['batch_size']}, seed)
  return {'widths': widths, 'n_params': hidden_params(widths),
          'n_params_full': full_params(data.inp_dim, widths, data.out_dim), 'finetune': ft,
          'finetuned': {k: evaluate(model, getattr(data, k), data.task, data.target_std) for k in ('valid', 'test')}}


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
  c = cast(Dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
  out_path = Path(cfg.results_dir) / cfg.experiment / cfg.dataset.name / f'seed{cfg.seed}.json'
  if out_path.exists() and not cfg.overwrite:
    print(f'{out_path} exists, skipping (overwrite=true to rerun)')
    return
  torch.set_num_threads(cfg.torch_threads)
  set_seed(cfg.seed)

  ds = make_raw_dataset(c['dataset'])
  data = prepare(ds, split_seed=cfg.seed)
  record: Dict[str, Any] = {
    'experiment': cfg.experiment, 'dataset': ds.name, 'task': ds.task, 'seed': cfg.seed,
    'split_seed': cfg.seed, 'model_seed': cfg.seed, 'data_seed': c['dataset']['data_seed'],
    'stream_seeds': {k: cfg.seed * 1000 + v for k, v in STREAM_OFFSETS.items()},
    'split_sizes': {k: len(getattr(data, k)[0]) for k in ('train', 'valid', 'test')},
    'target_mean': None if data.target_mean is None else data.target_mean.tolist(),
    'target_std': None if data.target_std is None else data.target_std.tolist(),
    'config': c, 'provenance': provenance(), 'arms': {}, 'failures': {},
  }
  for arm in c['arms']:
    try:
      record['arms'][arm] = run_arm(arm, data, c, cfg.seed)
    except Exception:  # keep going, but record it: failed runs are results too
      record['failures'][arm] = traceback.format_exc()
      print(record['failures'][arm], file=sys.stderr)
  if c['dense_reference']:
    record['dense_reference'] = run_dense_reference(data, c, cfg.seed)

  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(record, indent=1))
  for arm, r in record['arms'].items():
    key = 'mse_orig' if ds.task == 'regression' else 'accuracy'
    print(f"{ds.name} seed {cfg.seed} {arm:4s}: widths={r['widths']} best_epoch={r['search']['best_epoch']} "
          f"fixed={r['fixed_mask']['test'][key]:.4f} extracted={r['extracted']['test'][key]:.4f} "
          f"finetuned={r['finetuned']['test'][key]:.4f} max|diff|={r['extraction_max_abs_diff']:.2e}")
  print(f'-> {out_path}')


if __name__ == '__main__':
  main()
