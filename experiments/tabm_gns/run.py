"""
One (dataset, method, seed) cell of the TabGNS -> TabM transfer study. Writes
<results_dir>/<experiment>/<dataset>/<method>/seed<seed>.json and skips cells that already have one.

  python -m experiments.tabm_gns.run -m hydra/launcher=joblib hydra.launcher.n_jobs=6 \
      dataset=california,adult,otto method=tabm_large,tabm_gns,tabm_gns_member,mlp_large,mlp_gns seed=0,1,2

Methods (see REPORT.md):
  tabm_large       TabM with the full search space (TabM's default 3x512, k=32), trained directly
  tabm_gns         TabGNS on a gated TabM (one gate mask per pass for the whole ensemble), extract,
                   warm-start fine-tune; the found widths are also trained from scratch for comparison
  tabm_gns_member  same, but every ensemble member draws its own gate mask
  mlp_large        the replication's Large MLP (5x512), unchanged code, same split and preprocessing
  mlp_gns          the replication's TabGNS (tabgns/), unchanged code, same split and preprocessing
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

from experiments.tabm_gns.model import GatedTabM, TabM, mlp_params, tabm_params
from experiments.tabm_gns.protocol import Prepared, metrics, predict, prepare, search, train_tabm
from tabgns.utils import set_seed

ROOT = Path(__file__).resolve().parents[2]

# Separate random streams so that, within a seed, data order and gate noise do not depend on each other.
STREAMS = {'init': 0, 'data': 1, 'gates': 2, 'finetune_data': 3, 'scratch_init': 4, 'scratch_data': 5}


def gen(seed: int, stream: str) -> torch.Generator:
  return torch.Generator().manual_seed(10_000 * seed + STREAMS[stream])


def provenance() -> Dict[str, Any]:
  def git(*args):
    try:
      return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except Exception as e:
      return f'<unavailable: {e}>'
  src = sorted((ROOT / 'experiments' / 'tabm_gns').glob('*.py'))
  return {
    'git_head': git('rev-parse', 'HEAD').strip(),
    'git_dirty': bool(git('status', '--porcelain', '--', 'experiments', 'tabgns').strip()),
    'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in src},
    'python': sys.version.split()[0], 'platform': platform.platform(), 'torch': torch.__version__,
    'torch_threads': torch.get_num_threads(),
  }


def _eval_all(model, data: Prepared, predict_fn) -> Dict[str, Dict[str, float]]:
  return {name: metrics(data.task, predict_fn(model, split_[0]), split_[1], data)
          for name, split_ in (('valid', data.valid), ('test', data.test))}


def _tabm_predict(task):
  return lambda m, X: predict(m, X, task)


def run_tabm_large(data: Prepared, cfg: Dict, seed: int) -> Dict:
  t = cfg['tabm']
  torch.manual_seed(10_000 * seed + STREAMS['init'])
  model = TabM(data.inp_dim, t['widths'], data.out_dim, t['k'], t['dropout'], data.feature_chunks)
  train = train_tabm(model, data, cfg['train'], gen(seed, 'data'), tag=f'{data.name}/tabm_large/s{seed}')
  return {'widths': list(t['widths']), 'train': train, 'metrics': _eval_all(model, data, _tabm_predict(data.task)),
          'params': tabm_params(data.inp_dim, t['widths'], data.out_dim, t['k'])}


def run_tabm_gns(data: Prepared, cfg: Dict, seed: int) -> Dict:
  t, m = cfg['tabm'], cfg['method']
  # Same init stream as tabm_large: the supernet starts from the Large TabM's weights (gates come last).
  torch.manual_seed(10_000 * seed + STREAMS['init'])
  net = GatedTabM(data.inp_dim, t['widths'], data.out_dim, t['k'], t['dropout'], m['gate_initialisation_val'],
                  m['temperature'], m['noise'], data.feature_chunks)
  torch.manual_seed(10_000 * seed + STREAMS['gates'])  # Gumbel noise comes from the global RNG
  tag = f"{data.name}/{m['name']}/s{seed}"
  s = search(net, data, m, cfg['train'], gen(seed, 'data'), tag=tag)
  widths = net.curr_architecture()
  rec: Dict[str, Any] = {'widths': widths, 'search': s,
                         'params': tabm_params(data.inp_dim, widths, data.out_dim, t['k'])}

  sub = net.extract(scale=m['scale'])
  rec['metrics_extracted'] = _eval_all(sub, data, _tabm_predict(data.task))  # before fine-tuning
  if min(widths) == 0:
    rec['zero_width_layer'] = True
  rec['finetune'] = train_tabm(sub, data, cfg['train'], gen(seed, 'finetune_data'), tag=tag + ' finetune')
  rec['metrics'] = _eval_all(sub, data, _tabm_predict(data.task))

  if m.get('scratch_retrain', True) and min(widths) > 0:
    torch.manual_seed(10_000 * seed + STREAMS['scratch_init'])
    scratch = TabM(data.inp_dim, widths, data.out_dim, t['k'], t['dropout'], data.feature_chunks)
    rec['scratch_train'] = train_tabm(scratch, data, cfg['train'], gen(seed, 'scratch_data'), tag=tag + ' scratch')
    rec['metrics_scratch'] = _eval_all(scratch, data, _tabm_predict(data.task))
  return rec


def _mlp_ctx(data: Prepared, cfg: Dict, seed: int):
  from experiments.common import RunContext
  return RunContext(task=data.task, inp_dim=data.inp_dim, out_dim=data.out_dim, device='cpu', seed=seed,
                    search_widths=list(cfg['mlp']['widths']), general=dict(cfg['mlp']['general']))


def _mlp_predict(m, X):
  from experiments.common import predict as mlp_predict
  return mlp_predict(m, X, 'cpu')


def _mlp_targets(data: Prepared, split_):
  # MLP.fit / TabGNS take integer labels (classification) or float (n, d) targets (regression).
  return (split_[0], split_[1])


def run_mlp(data: Prepared, cfg: Dict, seed: int, which: str) -> Dict:
  from experiments.methods import run_large_mlp, run_tabgns
  ctx = _mlp_ctx(data, cfg, seed)
  set_seed(seed)
  fn = run_large_mlp if which == 'large' else run_tabgns
  res = fn(ctx, _mlp_targets(data, data.train), _mlp_targets(data, data.valid), cfg['method'])
  rec = {'widths': list(res['widths']), 'search': {'time_s': res['search_time_s'], 'arch_history': res.get('history')},
         'train': {'time_s': res['train_time_s']},
         'metrics': _eval_all(res['model'], data, _mlp_predict),
         'params': mlp_params(data.inp_dim, res['widths'], data.out_dim)}
  return rec


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
  c = cast(Dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
  method = c['method']['name']
  out = Path(c['results_dir']) / c['experiment'] / c['dataset']['name'] / method / f"seed{c['seed']}.json"
  if not out.is_absolute():
    out = ROOT / out
  if out.exists() and not c['overwrite']:
    print(f'{out} exists, skipping')
    return
  torch.set_num_threads(c['torch_threads'])
  set_seed(c['seed'])
  data = prepare(c['dataset'], c['seed'])
  start = time.perf_counter()
  rec: Dict[str, Any] = {'experiment': c['experiment'], 'dataset': data.name, 'method': method, 'seed': c['seed'],
                         'task': data.task, 'data_info': data.info}
  try:
    if method == 'tabm_large':
      rec.update(run_tabm_large(data, c, c['seed']))
    elif method.startswith('tabm_gns'):
      rec.update(run_tabm_gns(data, c, c['seed']))
    elif method == 'mlp_large':
      rec.update(run_mlp(data, c, c['seed'], 'large'))
    elif method == 'mlp_gns':
      rec.update(run_mlp(data, c, c['seed'], 'gns'))
    else:
      raise ValueError(method)
  except Exception:
    rec['error'] = traceback.format_exc()
    print(rec['error'], file=sys.stderr)
  rec['wall_time_s'] = time.perf_counter() - start
  rec['config'] = c
  rec['provenance'] = provenance()
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(rec, indent=1))
  mt = rec.get('metrics', {}).get('test')
  print(f"{data.name} / {method} / seed {c['seed']}: test={mt} widths={rec.get('widths')} "
        f"time={rec['wall_time_s']:.0f}s -> {out}")


if __name__ == '__main__':
  main()
