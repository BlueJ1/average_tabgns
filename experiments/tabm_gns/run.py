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
STREAMS = {'init': 0, 'data': 1, 'gates': 2, 'finetune_data': 3, 'scratch_init': 4, 'scratch_data': 5,
           'uniform_init': 6}


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


def _supernet(data: Prepared, cfg: Dict, seed: int) -> GatedTabM:
  """The gated supernet at initialisation. Same init stream as tabm_large, so it starts from exactly the
  Large TabM's weights (gates are created last and draw no random numbers)."""
  t, m = cfg['tabm'], cfg['method']
  torch.manual_seed(10_000 * seed + STREAMS['init'])
  return GatedTabM(data.inp_dim, t['widths'], data.out_dim, t['k'], t['dropout'], m['gate_initialisation_val'],
                   m['temperature'], m['noise'], data.feature_chunks)


def _search(data: Prepared, cfg: Dict, seed: int, tag: str):
  net = _supernet(data, cfg, seed)
  torch.manual_seed(10_000 * seed + STREAMS['gates'])  # Gumbel noise comes from the global RNG
  return net, search(net, data, cfg['method'], cfg['train'], gen(seed, 'data'), tag=tag)


def run_tabm_gns(data: Prepared, cfg: Dict, seed: int) -> Dict:
  t, m = cfg['tabm'], cfg['method']
  tag = f"{data.name}/{m['name']}/s{seed}"
  net, s = _search(data, cfg, seed, tag)
  widths = net.curr_architecture()
  rec: Dict[str, Any] = {'widths': widths, 'search': s,
                         'params': tabm_params(data.inp_dim, widths, data.out_dim, t['k']),
                         '_masks': [mk.numpy() for mk in net.keep_masks()]}

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


# ---------------------------------------------------------------------------------- controls
# For a finished TabGNS-TabM run (method tabm_gns or tabm_gns_member), two controls trained with the
# TabM protocol and the scratch run's data-order stream, so that they differ from the scratch run only
# in initialisation (rewind) or shape (uniform):
#   rewind:  the found widths, initialised with the supernet's *initial* weights of the kept neurons
#            (same seed and init stream as the supernet, sliced with the final masks). Warm start vs.
#            rewind = what the supernet's training contributes beyond a good shape and a good init.
#   uniform: equal widths in every layer, chosen so that the parameter count matches the found model.

CONTROL_NAMES = {'tabm_gns': ('tabm_rewind', 'tabm_uniform'),
                 'tabm_gns_member': ('tabm_rewind_member', 'tabm_uniform_member')}


def uniform_width(d_in: int, n_layers: int, d_out: int, k: int, target: int) -> int:
  return min(range(1, 4097), key=lambda w: abs(tabm_params(d_in, [w] * n_layers, d_out, k)['full'] - target))


def run_rewind(data: Prepared, cfg: Dict, seed: int, masks) -> Dict:
  t = cfg['tabm']
  model = _supernet(data, cfg, seed).extract(scale=False, masks=masks)
  widths = model.widths
  train = train_tabm(model, data, cfg['train'], gen(seed, 'scratch_data'),
                     tag=f"{data.name}/{cfg['method']['name']} rewind/s{seed}")
  return {'widths': widths, 'train': train, 'metrics': _eval_all(model, data, _tabm_predict(data.task)),
          'params': tabm_params(data.inp_dim, widths, data.out_dim, t['k'])}


def run_uniform(data: Prepared, cfg: Dict, seed: int, target: int) -> Dict:
  t = cfg['tabm']
  w = uniform_width(data.inp_dim, len(t['widths']), data.out_dim, t['k'], target)
  widths = [w] * len(t['widths'])
  torch.manual_seed(10_000 * seed + STREAMS['uniform_init'])
  model = TabM(data.inp_dim, widths, data.out_dim, t['k'], t['dropout'], data.feature_chunks)
  train = train_tabm(model, data, cfg['train'], gen(seed, 'scratch_data'),
                     tag=f"{data.name}/{cfg['method']['name']} uniform/s{seed}")
  return {'widths': widths, 'train': train, 'metrics': _eval_all(model, data, _tabm_predict(data.task)),
          'params': tabm_params(data.inp_dim, widths, data.out_dim, t['k']), 'target_params_full': target}


def run_controls(data: Prepared, gcfg: Dict, seed: int, gns_out: Path, masks, target: int,
                 do_rewind: bool = True, do_uniform: bool = True) -> None:
  """Runs the controls of the TabGNS-TabM result at gns_out and writes them next to it."""
  variant = gcfg['method']['name']
  for name, do, fn in ((CONTROL_NAMES[variant][0], do_rewind and masks is not None,
                        lambda: run_rewind(data, gcfg, seed, masks)),
                       (CONTROL_NAMES[variant][1], do_uniform, lambda: run_uniform(data, gcfg, seed, target))):
    out = gns_out.parent.parent / name / gns_out.name
    if not do or is_done(out):
      continue
    start = time.perf_counter()
    rec = _base_record(gcfg, data, name)
    rec['control_of'] = variant
    try:
      rec.update(fn())
    except Exception:
      rec['error'] = traceback.format_exc()
      print(rec['error'], file=sys.stderr)
    _write(rec, gcfg, out, start)


def run_tabm_controls(data: Prepared, c: Dict, seed: int, out: Path) -> Dict:
  """Controls for a TabGNS-TabM run that finished without saved masks: replays its search (deterministic
  on one CPU thread) to recover the masks, checks the replay against the stored histories, then trains
  the controls. Waits for the matched result if it is still being produced."""
  m = c['method']
  gout = out.parent.parent / m['match'] / out.name
  deadline = time.time() + 3600 * m['wait_hours']
  while not is_done(gout):
    if time.time() > deadline:
      raise TimeoutError(f'{gout} did not appear')
    time.sleep(60)
  stored = json.loads(gout.read_text())
  if 'error' in stored:
    raise RuntimeError(f'{gout} failed')
  gcfg = stored['config']
  info: Dict[str, Any] = {'match': m['match'], 'matched_widths': stored['widths']}
  masks = None
  mpath = gout.with_suffix('.masks.npz')
  if m['rewind']:
    if mpath.exists():
      masks = list(np.load(mpath).values())
      info['masks'] = 'saved'
    else:
      net, s = _search(data, gcfg, seed, tag=f"{data.name}/{m['match']} replay/s{seed}")
      replay_widths = net.curr_architecture()
      info['replay'] = {
        'arch_history_equal': s['arch_history'] == stored['search']['arch_history'],
        'loss_history_equal': [list(x) for x in s['loss_history']] == [list(x) for x in stored['search']['loss_history']],
        'epochs': s['epochs'], 'stored_epochs': stored['search']['epochs'], 'widths': replay_widths}
      masks = [mk.numpy() for mk in net.keep_masks()]
      if replay_widths == stored['widths']:
        np.savez(mpath, *masks)
      else:
        info['replay']['warning'] = 'replay found different widths; rewind uses the replayed masks'
  run_controls(data, gcfg, seed, gout, masks, stored['params']['full'], do_rewind=m['rewind'])
  info['widths'] = stored['widths']
  return info


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


def is_done(out: Path) -> bool:
  """A result exists and is not a placeholder (placeholders only block superseded sweeps)."""
  if not out.exists():
    return False
  try:
    return not json.loads(out.read_text()).get('placeholder', False)
  except json.JSONDecodeError:
    return False


def _base_record(c: Dict, data: Prepared, method: str) -> Dict[str, Any]:
  return {'experiment': c['experiment'], 'dataset': data.name, 'method': method, 'seed': c['seed'],
          'task': data.task, 'data_info': data.info}


def _write(rec: Dict[str, Any], c: Dict, out: Path, start: float) -> None:
  rec['wall_time_s'] = time.perf_counter() - start
  rec['config'] = c
  rec['provenance'] = provenance()
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(rec, indent=1))
  mt = rec.get('metrics', {}).get('test')
  print(f"{rec['dataset']} / {rec['method']} / seed {rec['seed']}: test={mt} widths={rec.get('widths')} "
        f"time={rec['wall_time_s']:.0f}s -> {out}")


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
  c = cast(Dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
  method = c['method']['name']
  out = Path(c['results_dir']) / c['experiment'] / c['dataset']['name'] / method / f"seed{c['seed']}.json"
  if not out.is_absolute():
    out = ROOT / out
  if is_done(out) and not c['overwrite']:
    print(f'{out} exists, skipping')
    return
  torch.set_num_threads(c['torch_threads'])
  set_seed(c['seed'])
  data = prepare(c['dataset'], c['seed'])
  start = time.perf_counter()
  rec = _base_record(c, data, method)
  masks = None
  try:
    if method == 'tabm_large':
      rec.update(run_tabm_large(data, c, c['seed']))
    elif method.startswith('tabm_gns'):
      rec.update(run_tabm_gns(data, c, c['seed']))
      masks = rec.pop('_masks')
      out.parent.mkdir(parents=True, exist_ok=True)
      np.savez(out.with_suffix('.masks.npz'), *masks)
    elif method.startswith('tabm_controls'):
      rec.update(run_tabm_controls(data, c, c['seed'], out))
    elif method == 'mlp_large':
      rec.update(run_mlp(data, c, c['seed'], 'large'))
    elif method == 'mlp_gns':
      rec.update(run_mlp(data, c, c['seed'], 'gns'))
    else:
      raise ValueError(method)
  except Exception:
    rec['error'] = traceback.format_exc()
    print(rec['error'], file=sys.stderr)
  rec.pop('_masks', None)
  _write(rec, c, out, start)
  if masks is not None and 'error' not in rec and c['method'].get('controls', True):
    run_controls(data, c, c['seed'], out, masks, rec['params']['full'])


if __name__ == '__main__':
  main()
