"""
Runs one (experiment, dataset, method, seed) cell and writes its result to
results/<experiment>/<dataset>/<method>/<run_name>.json. Use Hydra multirun for grids, e.g.

  python -m experiments.run -m experiment=table2 dataset=vod,sim_a method=large_mlp,tabgns,tabnas,age seed=0,1,2,3,4

See experiments/README.md for the command behind every table and figure of the paper.
"""
from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any, Dict, cast

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # needed by torch.use_deterministic_algorithms on CUDA

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from experiments.common import RunContext, full_params, hidden_params, predict, score
from experiments.datasets import load_dataset, split
from experiments.methods import METHODS
from tabgns.utils import set_seed


def _device_name(device: str) -> str:
  if device.startswith('cuda') and torch.cuda.is_available():
    return torch.cuda.get_device_name(torch.device(device))
  return platform.processor() or platform.machine()


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
  cfg_dict = cast(Dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
  out_path = Path(cfg.results_dir) / cfg.experiment / cfg.dataset.name / cfg.method.name / f'{cfg.run_name}.json'
  if out_path.exists() and not cfg.overwrite:
    print(f'{out_path} exists, skipping (overwrite=true to rerun)')
    return

  set_seed(cfg.seed)
  ds = load_dataset(cfg_dict['dataset'])
  train, valid, test = split(ds, seed=cfg.seed)
  ctx = RunContext(task=ds.task, inp_dim=ds.inp_dim, out_dim=ds.out_dim, device=cfg.device, seed=cfg.seed,
                   search_widths=list(cfg.search_space.layer_widths), general=cfg_dict['general'])

  result = METHODS[cfg.method.name](ctx, train, valid, cfg_dict['method'])
  widths = result['widths']
  test_metric = score(ds.task, test[1], predict(result['model'], test[0], cfg.device))

  record = {
    'experiment': cfg.experiment,
    'dataset': ds.name,
    'method': cfg.method.name,
    'seed': cfg.seed,
    'run_name': cfg.run_name,
    'task': ds.task,
    'metric': 'mse' if ds.task == 'regression' else 'accuracy',
    'test_metric': test_metric,
    'widths': widths,
    'n_params': hidden_params(widths),
    'n_params_full': full_params(ds.inp_dim, widths, ds.out_dim),
    'search_time_s': result['search_time_s'],
    'train_time_s': result['train_time_s'],
    # AgE always searches on CPU workers, whatever `device` says.
    'search_device': 'cpu' if cfg.method.name == 'age' else cfg.device.split(':')[0],
    'device_name': _device_name(cfg.device),
    'history': result.get('history') if cfg.save_history else None,
    'config': cfg_dict,
  }
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(record, indent=2))
  print(f'{ds.name} / {cfg.method.name} / seed {cfg.seed}: {record["metric"]}={test_metric:.4f}, '
        f'widths={widths}, params={record["n_params"]:,}, search={record["search_time_s"]:.1f}s -> {out_path}')


if __name__ == '__main__':
  main()
