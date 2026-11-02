"""
Builds the paper's tables and figures from the JSON results written by experiments.run.

  python -m experiments.report --results results --out reports

Writes table2.md, table3.md, headline.md and fig2/fig4/fig5 PNGs to --out. Missing cells are skipped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

METHODS = ['large_mlp', 'tabnas', 'age', 'tabgns']
METHOD_LABELS = {'large_mlp': 'Large MLP', 'tabnas': 'TabNAS', 'age': 'AgE', 'tabgns': 'TabGNS'}
DATASETS = ['vod', 'deepmimo', 'sim_a', 'sim_b', 'covertype', 'higgs']


def load_results(root: Path) -> pd.DataFrame:
  rows = [json.loads(p.read_text()) for p in root.rglob('*.json')]
  if not rows:
    raise SystemExit(f'No results under {root}')
  df = pd.DataFrame(rows)
  df['gate_init'] = df['config'].map(lambda c: c['method'].get('gate_initialisation_val'))
  return df


def _fmt(values: np.ndarray, digits: int) -> str:
  return f'{np.mean(values):,.{digits}f} ± {np.std(values, ddof=1) if len(values) > 1 else 0:,.{digits}f}'


def _best_set(groups: Dict[str, np.ndarray], higher_better: bool) -> set:
  """The best method plus every method not significantly different from it (Welch t-test, p >= 0.05)."""
  groups = {k: v for k, v in groups.items() if len(v)}
  if not groups:
    return set()
  best = (max if higher_better else min)(groups, key=lambda k: np.mean(groups[k]))
  bold = {best}
  for k, v in groups.items():
    if k != best and len(v) > 1 and len(groups[best]) > 1:
      if stats.ttest_ind(v, groups[best], equal_var=False).pvalue >= 0.05:
        bold.add(k)
  return bold


def table2(df: pd.DataFrame) -> str:
  df = df[df.experiment == 'table2']
  lines = ['| Dataset | Metric | ' + ' | '.join(METHOD_LABELS[m] for m in METHODS) + ' |',
           '|---|---|' + '---|' * len(METHODS)]
  for ds in [d for d in DATASETS if d in set(df.dataset)] + sorted(set(df.dataset) - set(DATASETS)):
    sub = df[df.dataset == ds]
    metric = sub.metric.iloc[0]
    higher = metric == 'accuracy'
    perf = {m: sub[sub.method == m].test_metric.to_numpy() for m in METHODS}
    size = {m: sub[sub.method == m].n_params.to_numpy() for m in METHODS}
    bold_perf, bold_size = _best_set(perf, higher), _best_set({k: v for k, v in size.items() if k != 'large_mlp'}, False)
    cell = lambda vals, b, d: '' if not len(vals) else (f'**{_fmt(vals, d)}**' if b else _fmt(vals, d))
    lines.append(f'| {ds} | {"Accuracy ↑" if higher else "MSE ↓"} | '
                 + ' | '.join(cell(perf[m], m in bold_perf, 3) for m in METHODS) + ' |')
    lines.append(f'| | # Parameters ↓ | '
                 + ' | '.join(cell(size[m], m in bold_size, 0) for m in METHODS) + ' |')
    n = {m: len(perf[m]) for m in METHODS}
    lines.append(f'| | runs | ' + ' | '.join(str(n[m]) for m in METHODS) + ' |')
  return '\n'.join(lines)


def table3(df: pd.DataFrame) -> str:
  df = df[df.method.isin(['tabnas', 'age', 'tabgns']) & df.experiment.isin(['table2', 'table3'])]
  methods = ['tabnas', 'age', 'tabgns']
  lines = ['| Dataset | Device | ' + ' | '.join(METHOD_LABELS[m] for m in methods) + ' |',
           '|---|---|' + '---|' * len(methods)]
  for ds in [d for d in DATASETS if d in set(df.dataset)]:
    for dev in ['cpu', 'cuda']:
      sub = df[(df.dataset == ds) & (df.search_device == dev)]
      if sub.empty:
        continue
      hours = {m: sub[sub.method == m].search_time_s.to_numpy() / 3600 for m in methods}
      bold = _best_set(hours, higher_better=False)
      cells = ['–' if not len(hours[m]) else (f'**{_fmt(hours[m], 2)}**' if m in bold else _fmt(hours[m], 2))
               for m in methods]
      lines.append(f'| {ds} | {dev.upper()} | ' + ' | '.join(cells) + ' |')
  return '\n'.join(lines)


def headline(df: pd.DataFrame) -> str:
  """The abstract/conclusion numbers: size reduction and search-time speed-up versus TabNAS and AgE."""
  lines = []
  t2 = df[df.experiment == 'table2']
  times = df[df.experiment.isin(['table2', 'table3'])]
  for ds in [d for d in DATASETS if d in set(t2.dataset)]:
    sub = t2[t2.dataset == ds]
    g = sub[sub.method == 'tabgns'].n_params.mean()
    for m in ['tabnas', 'age']:
      o = sub[sub.method == m].n_params.mean()
      if np.isfinite(g) and np.isfinite(o) and o > 0:
        lines.append(f'- {ds}: TabGNS uses {g / o:.0%} of {METHOD_LABELS[m]} parameters ({1 - g / o:.0%} smaller)')
    for dev in ['cpu', 'cuda']:
      tsub = times[(times.dataset == ds) & (times.search_device == dev)]
      g_t = tsub[tsub.method == 'tabgns'].search_time_s.mean()
      for m in ['tabnas', 'age']:
        # AgE only runs on CPU; compare GPU TabGNS against it too, as the paper's "up to 36x" does for TabNAS.
        o_t = (times[(times.dataset == ds) & (times.method == m)] if m == 'age' else tsub[tsub.method == m]).search_time_s.mean()
        if np.isfinite(g_t) and np.isfinite(o_t) and g_t > 0:
          lines.append(f'- {ds} ({dev}): TabGNS search is {o_t / g_t:.1f}x faster than {METHOD_LABELS[m]}')
  return '\n'.join(lines)


def fig2(df: pd.DataFrame, out: Path, dataset: str = 'vod') -> None:
  import matplotlib.pyplot as plt
  t2 = df[(df.experiment == 'table2') & (df.dataset == dataset)]
  cpu = df[(df.dataset == dataset) & (df.search_device == 'cpu') & df.experiment.isin(['table2', 'table3'])]
  ref = cpu[cpu.method == 'tabgns'].search_time_s.mean()
  if t2.empty or not np.isfinite(ref):
    return
  fig, ax = plt.subplots(figsize=(6, 4))
  for m in ['tabgns', 'tabnas', 'age']:
    perf, params = t2[t2.method == m].test_metric.mean(), t2[t2.method == m].n_params.mean()
    rel = cpu[cpu.method == m].search_time_s.mean() / ref
    if np.isfinite(perf) and np.isfinite(rel):
      ax.scatter(rel, perf, s=max(params / 500, 10), label=f'{METHOD_LABELS[m]} ({params / 1000:.0f}k params)')
  ax.set_xlabel('Relative search time (CPU)')
  ax.set_ylabel(t2.metric.iloc[0].upper())
  ax.legend()
  fig.tight_layout()
  fig.savefig(out / 'fig2.png', dpi=150)


def _param_curve(history: List[List[int]]) -> List[int]:
  return [sum(w[i] * w[i + 1] for i in range(len(w) - 1)) for w in history]


def fig4(df: pd.DataFrame, out: Path) -> None:
  import matplotlib.pyplot as plt
  sub = df[(df.experiment == 'fig4') & df.history.notna()]
  if sub.empty:
    return
  fig, ax = plt.subplots(figsize=(6, 4))
  for _, row in sub.sort_values('gate_init').iterrows():
    p_open = 1 / (1 + np.exp(-row.gate_init))
    ax.plot(_param_curve(row.history), label=f'p_open ≈ {p_open:.2f} (seed {row.seed})')
  ax.set_xlabel('Iterations (search epochs)')
  ax.set_ylabel('Number of parameters')
  ax.legend()
  fig.tight_layout()
  fig.savefig(out / 'fig4.png', dpi=150)


def fig5(df: pd.DataFrame, out: Path, dataset: str = 'vod') -> None:
  import matplotlib.pyplot as plt
  sub = df[(df.experiment == 'fig5') & (df.dataset == dataset)]
  if sub.empty:
    return
  agg = sub.groupby('gate_init').agg(mse=('test_metric', 'mean'), params=('n_params', 'mean')).reset_index()
  agg['p_open'] = 1 / (1 + np.exp(-agg.gate_init))
  fig, ax = plt.subplots(figsize=(6, 4))
  ax.plot(agg.p_open, agg.mse, 'o-', label='TabGNS (MSE)')
  t2 = df[(df.experiment == 'table2') & (df.dataset == dataset)]
  for m, style, color in [('age', ':', 'tab:orange'), ('tabnas', '--', 'tab:red')]:
    v = t2[t2.method == m].test_metric.mean()
    if np.isfinite(v):
      ax.axhline(v, linestyle=style, color=color, label=f'{METHOD_LABELS[m]} (MSE)')
  ax.set_xlabel('Initial probability of gate being open (p_open)')
  ax.set_ylabel('MSE')
  ax2 = ax.twinx()
  ax2.plot(agg.p_open, agg.params, 's-', color='tab:green', label='Number of parameters')
  ax2.set_yscale('log')
  ax2.set_ylabel('Number of parameters')
  ax.legend(loc='upper left')
  fig.tight_layout()
  fig.savefig(out / 'fig5.png', dpi=150)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument('--results', type=Path, default=Path('results'))
  parser.add_argument('--out', type=Path, default=Path('reports'))
  args = parser.parse_args()
  df = load_results(args.results)
  args.out.mkdir(parents=True, exist_ok=True)
  for name, fn in [('table2', table2), ('table3', table3), ('headline', headline)]:
    text = fn(df)
    (args.out / f'{name}.md').write_text(text + '\n')
    print(f'## {name}\n{text}\n')
  fig2(df, args.out)
  fig4(df, args.out)
  fig5(df, args.out)
  print(f'Figures written to {args.out}')


if __name__ == '__main__':
  main()
