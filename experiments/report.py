"""
Builds the paper's tables and figures from the JSON results written by experiments.run.

  python -m experiments.report --results results --out reports

Writes table2.md, headline.md and fig4/fig5 PNGs to --out. Missing cells are skipped.
Search time is out of scope (see STATUS.md), so Table III and Fig. 2 are not built.
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
# Row order of Table II; datasets not listed here are appended alphabetically.
DATASETS = ['deepmimo', 'covertype', 'higgs']


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


def _much_worse(perf: Dict[str, np.ndarray], higher_better: bool, tolerance: float) -> set:
  """Methods whose mean metric is more than `tolerance` (relative) worse than the best mean."""
  means = {k: np.mean(v) for k, v in perf.items() if len(v)}
  if not means:
    return set()
  best = (max if higher_better else min)(means.values())
  return {k for k, m in means.items() if abs(m - best) > tolerance * abs(best)}


def table2(df: pd.DataFrame, size_tolerance: float = 0.05) -> str:
  df = df[df.experiment == 'table2']
  lines = ['| Dataset | Metric | ' + ' | '.join(METHOD_LABELS[m] for m in METHODS) + ' |',
           '|---|---|' + '---|' * len(METHODS)]
  any_excluded = False
  for ds in [d for d in DATASETS if d in set(df.dataset)] + sorted(set(df.dataset) - set(DATASETS)):
    sub = df[df.dataset == ds]
    metric = sub.metric.iloc[0]
    higher = metric == 'accuracy'
    perf = {m: sub[sub.method == m].test_metric.to_numpy() for m in METHODS}
    size = {m: sub[sub.method == m].n_params.to_numpy() for m in METHODS}
    # Table II footnote 1: a small model with much worse predictions does not win on size.
    # Only methods that would otherwise be bold get the footnote, as in the paper.
    candidates = {k: v for k, v in size.items() if k != 'large_mlp'}
    excluded = _much_worse(perf, higher, size_tolerance) & _best_set(candidates, False)
    any_excluded |= bool(excluded)
    bold_perf = _best_set(perf, higher)
    bold_size = _best_set({k: v for k, v in candidates.items() if k not in excluded}, False)
    cell = lambda vals, b, d, note='': '' if not len(vals) else (f'**{_fmt(vals, d)}**' if b else _fmt(vals, d)) + note
    lines.append(f'| {ds} | {"Accuracy ↑" if higher else "MSE ↓"} | '
                 + ' | '.join(cell(perf[m], m in bold_perf, 3) for m in METHODS) + ' |')
    lines.append(f'| | # Parameters ↓ | '
                 + ' | '.join(cell(size[m], m in bold_size, 0, '¹' if m in excluded else '') for m in METHODS) + ' |')
    n = {m: len(perf[m]) for m in METHODS}
    lines.append(f'| | runs | ' + ' | '.join(str(n[m]) for m in METHODS) + ' |')
  if any_excluded:
    lines.append('')
    lines.append(f'¹ Parameter count ignored from bolding: mean prediction performance is more than '
                 f'{size_tolerance:.0%} worse than the best method.')
  return '\n'.join(lines)


def headline(df: pd.DataFrame) -> str:
  """The abstract/conclusion size numbers: TabGNS model size relative to TabNAS and AgE."""
  lines = []
  t2 = df[df.experiment == 'table2']
  for ds in [d for d in DATASETS if d in set(t2.dataset)] + sorted(set(t2.dataset) - set(DATASETS)):
    sub = t2[t2.dataset == ds]
    g = sub[sub.method == 'tabgns'].n_params.mean()
    for m in ['tabnas', 'age']:
      o = sub[sub.method == m].n_params.mean()
      if np.isfinite(g) and np.isfinite(o) and o > 0:
        lines.append(f'- {ds}: TabGNS uses {g / o:.0%} of {METHOD_LABELS[m]} parameters ({1 - g / o:.0%} smaller)')
  return '\n'.join(lines)


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


def fig5(df: pd.DataFrame, out: Path, dataset: str = 'covertype') -> None:
  import matplotlib.pyplot as plt
  sub = df[(df.experiment == 'fig5') & (df.dataset == dataset)]
  if sub.empty:
    return
  label = 'Accuracy' if sub.metric.iloc[0] == 'accuracy' else 'MSE'
  agg = sub.groupby('gate_init').agg(perf=('test_metric', 'mean'), params=('n_params', 'mean')).reset_index()
  agg['p_open'] = 1 / (1 + np.exp(-agg.gate_init))
  fig, ax = plt.subplots(figsize=(6, 4))
  ax.plot(agg.p_open, agg.perf, 'o-', label=f'TabGNS ({label})')
  t2 = df[(df.experiment == 'table2') & (df.dataset == dataset)]
  for m, style, color in [('age', ':', 'tab:orange'), ('tabnas', '--', 'tab:red')]:
    v = t2[t2.method == m].test_metric.mean()
    if np.isfinite(v):
      ax.axhline(v, linestyle=style, color=color, label=f'{METHOD_LABELS[m]} ({label})')
  ax.set_xlabel('Initial probability of gate being open (p_open)')
  ax.set_ylabel(label)
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
  parser.add_argument('--size-tolerance', type=float, default=0.05,
                      help='Relative gap to the best mean metric beyond which a method cannot win on size '
                           '(Table II footnote 1). 0.05 reproduces the paper\'s exclusions.')
  args = parser.parse_args()
  df = load_results(args.results)
  args.out.mkdir(parents=True, exist_ok=True)
  for name, fn in [('table2', lambda d: table2(d, args.size_tolerance)), ('headline', headline)]:
    text = fn(df)
    (args.out / f'{name}.md').write_text(text + '\n')
    print(f'## {name}\n{text}\n')
  fig4(df, args.out)
  fig5(df, args.out)
  print(f'Figures written to {args.out}')


if __name__ == '__main__':
  main()
