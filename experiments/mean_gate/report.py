"""
Paired comparison tables for the mean-gate study.

  python -m experiments.mean_gate.report --results results/mean_gate --out results/mean_gate

Writes runs.csv (one row per dataset/seed/arm) and summary.md. Seeds 0-2 are the primary pilot;
the 0-9 rows add the supplementary seeds run under the same frozen protocol.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy import stats

# (column, label, higher is better)
REG = [('mse_orig', 'MSE', False), ('mse', 'MSE (std. targets)', False)]
CLS = [('accuracy', 'Acc. %', True), ('cross_entropy', 'CE', False)]


def load(root: Path) -> pd.DataFrame:
  rows = []
  for p in sorted(root.glob('*/*/seed*.json')):
    r = json.loads(p.read_text())
    if r['experiment'] == 'mean_gate_smoke':
      continue
    for arm, failure in r['failures'].items():
      rows.append({'experiment': r['experiment'], 'dataset': r['dataset'], 'seed': r['seed'], 'arm': arm, 'failed': True})
    for arm, a in r['arms'].items():
      row = {'experiment': r['experiment'], 'dataset': r['dataset'], 'task': r['task'], 'seed': r['seed'],
             'arm': arm, 'failed': False, 'widths': a['widths'], 'zero_width_layers': len(a['zero_width_layers']),
             'n_params_full': a['n_params_full'], 'n_params': a['n_params'], 'best_epoch': a['search']['best_epoch'],
             'epochs_run': a['search']['epochs_run'], 'max_abs_diff': a['extraction_max_abs_diff'],
             'nonfinite': a['nonfinite'], 'search_time_s': a['search']['search_time_s'],
             'stats_time_s': a['search']['stats_time_s'], 'extraction_time_s': a['extraction_time_s'],
             'finetune_time_s': a['finetune']['finetune_time_s'], 'ft_epochs': a['finetune']['finetune_epochs_run'],
             'ft_best_epoch': a['finetune']['finetune_best_epoch'],
             'init_sha': a['init_param_sha256'][:12],
             'search_val_loss': a['search']['best_fixed_mask_valid_loss']}
      for stage in ('fixed_mask', 'stochastic', 'extracted', 'finetuned', 'extracted_without_bias_fold'):
        if stage in a:
          for split_ in ('valid', 'test'):
            for k, v in a[stage][split_].items():
              row[f'{stage}.{split_}.{k}'] = v
      rows.append(row)
    if 'dense_reference' in r:
      d = r['dense_reference']
      row = {'experiment': r['experiment'], 'dataset': r['dataset'], 'task': r['task'], 'seed': r['seed'],
             'arm': 'dense', 'failed': False, 'widths': d['widths'], 'n_params_full': d['n_params_full'],
             'n_params': d['n_params'], 'finetune_time_s': d['finetune']['finetune_time_s']}
      for split_ in ('valid', 'test'):
        for k, v in d['finetuned'][split_].items():
          row[f'finetuned.{split_}.{k}'] = v
      rows.append(row)
  return pd.DataFrame(rows)


def ms(x) -> str:
  x = np.asarray(x, dtype=float)
  return f'{x.mean():.3f} ± {x.std(ddof=1):.3f}' if len(x) > 1 else f'{x.mean():.3f}'


def paired(df: pd.DataFrame, col: str, higher: bool) -> str:
  """Three table cells: paired difference (mean - zero), seeds where the mean arm is better, t 95% CI."""
  z = df[df.arm == 'zero'].set_index('seed')[col]
  m = df[df.arm == 'mean'].set_index('seed')[col]
  d = (m - z).dropna()
  wins = int((d > 0).sum() if higher else (d < 0).sum())
  ci = ''
  if len(d) >= 5:
    lo, hi = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    ci = f'[{lo:+.3f}, {hi:+.3f}]'
  return f'{d.mean():+.3f} ± {d.std(ddof=1):.3f} | {wins}/{len(d)} | {ci}'


def dataset_section(df: pd.DataFrame, seeds: List[int]) -> List[str]:
  df = df[df.seed.isin(seeds)]
  task = df.task.dropna().iloc[0]
  mets = REG if task == 'regression' else CLS
  key, _, higher = mets[0]
  loss = 'mse' if task == 'regression' else 'cross_entropy'
  out = []

  # Per-seed table on the primary metric.
  out.append(f'| seed | arm | widths | params (full) | fixed-mask | extracted | fine-tuned | max abs diff | best ep. |')
  out.append('|---|---|---|---|---|---|---|---|---|')
  for seed in seeds:
    for arm in ('zero', 'mean', 'dense'):
      r = df[(df.seed == seed) & (df.arm == arm)]
      if r.empty:
        continue
      r = r.iloc[0]
      if r.get('failed'):
        out.append(f'| {seed} | {arm} | FAILED | | | | | | |')
        continue
      if arm == 'dense':
        out.append(f"| {seed} | dense ref. | {r.widths} | {r.n_params_full:,} | | | {r[f'finetuned.test.{key}']:.3f} | | |")
      else:
        out.append(f"| {seed} | {arm} | {r.widths} | {r.n_params_full:,} | {r[f'fixed_mask.test.{key}']:.3f} | "
                   f"{r[f'extracted.test.{key}']:.3f} | {r[f'finetuned.test.{key}']:.3f} | {r.max_abs_diff:.1e} | {int(r.best_epoch)} |")
  out.append('')

  # Summary: arm means and paired differences (mean - zero).
  out.append('| quantity (test unless noted) | zero | mean | Δ mean − zero | mean better | 95% CI of Δ |')
  out.append('|---|---|---|---|---|---|')
  arms = {a: df[df.arm == a] for a in ('zero', 'mean')}
  rows = [('full parameter count (better = smaller)', 'n_params_full', False)]
  for k, label, hi in mets:
    rows += [(f'fixed-mask {label}', f'fixed_mask.test.{k}', hi),
             (f'extracted {label}', f'extracted.test.{k}', hi),
             (f'fine-tuned {label}', f'finetuned.test.{k}', hi)]
  rows += [(f'fine-tuned valid {mets[0][1]}', f'finetuned.valid.{key}', higher),
           (f'stochastic-mask {mets[0][1]} (20 masks)', f'stochastic.test.{key}', higher)]
  for label, col, hi in rows:
    out.append(f'| {label} | {ms(arms["zero"][col])} | {ms(arms["mean"][col])} | {paired(df, col, hi)} |')
  # Gap between the deterministic network and the stochastic one it was trained as.
  gap = {n: (a[f'fixed_mask.test.{loss}'] - a[f'stochastic.test.{loss}']).values for n, a in arms.items()}
  out.append(f'| fixed-mask minus stochastic test loss | {ms(gap["zero"])} | {ms(gap["mean"])} | '
             f'{ms(gap["mean"] - gap["zero"])} | | |')
  nb = arms['mean'][f'extracted_without_bias_fold.test.{key}']
  out.append(f'| mean arm extracted *without* bias fold, {mets[0][1]} | | {ms(nb)} | | | |')
  dense = df[df.arm == 'dense'][f'finetuned.test.{key}']
  out.append(f'| dense full-width reference, fine-tuned {mets[0][1]} | {ms(dense)} | (same) | | | |')
  for label, col in (('search time s', 'search_time_s'), ('mean-statistics update time s', 'stats_time_s'),
                     ('extraction time s', 'extraction_time_s'), ('fine-tune time s', 'finetune_time_s'),
                     ('fine-tune epochs run', 'ft_epochs')):
    out.append(f'| {label} | {ms(arms["zero"][col])} | {ms(arms["mean"][col])} | | | |')
  out.append('')

  # Size vs accuracy: is the paired metric change explained by the paired size change?
  z, m = arms['zero'].set_index('seed'), arms['mean'].set_index('seed')
  dp = (m.n_params_full - z.n_params_full) / 1000
  dm = m[f'finetuned.test.{key}'] - z[f'finetuned.test.{key}']
  if len(dp) >= 5 and dp.std() > 0:
    fit = stats.linregress(dp, dm)
    out.append(f'Size check (fine-tuned {mets[0][1]}): corr(Δparams, Δmetric) = {fit.rvalue:+.2f}; '
               f'least-squares Δmetric at Δparams = 0: {fit.intercept:+.3f} (slope {fit.slope:+.3f} per 1k params).')
    out.append('')
  flags = df[(df.arm != 'dense') & ((df.zero_width_layers > 0) | df.nonfinite.astype(bool) | df.failed.astype(bool))]
  out.append(f'Zero-width layers / non-finite / failed runs: {len(flags)}'
             + ('' if flags.empty else ' — ' + ', '.join(f'seed {r.seed} {r.arm}' for r in flags.itertuples())))
  out.append('')
  return out


def compact(df: pd.DataFrame) -> List[str]:
  """One row per dataset and seed set: size, extracted and fine-tuned test metric, paired difference."""
  out = ['| dataset | seeds | metric | params zero / mean | extracted zero / mean | fine-tuned zero / mean | '
         'Δ fine-tuned (mean − zero) | mean better | max abs extraction diff |',
         '|---|---|---|---|---|---|---|---|---|']
  for ds, g in df.groupby('dataset', sort=False):
    task = g.task.dropna().iloc[0]
    key, label, higher = (REG if task == 'regression' else CLS)[0]
    seeds_all = sorted(g.seed.unique())
    for seeds in ([s for s in seeds_all if s <= 2], seeds_all) if len(seeds_all) > 3 else ([s for s in seeds_all if s <= 2],):
      h = g[g.seed.isin(seeds)]
      z, m = h[h.arm == 'zero'], h[h.arm == 'mean']
      cell = lambda c: f'{z[c].mean():.3g} / {m[c].mean():.3g}'
      d, wins, ci = paired(h, f'finetuned.test.{key}', higher).split(' | ')
      out.append(f'| {ds} | {seeds[0]}–{seeds[-1]} | {label} | {z.n_params_full.mean():.0f} / {m.n_params_full.mean():.0f} | '
                 f'{cell(f"extracted.test.{key}")} | {cell(f"finetuned.test.{key}")} | {d} {ci} | {wins} | '
                 f'{h[h.arm != "dense"].max_abs_diff.max():.1e} |')
  return out + ['']


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument('--results', default='results/mean_gate')
  ap.add_argument('--out', default='results/mean_gate')
  args = ap.parse_args()
  df = load(Path(args.results))
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  df.to_csv(out / 'runs.csv', index=False)

  lines = ['# Mean replacement vs. zero gating: paired results', '',
           'Generated by `python -m experiments.mean_gate.report`. Values are mean ± sample s.d. over seeds;',
           'paired differences are mean arm minus zero arm within a seed. "mean better k/n" counts seeds.',
           'The 95% CI (t, n ≥ 5 only) is descriptive; seeds are the only replication unit.', '',
           '## Compact comparison (test split)', '']
  lines += compact(df)
  for (exp, ds), g in df.groupby(['experiment', 'dataset'], sort=False):
    all_seeds = sorted(g.seed.unique())
    blocks = [('primary pilot, seeds 0–2', [s for s in all_seeds if s <= 2])]
    if len(all_seeds) > 3:
      blocks.append((f'all seeds 0–{max(all_seeds)} (supplementary seeds 3+ added)', all_seeds))
    for title, seeds in blocks:
      lines += [f'## {ds} — {exp}, {title}', '']
      lines += dataset_section(g, seeds)
  (out / 'summary.md').write_text('\n'.join(lines))
  print('\n'.join(lines))


if __name__ == '__main__':
  main()
