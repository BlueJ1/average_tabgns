# Replicating the TabGNS paper

Harness for the experiments in *Automated Model Design using Gated Neuron Selection in Telecom*
(Orucu et al., NOMS 2026, [arXiv:2602.10854](https://arxiv.org/abs/2602.10854)).

**Scope** (see [STATUS.md](../STATUS.md)): only the public datasets, and only prediction performance and
model size. Search time is not studied, so the paper's Table III and Fig. 2 are not reproduced, and
runs may share the CPU freely.

Every run writes one JSON file to `results/<experiment>/<dataset>/<method>/<run_name>.json`
(test metric, widths, parameter counts, search/train time, device). Runs that already have a
result are skipped, so a sweep that was interrupted can be restarted with the same command.
`python -m experiments.report` turns the results into the tables and figures.

```
experiments/
  run.py              one (experiment, dataset, method, seed) cell, Hydra entry point
  report.py           Table II, headline size numbers, Fig. 4, 5
  datasets.py         loaders for the datasets (+ synthetic ones for smoke tests)
  methods.py          Large MLP, TabGNS (wraps tabgns/), TabNAS and AgE entry points
  baselines/tabnas.py TabNAS re-implementation (no public code exists)
  baselines/age.py    aging evolution on parallel CPU workers
  conf/               config.yaml + dataset/*.yaml + method/*.yaml
```

## Data

CoverType (OpenML 1596) and Higgs (OpenML 44129) download from OpenML on first use. Download them
once before a parallel sweep, so that jobs do not write the OpenML cache at the same time:

```bash
python -c "from load_data import load_covertype, load_higgs; load_covertype(); load_higgs(small=False)"
```

The paper's four telecom datasets are not used. VoD, Sim-A and Sim-B are Ericsson-internal and
their configs have been removed. DeepMIMO is public but has to be generated from
https://www.deepmimo.net/ (scenario O1, 3.5 GHz path loss per BS → 28/60/140 GHz, 18 → 54 columns);
`conf/dataset/deepmimo.yaml` reads it from `$TABGNS_DATA/deepmimo.csv` (default `./data`) if you add it.

## Commands

Everything runs on the CPU. On Apple silicon the MPS backend is 2–7× slower for these small MLPs at
batch size 256 and breaks TabGNS (see Known quirks). A single run barely uses more than one thread,
so independent runs go in parallel with Hydra's joblib launcher, one thread each
(`hydra/launcher=joblib hydra.launcher.n_jobs=6 threads=1`). Six jobs should fit 16 GB of memory with
Higgs (an estimate of ~1 GB peak per job, not yet measured on the full data).
Five seeds are used below; the paper does not say how many it used.

**Experiment 1, Table II (prediction performance and size).**

```bash
python -m experiments.run -m hydra/launcher=joblib hydra.launcher.n_jobs=6 threads=1 experiment=table2 dataset=covertype,higgs method=large_mlp,tabgns,tabnas seed=0,1,2,3,4
```

AgE runs its own pool of single-threaded workers, so it uses the default (sequential) launcher. The
paper used 20 workers; on an 8-core machine 8 are as fast.

```bash
python -m experiments.run -m experiment=table2 dataset=covertype,higgs method=age method.n_workers=8 seed=0,1,2,3,4
```

**Experiment 4, Fig. 4 (size during search).** The paper uses VoD; here CoverType. Gate logits −2
and +2 give p_open ≈ 0.12 and 0.88.

```bash
python -m experiments.run -m hydra/launcher=joblib hydra.launcher.n_jobs=2 threads=1 experiment=fig4 dataset=covertype method=tabgns method.gate_initialisation_val=-2,2 'run_name=g${method.gate_initialisation_val}_seed${seed}' save_history=true seed=0
```

**Experiment 5, Fig. 5 (gate-initialisation sweep).** The paper uses VoD; here CoverType. The paper's
grid is not given; logits −5…+2 cover p_open ≈ 0.007…0.88.

```bash
python -m experiments.run -m hydra/launcher=joblib hydra.launcher.n_jobs=6 threads=1 experiment=fig5 dataset=covertype method=tabgns method.gate_initialisation_val=-5,-4,-3,-2,-1,0,1,2 'run_name=g${method.gate_initialisation_val}_seed${seed}' seed=0,1,2,3,4
```

**Tables and figures.**

```bash
python -m experiments.report --results results --out reports
```

**Smoke test** (seconds, synthetic data, tiny search space):

```bash
python -m experiments.run -m hydra/launcher=joblib hydra.launcher.n_jobs=6 threads=1 experiment=smoke dataset=synthetic_reg,synthetic_cls method=large_mlp,tabgns,tabnas general.iters=4 'search_space.layer_widths=[32,32,32]'
```

Method-specific options can only be overridden in a sweep over that one method, e.g.
`method=age method.iterations=6 method.population_size=4 method.sample_size=2 method.n_workers=2`.

## Cost

Measured on an M1 Pro (6+2 cores, 16 GB): per-epoch times of every component, and epoch counts from
full runs on a 50k-row CoverType subsample (Large MLP ~60 epochs, TabGNS search ~140 + fine-tune ~23,
TabNAS search ~270–300 + retrain ~85).

"Run-hours" is the time to do every run one after another, each run as it would go alone: one job
on all cores for Large MLP, TabGNS and TabNAS, and one search with its 8 workers for AgE. It is not
core-hours (an AgE search keeps all 8 cores busy). "Wall-clock" is with the commands above.

| Part | Run-hours | Wall-clock |
|---|---|---|
| Table II, Large MLP + TabGNS + TabNAS (30 runs) | ~20 | ~6 h |
| Table II, AgE (10 searches) | ~43 | ~43 h |
| Fig. 4 (2 runs) | ~1.5 | ~1 h |
| Fig. 5 (40 runs) | ~31 | ~9 h |
| **Total** | **~95** | **~58 h (~2.4 days)**, 1.5–5 days depending on epoch counts |

AgE dominates: it trains all 300 candidates to convergence (~3 h per CoverType search, ~5 h per Higgs
search). Fig. 5 is the next largest; fewer seeds or gate values cut it proportionally.

## Choices the paper leaves open

These are set in `conf/` and can be changed there.

- **Seeds and statistics:** 5 seeds; split seed = run seed; Welch's t-test (p < 0.05) for bolding.
- **Size bolding (Table II footnote 1):** a method cannot win on size if its mean metric is more than
  5% worse than the best method's (`report.py --size-tolerance`); it gets a ¹ instead. On the paper's
  own numbers this excludes the same entries the paper footnotes.
- **Splits and preprocessing:** 60/20/20, quantile-normal feature scaling fitted on the full data
  (as in `load_data.py`), regression targets left unscaled.
- **Batch size:** 256 (from `conf/paper.yaml`).
- **TabGNS fine-tuning after warm start:** `finetune_epochs: null` means up to 300 epochs with early
  stopping, as `TabGNSTrainer` does.
- **TabNAS/AgE width options:** 20 geometrically spaced widths from 2 to 512.
- **TabNAS:** the reward is the negative validation-batch loss (MSE or cross-entropy), as in the
  TabNAS appendix. The search stops early once the validation loss of the policy's most likely
  architecture has not improved for 20 epochs after warm-up. With `max_params` set, the final
  architecture is the most likely one within the budget.
- **AgE:** a self-contained asynchronous aging evolution instead of DeepHyper (whose API has changed
  across versions). It only mutates layer widths, since the paper fixes the search space to plain
  5-layer MLPs. Candidates are scored by validation −MSE or accuracy.
- **Parameter count:** `n_params` counts hidden-to-hidden weights only, which is how the paper gets
  1,048,576 for the 5×512 MLP. `n_params_full` includes input/output layers and biases.

## Known quirks in `tabgns/`

- `BinaryGumbelSoftmax` uses logits `[g, 1]`, so during the search a gate is open with probability
  sigmoid(g − 1), while extraction keeps neurons with sigmoid(g) ≥ 0.5. The paper's "−3 → 4.7%"
  matches sigmoid(g). Configs and figures use sigmoid(g).
- `TabGNS` passes `self.net` itself to `EarlyStopper`, so its "best model" is the net from the last
  epoch, not the one from the epoch with the best validation loss.
- A layer can end up with 0 open gates; the extracted model then predicts a constant. On MPS
  (torch 2.5.1) fine-tuning such a model crashes ("Placeholder tensor is empty"), and the TabGNS
  search on CoverType-sized data produced NaN gate logits. Use the CPU.
