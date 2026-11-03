# Replicating the TabGNS paper

Harness for the experiments in *Automated Model Design using Gated Neuron Selection in Telecom*
(Orucu et al., NOMS 2026, [arXiv:2602.10854](https://arxiv.org/abs/2602.10854)).

Every run writes one JSON file to `results/<experiment>/<dataset>/<method>/<run_name>.json`
(test metric, widths, parameter counts, search/train time, device). Runs that already have a
result are skipped, so a sweep that was interrupted can be restarted with the same command.
`python -m experiments.report` turns the results into the tables and figures.

```
experiments/
  run.py              one (experiment, dataset, method, seed) cell, Hydra entry point
  report.py           Table II, Table III, headline numbers, Fig. 2, 4, 5
  datasets.py         loaders for the six datasets (+ synthetic ones for smoke tests)
  methods.py          Large MLP, TabGNS (wraps tabgns/), TabNAS and AgE entry points
  baselines/tabnas.py TabNAS re-implementation (no public code exists)
  baselines/age.py    aging evolution on parallel CPU workers
  conf/               config.yaml + dataset/*.yaml + method/*.yaml
```

## Data

CoverType and Higgs download from OpenML. The four telecom datasets are not public. Export each one
as a CSV with the input columns first and the target columns last (or set `target_columns` in its
`conf/dataset/*.yaml`) and put them in `$TABGNS_DATA` (default `./data`):

| File | Inputs → targets | Rows |
|---|---|---|
| `vod.csv` | 46 → 1 (frame rate) | 50,000 |
| `deepmimo.csv` | 18 → 54 (path loss, 3.5 GHz → 28/60/140 GHz, scenario O1) | 31,419 |
| `sim_a.csv` | 9 → 9 (RSRP 900 MHz → 4.5 GHz) | 99,999 |
| `sim_b.csv` | 9 → 27 (RSRP 800 MHz → 1800/2600/3500 MHz) | 50,100 |

DeepMIMO can be generated from https://www.deepmimo.net/. The other three are Ericsson-internal.

## Commands

Five seeds are used below; the paper does not say how many it used.

**Experiment 1, Table II (prediction performance and size).** TabGNS, TabNAS and the Large MLP on
GPU, which also gives Table III's GPU times. AgE always searches on 20 CPU workers.

```bash
python -m experiments.run -m experiment=table2 device=cuda dataset=vod,deepmimo,sim_a,sim_b,covertype,higgs method=large_mlp,tabgns,tabnas seed=0,1,2,3,4
```

```bash
python -m experiments.run -m experiment=table2 dataset=vod,deepmimo,sim_a,sim_b,covertype,higgs method=age seed=0,1,2,3,4
```

**Experiment 2, Table III (search time).** The CPU times for TabGNS and TabNAS. GPU and AgE times
come from the Table II runs.

```bash
python -m experiments.run -m experiment=table3 device=cpu dataset=vod,deepmimo,sim_a,sim_b,covertype,higgs method=tabgns,tabnas seed=0,1,2,3,4
```

**Experiment 3, Fig. 2.** No extra runs; built from Experiments 1 and 2 (VoD, CPU times).

**Experiment 4, Fig. 4 (size during search).** Gate logits −2 and +2 give p_open ≈ 0.12 and 0.88.

```bash
python -m experiments.run -m experiment=fig4 dataset=vod method=tabgns method.gate_initialisation_val=-2,2 'run_name=g${method.gate_initialisation_val}_seed${seed}' save_history=true seed=0
```

**Experiment 5, Fig. 5 (gate-initialisation sweep).** The paper's grid is not given; logits
−5…+2 cover p_open ≈ 0.007…0.88.

```bash
python -m experiments.run -m experiment=fig5 dataset=vod method=tabgns method.gate_initialisation_val=-5,-4,-3,-2,-1,0,1,2 'run_name=g${method.gate_initialisation_val}_seed${seed}' seed=0,1,2,3,4
```

**Tables and figures.**

```bash
python -m experiments.report --results results --out reports
```

**Smoke test** (seconds, synthetic data, tiny search space):

```bash
python -m experiments.run -m experiment=smoke dataset=synthetic_reg,synthetic_cls method=large_mlp,tabgns,tabnas general.iters=4 'search_space.layer_widths=[32,32,32]'
```

Method-specific options can only be overridden in a sweep over that one method, e.g.
`method=age method.iterations=6 method.population_size=4 method.sample_size=2 method.n_workers=2`.

## Cost

AgE trains every one of its 300 candidates to convergence. The paper reports 10–233 hours per
AgE search on 20 cores, so the AgE row of Table II is by far the most expensive part
(roughly 2,000+ wall-clock hours for 6 datasets × 5 seeds). TabGNS takes minutes to ~2 hours per
search on a GPU.

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

## Exploration: mean replacement instead of zero gating

`experiments/mean_gate/` is an opt-in, self-contained study of `h̃ = g·h + (1−g)·μ` against the
original `h̃ = g·h`, on a controlled pipeline that fixes the quirks below for both arms. It does not
change `tabgns/` or `experiments/run.py`. See `experiments/mean_gate/REPORT.md` for the protocol,
commands and results.

## Known quirks in `tabgns/`

- `BinaryGumbelSoftmax` uses logits `[g, 1]`, so during the search a gate is open with probability
  sigmoid(g − 1), while extraction keeps neurons with sigmoid(g) ≥ 0.5. The paper's "−3 → 4.7%"
  matches sigmoid(g). Configs and figures use sigmoid(g).
- `TabGNS` passes `self.net` itself to `EarlyStopper`, so its "best model" is the net from the last
  epoch, not the one from the epoch with the best validation loss.
- A layer can end up with 0 open gates; the extracted model then predicts a constant.
