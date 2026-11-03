# Mean replacement vs. zero gating in TabGNS — pilot study

**Question.** TabGNS gates each hidden neuron after its ReLU: `h̃ = g·h`, so a closed neuron outputs 0.
Mean ablation (Wang et al., *Interpretability in the Wild*, Sec. 2.2) argues that zeroing a component also
removes its average contribution, which downstream units may rely on as an implicit bias; replacing it
with its mean removes only the input-dependent part. The variant tested here is

    h̃ = g·h + (1 − g)·μ,    μ = running mean of the post-ReLU, pre-gate activation.

Unlike the interpretability setting (ablations on a frozen, pretrained GPT-2), TabGNS trains weights
and gates jointly, so downstream biases can learn to compensate under zero gating. The pilot asks
whether mean replacement nevertheless (1) is preserved exactly by extraction and (2) gives better
final models for the size and compute used.

**Short answer.** (1) Yes. With the bias fold below, extraction reproduces each arm's fixed-mask
network to ≤ 2.9e-6, and leaving the fold out costs a lot. On these corrected runs, which select
checkpoints by fixed-mask validation loss, neither arm's fixed-mask network is worse on average than
its sampled-mask network. (2) On the classification task, no clear average difference was detected
(Δ accuracy +0.01 pp, descriptive 95% interval −0.68 to +0.70 pp). This is compatible with a gain or a
loss of about 0.7 pp, so it is not evidence of equivalence. On the nonlinear regression task,
fine-tuned MSE is about 6% lower on average, in 8/10 seeds, at a similar average parameter count. That
result is borderline, comes from one of several metrics, and both stages hit their epoch caps. It is a
possible fixed-budget benefit, not a size-independent gain. It justifies one bounded follow-up, not a
paper-scale sweep.

## What was run

Everything lives in the opt-in package `experiments/mean_gate/`. `tabgns/` and the existing harness
are unchanged, so the original method is still what `python -m experiments.run method=tabgns` runs.

| file | role |
|---|---|
| `model.py` | `ControlledGatedBlock` (subclass of `tabgns.gates.GatedBlock`, same parameters and creation order), `ControlledGatedNet`, `extract_mlp` |
| `protocol.py` | data prep, search loop, fine-tuning, evaluation, random streams |
| `run.py` | Hydra entry point; one job runs both arms + a dense reference on one (dataset, seed) |
| `report.py` | builds `results/mean_gate/summary.md` and `runs.csv` |
| `test_mean_gate.py` | correctness checks (stdlib `unittest`; pytest is not installed) |
| `conf/` | `config.yaml` + `dataset/*.yaml` |

**Arms.** `gate_mode=zero` (`h̃ = g·h`) and `gate_mode=mean` (`h̃ = g·h + (1−g)·μ`). Within a seed both
arms share the split, the initial weights and gate logits (checked: identical parameter hashes in every
pair), the optimisers, the stopping rules and the fine-tuning budget. They also share the random
streams. Weight init uses `torch.manual_seed(seed)` before construction. Train order, validation
order, gate noise and stochastic evaluation each use their own `torch.Generator`. The mean bookkeeping
draws no random numbers, so both arms see the same gate-noise sequence.

**Means.** `μ` is a detached per-neuron buffer, updated with an EMA (momentum 0.1, BatchNorm's default)
of the batch mean of `h`, and initialised from the first training batch. It is updated only during the
forward pass of a weight step on a training batch (`update_stats=True`). Validation-batch logging, the
gate step on validation data, all evaluation and extraction leave it untouched. There are no
label-conditioned means and no sweep over the coefficient. `μ` and its update counter are buffers, so
they are saved, restored and snapshotted with the checkpoint. A side effect, checked by a test: the gate
gradient is `∂L/∂h̃ · (h − μ)` in the mean arm instead of `∂L/∂h̃ · h`. A gate is credited only for its
neuron's input-dependent variation.

**Extraction.** Neurons with `s ≥ 0` are kept. For the mean arm, the constant contribution of the
removed set D of layer ℓ is folded into layer ℓ+1 before its columns are dropped,
`b' = b + W[:, D] μ_D`. This includes the output layer. There is no probability scaling. The extracted
model is the repo's own `tabgns.mlp.MLP`, and zero-width layers are supported.

### Controlled protocol vs. repository defaults (applied identically to both arms)

| aspect | repository default (`experiments/run.py`, `tabgns/`) | this study |
|---|---|---|
| gate sampling | Gumbel-softmax logits `[s, 1]` → P(open) = σ(s − 1) | logits `[s, 0]` → P(open) = σ(s) (same distribution and ST gradient as `F.gumbel_softmax`) |
| extraction rule | σ(s) ≥ 0.5, `scale=true` multiplies weights by σ(s) | s ≥ 0 (≡ σ(s) ≥ 0.5), no scaling |
| effective initial P(open) at logit −3 | σ(−4) = 1.8% | σ(−3) = 4.7% (what the paper states) |
| search checkpoint | `EarlyStopper` holds a reference to the live net → last epoch is used | deep-copied snapshot of the epoch with the lowest **fixed-mask** validation loss (repo criterion: mean stochastic validation-batch loss) |
| feature preprocessing | quantile-normal fitted on the full dataset before splitting | fitted on the training split only |
| regression targets | unscaled | standardised with training mean/s.d.; MSE reported in original units and standardised |
| fine-tuning | `MLP.fit`: ≤ 300 epochs, patience 20, global-RNG shuffling | same optimiser (AdamW, lr 1e-3) and patience; ≤ 100 epochs; seeded shuffling; the extracted model itself is the epoch −1 candidate |
| search budget / space | 300 epochs, 5 × 512 | 50 epochs (patience 20), 2 × 32 |
| zero-width layers | possible | supported; none occurred in the pilot |

The search loop is otherwise the original one. Each training batch gets a weight step with sampled
gates, a no-grad logging pass on a validation batch, then a gate step on that validation batch (Adam,
arch lr 0.05, weight lr 1e-3, temperature 1, batch 256). The validation split is therefore used for
gate learning and checkpoint selection. The test split is used only for the final numbers.

**Data.** A 60/20/20 split with split seed = model seed = run seed. The dataset-generation seed is fixed at 0.
- `friedman_reg`: `make_friedman1`, 4,000 × 10, noise 1.0. Nonlinear, 5 informative features.
- `synthetic_cls_4k`: the repo's `synthetic_cls` generator (`make_classification`, 20 features, 10
  informative, 3 classes) with 4,000 rows.
- Supplementary `diabetes`: sklearn's bundled real dataset (442 × 10, no download), with batch 32 so
  that its 265 training rows give about 9 steps per epoch, like the synthetic sets.

**Staging.** I ran the smoke tests first, then one protocol check on seed 0 that looked only at
validation-side search history. That check showed gates start opening around epoch 8 and validation
loss still falling at epoch 40, so the search budget was fixed at 50 epochs, the top of the planned
range. Before I looked at any test result, I fixed the protocol and declared two supplements: seeds
3–9 on the same frozen protocol, and the diabetes set. The primary pilot is seeds 0–2.

**Compute.** MacBook (arm64, 8 cores), CPU only, 1 torch thread; torch 2.5.1, numpy 1.26.0,
scikit-learn 1.3.2, hydra 1.3.8, Python 3.11.5. Pilot (12 arm runs + 6 dense references): 47 s.
Seeds 3–9: 84 s. Diabetes: 10 s. Extraction takes about 1 ms. Timing the EMA update body on its own
gives about 0.05 s per search. The observed total search time differs by more than that: 1.302 vs
1.396 s on Friedman and 1.416 vs 1.575 s on classification, zero vs mean arm, 10 seeds. These are
short wall-clock timings, and the two arms always run in the same order (zero, then mean) in one
process. Read them as rough indications, not as a measured overhead of mean replacement.

## Correctness checks

`python -m unittest -v experiments.mean_gate.test_mean_gate`: **16/16 pass**
(log: `results/mean_gate/correctness_checks.txt`).

- Open gates pass `h`. Closed gates give 0 (zero arm) or `μ` (mean arm). Saturated stochastic gates equal the fixed mask.
- Empirical open frequency of the sampler matches σ(s) (±0.015, 20k draws). The fixed mask is σ(s) ≥ 0.5.
- With μ = 0 the mean arm equals the zero arm exactly, for both stochastic and fixed masks.
- The gate gradient equals Σ(h − μ)·∂L/∂h̃.
- EMA update rule and first-batch initialisation. Means are unchanged by stochastic and fixed passes in train and eval mode.
- In a real `search()` run, the means are updated exactly once per training batch.
- Mean buffers round-trip through `state_dict` save/load.
- Paired arms have identical initial parameters, and mean bookkeeping does not advance the gate-noise generator.
- Extraction equals the fixed-mask network (< 1e-5) for mixed masks across 3 hidden layers, for all-open masks, and for each hidden layer fully removed in turn, in both arms. With an empty layer the output is constant.
- Omitting the bias fold breaks equivalence for the mean arm (diff > 1e-2).
- The report never pools runs across experiments. A second protocol beside `pilot` gets its own rows,
  and runs with different protocols under one experiment name, or a duplicated (seed, arm) pair, are
  refused.

The checks have dropout and scaling disabled and compare against the deterministic fixed-mask forward, never a fresh stochastic pass.

## Results

Test split; mean over seeds. Paired Δ = mean arm − zero arm within a seed, ± s.d. over seeds; the t 95% CI is
shown for n ≥ 5 and is descriptive only. Full per-seed tables are in `results/mean_gate/summary.md`.

| dataset | seeds | metric | full params zero / mean | extracted zero / mean | fine-tuned zero / mean | Δ fine-tuned | mean better |
|---|---|---|---|---|---|---|---|
| friedman_reg | 0–2 | MSE | 503 / 478 | 4.24 / 3.93 | 1.83 / 1.72 | −0.113 ± 0.063 | 3/3 |
| friedman_reg | 0–9 | MSE | 538 / 539 | 3.81 / 3.75 | 1.77 / 1.66 | −0.110 ± 0.127, CI [−0.201, −0.019] | 8/10 |
| synthetic_cls_4k | 0–2 | Acc. % | 865 / 1004 | 86.7 / 87.4 | 88.8 / 89.3 | +0.46 ± 0.62 | 2/3 |
| synthetic_cls_4k | 0–9 | Acc. % | 942 / 974 | 87.6 / 87.4 | 89.4 / 89.4 | +0.01 ± 0.97, CI [−0.68, +0.70] | 5/10 |
| diabetes (real, suppl.) | 0–2 | MSE | 298 / 351 | 3198 / 3166 | 3201 / 3166 | −35 ± 121 | 2/3 |

Dense 2 × 32 reference (1,441 params for Friedman), trained from scratch with the same 100-epoch cap:
Friedman MSE 1.80, classification 89.1%, diabetes 3399.

### Q1: Does extraction preserve each arm's fixed-mask computation?

- **Yes, in all 46 arm runs.** The maximum |fixed-mask − extracted| over train+valid+test inputs is ≤ 9.5e-7
  (regression) and ≤ 2.9e-6 (classification logits). The extracted and fixed-mask test metrics agree
  to the printed precision.
- **The bias fold matters.** Deleting the mean arm's closed neurons without folding μ gives Friedman MSE
  5.11 vs 3.75 with the fold (10 seeds), classification 86.4% vs 87.4%, and diabetes 3723 vs 3166.
- **Fixed vs. sampled masks on these runs.** On average, the deterministic network is about as good as
  the average of 20 sampled-mask networks in both arms (fixed − stochastic test loss below). About
  86–90% of the gates are saturated (p < 0.1 or > 0.9) at the selected checkpoint. These checkpoints
  were selected by fixed-mask validation loss, which favours the deterministic network being assessed.

| fixed − stochastic test loss | zero | mean |
|---|---|---|
| Friedman (std. MSE) | −0.005 | −0.005 |
| classification (CE) | −0.006 | −0.003 |
| diabetes (std. MSE) | −0.009 | −0.010 |

So in these corrected pilot runs, neither arm's fixed-mask network is worse on average than its
sampled-mask network, and there is no average fixed-vs-sampled gap for mean replacement to reduce.
This does not show that replacing stochastic gates with a fixed mask cannot hurt in other settings.
Extraction equivalence only shows that the compact model implements its chosen fixed-mask reference.
Nor do these runs say where any damage in the repository's default pipeline comes from. That pipeline
was not run. Attributing damage to individual default behaviours (the σ(s−1) vs σ(s) threshold,
`scale=true`, stochastic-loss checkpoint selection) would need separate ablations.

### Q2: Does mean replacement give better final models for the size and compute used?

- **Classification: no clear average difference detected under this protocol.** The interval is too
  wide to call this equivalence:
  - Fine-tuned accuracy: Δ +0.01 pp, descriptive 95% CI [−0.68, +0.70], 5/10 seeds. This is compatible
    with a gain or a loss of about 0.7 pp.
  - Cross-entropy: Δ −0.002, CI [−0.008, +0.005].
  - Before fine-tuning: Δ −0.16 pp, CI [−1.10, +0.77].

  Demonstrating practical equivalence would need a prespecified tolerance and enough precision to put
  the interval inside it. In the 3-seed pilot the mean arm kept more neurons (+139 params, 0/3
  smaller); over 10 seeds sizes were similar (+32). Per-seed accuracy and size differences are
  correlated (corr +0.62, descriptive).
- **Friedman regression: a possible fixed-budget benefit at similar average parameter count, only
  after fine-tuning.**
  - Fine-tuned test MSE is 6% lower on average (1.66 vs 1.77), better in 8/10 seeds. The average full
    parameter count is about the same (Δ +0.6).
  - That is a statement about averages, not a matched-size comparison. Size is itself an outcome of the
    search, and the arms select different widths within a seed: seed 0 compares [18, 17] with [15, 15],
    seed 9 compares [17, 22] with [20, 23]. The per-seed regression of Δmetric on Δparams (corr −0.57,
    intercept −0.11) is descriptive only. One parameter-count covariate does not control architecture
    shape or inherited weights.
  - Separating architecture selection from the effect on training and warm-start weights would need a
    matched-width or fixed-mask control. One example: fine-tune both arms' weights on the same mask.
  - Fine-tuned validation MSE moves the same way (Δ −0.12, 7/10). But validation data also trains the
    gates and selects both checkpoints, so this is a selection diagnostic, not independent confirmation.
  - Before fine-tuning there is no reliable effect: the fixed-mask/extracted MSE difference is
    −0.06 ± 0.57 (7/10, CI spans 0). In the 3-seed pilot it looked consistent (3/3, −0.31), but the
    extra seeds removed that. Seed 4 alone has the mean arm far worse (5.15 vs 3.83).
- **Diabetes (real, 3 seeds): inconclusive.** Differences are small relative to seed noise. Fine-tuning
  never improved the validation loss on 5 of 6 runs, so the extracted model was kept as is.
- **Compute.** The EMA update body takes about 0.05 s per search when timed alone. The observed total
  search time was 0.09–0.16 s longer for the mean arm, but these short, fixed-order timings are not a
  reliable overhead estimate (see Compute above). Epoch budgets are identical.

**Search behaviour.** Both arms open their first neuron in every layer at about the same time
(epochs 9.5 vs 10.5 Friedman, 11.3 vs 10.9 classification). They end with similar mean P(open) (0.55
vs 0.55; 0.64 vs 0.65) and select almost the same fixed-mask validation loss (Friedman std. 0.148 vs
0.142; CE 0.329 vs 0.323). Learned means average 0.28 (Friedman) to 0.50 (classification), so the two
arms do differ: closed neurons pass non-trivial constants in the mean arm.

## Limitations

- Three primary seeds on two synthetic tasks, extended to 10 seeds as a supplement. One tiny real
  dataset. This is preliminary evidence only.
- I looked at several metrics × datasets × seed sets. One CI excluding zero (Friedman fine-tuned MSE,
  10 seeds) is what multiple comparisons alone would often produce. Treat it as a lead, not a finding.
  No primary endpoint was fixed before the test results were seen.
- Validation metrics are not independent evidence. The validation split trains the gates, selects the
  search checkpoint and stops fine-tuning.
- Checkpoints are selected by fixed-mask validation loss for both arms, unlike the repository's
  stochastic-loss criterion. This favours good deterministic networks, which matters when reading the
  fixed-vs-sampled comparison.
- **Both stages are budget-capped.** The search selected epoch 46–49 in every synthetic run (38–49 on
  diabetes), so validation loss was still falling at the 50-epoch cap. Friedman fine-tuning ran the
  full 100 epochs in every run, for both arms and the dense reference. The Friedman gain may therefore be faster convergence of the mean-arm extracted
  network under a fixed cap, not a better optimum. A longer cap would tell these apart.
- Tiny search space (2 × 32) and no size penalty. The paper uses 5 × 512, where layer interactions and
  zero-width layers are more likely. Zero-width layers did not occur here, so that path is covered only
  by unit tests.
- Single choices without sweeps: EMA momentum 0.1, gate logit −3, arch lr 0.05, temperature 1. I did not
  recalibrate μ under the final fixed mask before extraction. The fold uses the running mean from
  stochastic-mask training, which is the quantity the mean arm was trained with.
- The comparison is between two *corrected* pipelines. It says nothing about how either arm compares to
  the repository's default TabGNS (threshold mismatch, `scale=true`, last-epoch checkpoint).
- The local PDF of *Interpretability in the Wild* could not be rendered (no PDF tools installed). The
  mean-ablation rationale is taken from the task description of Sec. 2.2, consistent with the paper.

## Are larger experiments justified?

A paper-scale sweep is not justified yet. This pilot supports keeping the result as preliminary
evidence: a possible fixed-budget Friedman benefit at similar average size. It does not establish a
size-independent gain, classification equivalence, or the causes of damage in the untested default
pipeline. The original TabGNS stays available and unchanged. A bounded follow-up (tens of CPU minutes)
should:

1. **Freeze the primary endpoint before looking at new test results.** For example: fine-tuned test MSE
   on a Friedman-type task, paired over 10 seeds. Name the secondary endpoints, and a tolerance if
   equivalence on classification is a goal.
2. **Use fresh data for confirmation.** Regenerate with a new `dataset.data_seed` (e.g. `data_seed=1`),
   or use an untouched dataset, rather than the generated sets already examined here.
3. **Vary the search and fine-tuning caps separately**, since both are capped here. For example:
   `search.epochs` ∈ {50, 150} × `finetune.epochs` ∈ {100, 300}, each under its own `experiment=` name.
   `report.py` now refuses to pool different protocols, so each cell stays separate.
4. **Add a matched-width / fixed-mask control** to separate architecture selection from the effect on
   trained weights. For example, fine-tune both arms on one shared mask, or retrain both selected
   architectures from the same random init.
5. If the effect survives, add a subsampled real set (e.g. CoverType or Higgs, 5k rows) and the paper's
   5 × 512 space before considering the full Table II protocol.
6. Separately, if extraction damage in the *default* pipeline matters, ablate its pieces one at a time
   against the corrected zero arm: σ(s−1) sampling, `scale=true`, and stochastic-loss checkpoint
   selection. That question is distinct from mean replacement.

## Reproduction

```bash
.venv/bin/python -m unittest -v experiments.mean_gate.test_mean_gate
```

```bash
.venv/bin/python -m experiments.mean_gate.run -m experiment=mean_gate_smoke dataset=synthetic_reg,synthetic_cls seed=0 search.epochs=4 finetune.epochs=4 'search.layer_widths=[32,32,32]' results_dir=results/mean_gate overwrite=true
```

```bash
.venv/bin/python -m experiments.mean_gate.run -m experiment=pilot dataset=friedman_reg,synthetic_cls_4k seed=0,1,2 results_dir=results/mean_gate
```

```bash
.venv/bin/python -m experiments.mean_gate.run -m experiment=pilot dataset=friedman_reg,synthetic_cls_4k seed=3,4,5,6,7,8,9 results_dir=results/mean_gate
```

```bash
.venv/bin/python -m experiments.mean_gate.run -m experiment=supp_diabetes dataset=diabetes general.batch_size=32 seed=0,1,2 results_dir=results/mean_gate
```

```bash
.venv/bin/python -m experiments.mean_gate.report --results results/mean_gate --out results/mean_gate
```

Outputs are in `results/mean_gate/`:
- `<experiment>/<dataset>/seed<k>.json`: per (dataset, seed), both arms and the dense reference. Each
  file holds the resolved config, data-generation settings, split/model/stream seeds, split sizes,
  target scaling, per-epoch architecture history and losses, selected widths, P(open), μ, all
  metrics, timings, extraction discrepancy, git HEAD/status/diff hash, source hashes, and library
  versions.
- `runs.csv`, `summary.md`
- run logs `*.log`
- `correctness_checks.txt`

Existing results are skipped unless `overwrite=true`. The results were produced from uncommitted
changes on top of `35e9198`; each JSON records the porcelain status and the source-file hashes.
