# TabGNS on TabM

**Question.** TabGNS (Orucu et al., NOMS 2026) searches the widths of a plain MLP by putting a learnable
Gumbel gate on every hidden neuron and training the gates on validation batches, alternating with the
weights. Does the method carry over to TabM (Gorishniy et al., ICLR 2025), a parameter-efficient
ensemble of k MLPs that share their weight matrices? In particular: does it find TabMs that are
clearly smaller than TabM's default architecture without losing accuracy, as it does for MLPs?

<!-- RESULTS -->

## What TabM is, and where the gates go

TabM (`arch_type='tabm'` in the reference package `tabm` 0.0.3) runs k = 32 MLPs in parallel on the
same input. Every linear layer is a BatchEnsemble layer: one weight matrix W shared by all members,
plus per-member input scaling r_i, output scaling s_i and bias b_i,

    y_i = s_i ⊙ (W (r_i ⊙ x_i)) + b_i,

followed by ReLU and dropout. Each member has its own linear head; the prediction is the mean of the
members' probabilities (or outputs), and training minimises the mean of the members' losses. The
first layer's r is initialised with random ±1 signs (one per input feature), all other r and s with
ones, and the k biases start equal. The defaults are 3 blocks of 512 units, k = 32, dropout 0.1.

`experiments/tabm_gns/model.py` re-implements this so that every layer can be gated and pruned. A unit
test copies the weights of a reference `tabm.TabM.make(...)` into it and checks that the outputs and
parameter counts match.

**Gate placement.** One TabGNS gate per hidden neuron, applied after ReLU and dropout, exactly where
`tabgns.gates.GatedBlock` applies it. **A gate is shared by all k members.** A neuron of a TabM layer is
a row of the shared W together with one entry of every member's s_i and b_i and one column of the next
layer's W and r_i. Removing it for some members but not others would turn the shared W into k
different matrices and lose what makes TabM cheap, so the architecture TabGNS searches is one list of
widths for the whole ensemble.

**Gate noise (the one TabM-specific choice).** TabGNS draws one hard Gumbel sample per gate per forward
pass. In TabM there are two natural readings, and both are run:

- `tabm_gns` (**shared mask**): one mask per pass for the whole ensemble. This is TabGNS unchanged.
- `tabm_gns_member` (**per-member masks**): every member draws its own mask from the same gate
  logits. Each step then sees k = 32 sub-networks of the same distribution, which averages the gate
  gradient over more samples. It also means the members are trained on different sub-networks.

**Extraction.** As in `TabGNS.extract_net`: keep neurons with sigmoid(g) ≥ 0.5, copy their rows of W
and columns of r, s, b and the heads; with `scale: true`, multiply the kept neurons' s and b by
sigmoid(g), which equals TabGNS's scaling of W's rows and b because sigmoid(g) > 0 commutes with ReLU.
Unit tests check that the extracted TabM reproduces the supernet with the corresponding fixed mask to
1e-6, with and without scaling.

## Protocol

| | value | source |
|---|---|---|
| datasets | California (OpenML 44025, regression, 20,640 × 8), Adult (1590, binary, 48,842 × 14, 8 categorical), Otto (45548, 9 classes, 61,878 × 93) | all three are in the TabM benchmark; none is used by the replication |
| split | 60/20/20, split seed = run seed | as `experiments.datasets.split` |
| preprocessing | fitted on the training split only: quantile-normal numeric features, one-hot categoricals (missing = own level), standardised regression target | train-only fitting as in `experiments/mean_gate` |
| TabM | 3 × 512, k = 32, dropout 0.1, no numerical embeddings | `tabm.TabM.make` defaults |
| TabM training | AdamW lr 2e-3, weight decay 3e-4, batch 256, ≤ 300 epochs, early stopping (patience 20) on the ensemble's validation loss, best epoch restored | lr/wd from the `tabm` README; epochs/patience/batch from the replication |
| search space | the Large TabM itself (3 × 512) | |
| TabGNS search | weight step on a training batch, no-grad validation-loss evaluation on the next validation batch (eval mode), gate step on that batch (still eval mode); Adam lr 0.05 for gates, gate logit init −3, temperature 1, logits [g, 1]; early stopping (patience 20) on the epoch's mean validation-batch loss; the **last** epoch's net is used | all as `tabgns.TabGNS.fit` and `conf/method/tabgns.yaml`, including its quirks (P(open) = sigmoid(g − 1) while sampling, last-epoch checkpoint) |
| search weights | same optimiser and dropout as TabM training | |
| after the search | extract, warm-start fine-tune with the TabM training protocol; also train the found widths from a fresh initialisation ("scratch") | fine-tuning as `TabGNSTrainer` |
| pairing | within a seed, the supernet starts from exactly the Large TabM's initial weights (unit test); data order, gate noise and the scratch init use separate derived random streams | |
| MLP reference | `mlp_large` (5 × 512) and `mlp_gns` call `experiments.methods.run_large_mlp` / `run_tabgns` unchanged, with the replication's protocol (Adam/AdamW lr 1e-3, no dropout), on the same splits and preprocessed features | |
| seeds | 0, 1, 2 | |
| metrics | test accuracy (and log-loss) for classification; test RMSE for California, in the dataset's units log(1 + price / $100k) | |
| parameters | "params" = all trainable parameters; for TabM that is shared W + the 32 members' r, s, b + the 32 heads. "hidden" = hidden-to-hidden weights (the paper's count) | |

The California target on OpenML 44025 is log(1 + median house value / $100k) (the Grinsztajn et al.
benchmark's transformation), so its RMSE is not comparable to RMSEs in dollars.

## Reproduce

```bash
python -m unittest -v experiments.tabm_gns.test_tabm_gns   # the parity test needs `pip install tabm`
python -m experiments.tabm_gns.run -m hydra/launcher=joblib hydra.launcher.n_jobs=7 \
    dataset=otto,adult,california method=tabm_gns,tabm_gns_member,tabm_large,mlp_gns,mlp_large seed=0,1,2
python -m experiments.tabm_gns.report
```

Each job runs one (dataset, method, seed) cell on one CPU thread and writes
`results/tabm_gns/<dataset>/<method>/seed<seed>.json` with the config, metrics on validation and test,
widths, parameter counts, per-epoch architecture and loss histories, timings and provenance. Cells
that already have a result are skipped.
