# TabGNS: Gated Neuron Selection for tabular data

TabGNS is a neural arhcitecture search method for multi layer perceptrons. It uses gradient descent to decide on a models architecture at level of neurons. The paper describing the method will be published at NOMS 2026.

This repository contains an implementation of TabGNS. The method can be run on simple task using `eval.py`. For comparison, the script trains two separete models; TabGNS and the largest MLP within the search. To run TabGNS using hyperparameters used in the paper, use configurations in `conf/paper.yaml` instead. This can be done by running `python eval.py --config-name=paper`.

An example use of TabGNS is provided in [nb.ipynb](nb.ipynb).

To reproduce the paper's tables and figures, including the TabNAS and AgE baselines, see [experiments/README.md](experiments/README.md).