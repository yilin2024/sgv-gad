# Zero-Shot Graph Anomaly Detection with a Spectral Graph Vocabulary

This is the code for the method in the submission. A single model is trained on labelled **source** graphs and then
scores anomalies on unseen **target** graphs **zero-shot**, with no fine-tuning and no target labels. Target graphs can
have a different attribute space and a different size.

No datasets or trained weights are included.

## Method in brief

Each graph goes through the same label-free tokenizer (`sgv/tokenizer.py`), which produces:

| Block | Width | What it is |
|---|---|---|
| Feature block `F` | 64 | PCA of the column-scaled attributes (uncentred projection). Columns are ordered from smoothest to roughest on the graph and their signs are fixed. When a graph has 64 or fewer attributes, the raw attributes are used, zero-padded. |
| **Spectral Graph Vocabulary** `Ω` | 64 | For every column `f` of `F`, each node's share of the column's local Dirichlet energy, `log(1 + ε_i / mean_k δ_k)`. With `v = D^{-1/2} f`: `ε_i = Σ_{j∈N(i)} (v_i − v_j)²` and `δ_i = v_i² + Σ_{j∈N(i)} v_j²`. |
| Joint energy | 9 | Energy scalars of the whole column block: energy ratio at hops 0, 1, 2; Beta-wavelet band fractions β₀..β₄; log local power. They are invariant to any orthogonal change of attribute basis. |
| Structure | 9 | Random-walk return profile (8 steps) and log degree. |

Every input column is rank-normalised **per graph**: its ECDF rank `r` becomes `2r − 1`, and `|2r − 1|` is appended.
This uses only the graph's own unlabelled nodes, so the model sees no raw attribute values and no statistics from
other graphs.

The backbone (`sgv/model.py`) stacks 3 pre-LayerNorm blocks. Each block mixes, with a learnable weight μ:
- **graph attention** over the (sampled) neighbourhood;
- a **routed spectral mixture of experts**. There are 5 Chebyshev graph filters, initialised exactly to the Beta
  wavelet kernels β₀..β₄. β₀ is shared; β₁..β₄ are weighted per node by a softmax router whose temperature is annealed
  from 1.0 to 0.5.

An MLP head on the final node state gives the anomaly logit.

Training (`sgv/train.py`) uses class-weighted binary cross-entropy on the source graphs, with one Adam step per source
graph per epoch. For the first 5 epochs, a KL term pulls the router toward each node's band-energy profile.

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.12, PyTorch 2.8, NumPy 1.26, SciPy 1.16, scikit-learn 1.8. A CUDA GPU is used when available;
without one, everything runs on CPU automatically.

## Data format

Put one `.mat` file per graph in one directory. The graph's name is the file stem.

| Key | Content |
|---|---|
| `Network` | `n × n` adjacency, dense or sparse. It is symmetrised; self-loops and edge weights are dropped. |
| `Attributes` | `n × d` node attributes, dense or sparse. `d` may differ between graphs. |
| `Label` (or `gnd`) | `n` labels: 1 = anomalous, 0 = normal. |
| `keep_mask` *(optional)* | Indices (or a boolean mask) of the nodes to keep. The graph is restricted to the subgraph they induce. |

This is the format used by common graph anomaly detection benchmarks.

## Usage

```bash
# check the pipeline end to end on synthetic graphs (no data needed; ~1 min on CPU)
python examples/make_toy.py --out toy_data
python -m sgv.train --data_root toy_data --sources toy_a,toy_b --targets toy_c --set train.epochs=5

# train on your source graphs, score your target graphs zero-shot
python -m sgv.train --data_root /path/to/mat_dir \
    --sources graphA,graphB,graphC --targets graphD,graphE,graphF \
    --seeds 3 --out runs/my_split
```

- A graph listed in both `--sources` and `--targets` is removed from the targets.
- Tokens are cached per graph in `--cache_dir` (default `cache/`; pass `''` to disable caching).
- Any hyperparameter can be overridden with `--set section.field=value`, for example `--set train.epochs=100` or
  `--set model.hidden=192`.

**Outputs** (in `--out`):
- `config.yaml`: the full configuration and the split.
- `results.csv`: one row per seed × target × `when`, with AUROC and AUPRC. `when` is `last` (final epoch) or `best`
  (the epoch with the best mean AUPRC on the 10% source hold-out).
- A per-target summary printed at the end.

**Protocol.** Nothing computed from a target graph reaches training. Target labels are read only to compute the
metrics.

## Hyperparameters (defaults, `sgv/config.py`)

| Group | Setting |
|---|---|
| Tokenizer | `z_dim` 64, Beta order C = 4, numerical-rank tolerance 1e-6, random-walk profile 8 steps (64 probes) |
| Backbone | hidden 128, 3 blocks, 4 attention heads, Chebyshev order K = 8, router hidden 32, FFN ×4, dropout 0.1 |
| Router | temperature 1.0 → 0.5 (linear), warm-up KL weight 0.1 for 5 epochs |
| Attention | 64 sampled neighbours per node while training, all neighbours at evaluation, bf16 on GPU |
| Training | Adam, lr 1e-3, weight decay 0, gradient clipping 1.0, 200 epochs, 10% source hold-out |

## Code layout

```
sgv/
  config.py     hyperparameters and --set overrides
  data.py       .mat loader
  graph_ops.py  sparse normalised adjacency, random walk and Beta-wavelet operators
  tokenizer.py  feature block, Spectral Graph Vocabulary, joint energies, structural profile
  model.py      attention, routed Chebyshev spectral experts, backbone and head
  train.py      training on sources and zero-shot evaluation on targets
examples/
  make_toy.py   synthetic graphs for a smoke test
```
