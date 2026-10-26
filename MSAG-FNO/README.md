# MSAG-FNO

**A Multi-Scale Adaptive Gated Fourier Neural Operator for Urban Flood Prediction**

Reference implementation of the MSAG-FNO surrogate model and its transfer-learning
variant for urban pluvial flood forecasting. Built with PyTorch.

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/pytorch-2.x-ee4c2c">
</p>

---

## Overview

High-fidelity urban flood simulation with hydrodynamic models (e.g. SWMM / 2D
solvers) is accurate but computationally expensive, which makes it impractical for
real-time warning and large-scale scenario screening. This repository contains a
neural-operator surrogate that learns the mapping

```
(terrain + rainfall forcing)  ->  (water depth over time, maximum inundation depth)
```

directly in function space, so that a single forward pass replaces a full
hydrodynamic simulation.

Two ingredients make up the method:

1. **Multi-scale spectral learning.** Standard Fourier neural operators truncate the
   spectrum at a single cutoff, which forces one fixed receptive scale over the whole
   domain. Urban flooding is simultaneously broad (overland flow, drainage
   catchments) and local (streets, underpasses, inlets), so MSAG-FNO combines
   spectral branches with different frequency truncations to capture both regimes.
2. **Adaptive gating.** A lightweight gating module (global average pooling + MLP,
   together with a spatial descriptor) re-weights the spectral response per channel
   and per location, letting the network decide *where* and *for which variable*
   fine scale detail matters.

The second contribution is a **transfer-learning** scheme: an MSAG-FNO pre-trained on
a source domain (e.g. one city / one drainage layout) is fine-tuned on a target
domain, where only a subset of the operator layers is unfrozen. This makes the model
reusable across cities without retraining from scratch.

---

## Highlights

- **Function-space surrogate** — one forward pass replaces a hydrodynamic simulation.
- **Multi-scale spectral truncation** — coarse and fine spectral branches combined in
  every operator block.
- **Adaptive gating** — channel-wise and spatially varying re-weighting of the
  spectral response.
- **Physics-aware output** — a `softplus` head plus a negatively initialised output
  bias keep predicted water depths non-negative and biased towards dry cells, which
  dominate urban domains.
- **Masked, depth-aware loss** — cells outside the valid domain are ignored, and the
  depth time series and the maximum-depth channel are normalised separately.
- **Ablation-ready** — the same class instantiates the plain `FNO`, `MS-FNO`
  (multi-scale only) and `AG-FNO` (gating only) baselines via a single `kind` flag.
- **Fine-tuning strategies** — `all / none / first1 / first2 / first3` layer-freezing
  schemes, with an integrity check that frozen weights really stayed frozen.

---

## Repository structure

```
MSAG-FNO/
├── code/
│   ├── FNO.py                        # SpectralConv, plain FNO baseline, shared training loop
│   ├── MSAG-FNO.py                   # MSAG-FNO operator block and full model
│   └── Fine-tuning-based MSAG-FNO.py # transfer-learning / layer-freezing utilities
├── figures/
│   ├── Figure1.png                   # Overall framework, results and transfer-learning evaluation
│   ├── Figure2.png                   # MSAG-FNO layer in detail
│   └── Figure3.png                   # Pre-training -> fine-tuning workflow
└── README.md
```

| File | Provides |
| --- | --- |
| `code/FNO.py` | `SpectralConv`, `FNO`, `masked_depth_loss`, `fit_model`, `seed_all`, `train_fno` |
| `code/MSAG-FNO.py` | `MSAGOperatorBlock`, `MSAGFNO`, `train_msag_fno` |
| `code/Fine-tuning-based MSAG-FNO.py` | `set_finetune`, `prepare_finetuning`, `load_pretrained_model`, `fine_tune` |

> **Note on file names.** `MSAG-FNO.py` and `Fine-tuning-based MSAG-FNO.py` contain
> spaces and hyphens, so they cannot be imported with a normal `import` statement.
> `Fine-tuning-based MSAG-FNO.py` already loads its sibling via `importlib`.
> See [Loading the modules](#1-loading-the-modules) for a copy-paste helper.

---

## Method

### MSAG-FNO operator block

Each block takes an `(N, C, H, W)` feature map and computes

```
x  ->  fine spectral branch   (full resolution, cutoff K)
   ->  coarse spectral branch (x2 average-pooled, cutoff K/2, upsampled back)
   ->  average the two branches
   ->  multiply by adaptive gate  g(x) = channel_gate(x) * spatial_gate([mean(x), max(x)])
   ->  +  local 1x1 convolution (residual)
   ->  GELU
```

Four such blocks are stacked between a linear lifting layer and a two-layer
projection head. See `figures/Figure2.png` for the detailed diagram.

### Baselines and ablations

`MSAGFNO(kind=...)` switches the block configuration, which keeps the comparison
between variants exactly apples-to-apples (same width, modes, depth and training loop):

| `kind` | Multi-scale branches | Adaptive gating |
| --- | :---: | :---: |
| `FNO` (in `FNO.py`) | ✗ | ✗ |
| `MS-FNO` | ✓ | ✗ |
| `AG-FNO` | ✗ | ✓ |
| `MSAG-FNO` | ✓ | ✓ |

### Transfer learning

```
source domain  --pre-train-->  MSAG-FNO weights
                                  |
                                  v
target domain  --fine-tune-->  MSAG-FNO (partially unfrozen)
```

| Strategy | Frozen | Trainable |
| --- | --- | --- |
| `none`  (TL2) | everything | nothing — evaluates the source model as-is |
| `first1` (TL3) | blocks 2–4, lift, head | block 1 |
| `first2` (TL4) | blocks 3–4, lift, head | blocks 1–2 |
| `first3` (TL5) | block 4, lift, head | blocks 1–3 |
| `all`   (TL1) | nothing | whole network |

See `figures/Figure3.png`.

---

## Installation

```bash
git clone <this-repository>
cd MSAG-FNO
pip install torch numpy          # CPU build is enough to run the snippets below
```

Requirements: Python ≥ 3.9, PyTorch ≥ 1.13 (any recent 2.x recommended), NumPy.
No other dependencies.

---

## Data format

The repository ships the model and the training utilities only; the dataset and the
preprocessing pipeline that builds these tensors are **not** included (see
[Data availability](#data-availability)). Everything is expressed as a plain tuple of
three tensors, so any loader that produces them will work:

```python
batch = (inputs, target, mask)
```

| Tensor | Shape | dtype | Meaning |
| --- | --- | --- | --- |
| `inputs` | `(N, 17, H, W)` | `float32` | feature tensor produced by the spatiotemporal normalisation and physics-aware embedding & fusion stage |
| `target` | `(N, 26, H, W)` | `float32` | channels `0 … 24`: water depth at 25 successive time steps; channel `25`: maximum inundation depth |
| `mask` | `(N, 26, H, W)` | `bool` | `True` for supervised cells (valid domain, wet or dry); `False` for cells to be ignored |

Contract enforced by the code:

- `inputs` and `target` must share `(H, W)`; masks must have exactly the `target` shape.
- All tensors must already be on the model's device and match the model's real dtype.
- `inputs` and `target` must be finite everywhere, including masked cells.
- `sequence_channels` (default `25`) splits the output channels into the
  *time-series* group and the *maximum-depth* group; it must satisfy
  `0 < sequence_channels < outputs`.

### Loss

`masked_depth_loss` computes a masked MSE per group and averages the two with equal
weight:

```
L = 0.5 * MSE_mask(depth_{0..24})  +  0.5 * MSE_mask(depth_25)
```

so the maximum-depth channel is not swamped by the 25 time-step channels.

---

## Usage

### 1. Loading the modules

Because of the hyphens in the file names, either rename the files or use this helper:

```python
import sys, importlib.util
from pathlib import Path

sys.path.insert(0, "code")          # required: the modules import each other by bare name

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

fno  = load("code/FNO.py", "fno")
msag = load("code/MSAG-FNO.py", "msag_fno")
ft   = load("code/Fine-tuning-based MSAG-FNO.py", "msag_finetune")
```

Renaming the files to `fno.py`, `msag_fno.py` and `finetune_msag_fno.py` makes plain
imports work instead:

```python
from FNO import FNO, masked_depth_loss, train_fno
from msag_fno import MSAGFNO, MSAGOperatorBlock, train_msag_fno
from finetune_msag_fno import STRATEGIES, load_pretrained_model, fine_tune
```

### 2. Train an MSAG-FNO from scratch

```python
import torch
from MSAGFNO_module import MSAGFNO, train_msag_fno   # see "Loading the modules"

model = MSAGFNO(
    inputs=17,      # input channels  (feature tensor)
    width=12,       # hidden width
    modes=5,        # spectral cutoff of the fine branch
    outputs=26,     # 25 depth steps + 1 maximum depth
    kind="MSAG-FNO",
)

# train_batch / val_batch: (inputs, target, mask), already on the right device
model = train_msag_fno(
    model, train_batch, val_batch,
    epochs=60, learning_rate=3e-3, sequence_channels=25, seed=2026,
)

prediction = model(val_batch[0])                    # (N, 26, H, W)
prediction, features = model(val_batch[0], features=True)   # + pooled latent (N, width)
```

The same call signature trains the plain baseline — only the class changes:

```python
from FNO import FNO, train_fno
model = train_fno(FNO(inputs=17, width=12, modes=5, outputs=26),
                  train_batch, val_batch, epochs=60)
```

### 3. Fine-tune on a target domain

```python
import torch

checkpoint = torch.load("pretrained_msag_fno.pt", map_location="cpu")
model = ft.load_pretrained_model(
    checkpoint, inputs=17, width=12, modes=5, outputs=26,
    strategy="first2",            # or "all" / "none" / "first1" / "first3"
    device="cuda",
)

model = ft.fine_tune(
    model, target_train_batch, target_val_batch,
    strategy="first2", epochs=40, learning_rate=3e-3,
    sequence_channels=25, seed=2026,
)
```

`load_pretrained_model` accepts either a bare `state_dict` or a dict containing
`{"state_dict": ...}` plus optional `kind` / `width` / `modes` metadata, which are
checked for consistency before loading (strict, no silent partial loads).

To freeze/unfreeze by hand:

```python
trainable = ft.set_finetune(model, "first2")   # returns the names now requiring grad
model = ft.prepare_finetuning(pretrained_model, "first3")   # deep-copied, so the source stays intact
```

### 4. Ablation runs

```python
for kind in ("MS-FNO", "AG-FNO", "MSAG-FNO"):
    model = MSAGFNO(kind=kind)
    model = train_msag_fno(model, train_batch, val_batch, epochs=60)
```

---

## Default hyper-parameters

| Setting | Value |
| --- | --- |
| Operator blocks | 4 |
| Hidden width | 12 |
| Spectral cutoff (fine branch) | 5 |
| Spectral cutoff (coarse branch) | `max(2, modes // 2)` |
| Input / output channels | 17 / 26 |
| Optimiser | Adam |
| Learning rate | 3e-3 |
| Epochs (from scratch) | 60 |
| Epochs (fine-tuning) | 40 |
| Gradient clipping | global norm 1.0 |
| Seed | 2026 |
| Model size | ≈ 69k real-valued parameters (default configuration) |

Training runs a single batch per epoch (full-batch surrogate fitting), evaluates on
the validation batch after every epoch, keeps the best-validation weights, and
restores them at the end.

---

## Built-in safeguards

The utilities fail loudly instead of silently producing nonsense:

- shape / dtype / device / finiteness validation on every batch before training
  (`_validate_batch`);
- `ValueError` on inconsistent positive-integer hyper-parameters;
- best-checkpoint restore so a diverging epoch cannot corrupt the returned model;
- `FloatingPointError` if the training or validation loss becomes non-finite;
- `RuntimeError` if any parameter frozen for fine-tuning changed during training;
- `TypeError` if a fine-tuning strategy is applied to a model that is not a full
  MSAG-FNO with four gated multi-scale blocks.

---

## Results

Quantitative comparison (NSE / RMSE against a hydrodynamic reference, for water depth
and maximum depth) and the transfer-learning evaluation are reported in the paper.
The prediction maps and hydrographs for the three validation events are in
`figures/Figure1.png`.

---

## Data availability

The urban flood dataset, the hydrodynamic reference simulations and the trained
checkpoints are **not** part of this repository. [Describe here: available on
request / released with the paper / link to a data repository.]

---

## Citation

```bibtex
@article{[key],
  title   = {A Multi-Scale Adaptive Gated Fourier Neural Operator for Urban Flood Prediction},
  author  = {[authors]},
  journal = {[journal]},
  year    = {[year]},
  doi     = {[doi]}
}
```

## License

[LICENSE — e.g. MIT / Apache-2.0 / CC BY-NC-4.0]

## Contact

[Name] — [email] — [affiliation]

---

## Acknowledgements

[Funding / project / computing resources]
