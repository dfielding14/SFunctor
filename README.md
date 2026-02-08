# SFunctor

SFunctor computes anisotropic, angle-resolved structure-function histograms from 2D slices of 3D MHD simulations.

All current analysis outputs use the **unified histogram API**:
- `hist`: a single `(N_CHANNELS, n_ell, n_theta, n_phi, n_delta)` count array
- `delta_bin_edges`: per-channel Δ bin edges (stored as an object array in `*.npz`)

## Requirements

- Python **3.10+** (the codebase uses PEP604 `X | None` type syntax)
- A working C toolchain for Numba
- Optional: `mpi4py` + an MPI runtime for MPI batch runs
- Slice extraction requires an AthenaK reader named `bin_convert_new` (not vendored in this repo)

## Install

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Workflow A: Distributed (no MPI, recommended)

This is the production pipeline used on Frontier/Andes (see `README_DISTRIBUTED.md` and `scripts/root/`).

1) Generate a displacement set once:
```bash
python scripts/production/generate_displacements.py \
  --n_disp_total 5000 \
  --n_ell_bins 128 \
  --Nres 1024 \
  --stencil_width 2 \
  --output displacements.npz \
  --seed 42
```

2) Run analysis for a slice (single-node example):
```bash
python scripts/production/run_node_analysis.py \
  --slice path/to/slice.npz \
  --displacements displacements.npz \
  --node_id 0 \
  --total_nodes 1 \
  --output_dir out \
  --stride 1 \
  --N_random_subsamples 2000 \
  --stencil_width 2
```

3) Combine node outputs into one result:
```bash
python scripts/production/combine_histograms_fast.py \
  --pattern "out/histogram_*.npz" \
  --output sf_results_slice.npz \
  --mode node
```

4) Plot:
```bash
python plotting_scripts/plot_structure_functions.py sf_results_slice.npz
```

## Workflow B: Python API (single slice)

```python
import numpy as np
from sfunctor.io.slice_io import load_slice_npz
from sfunctor.analysis.single_slice import analyze_slice
from sfunctor.core.histograms import Channel

slice_data = load_slice_npz("slice.npz", stride=2)
result = analyze_slice(
    slice_data,
    n_displacements=2000,
    n_random_subsamples=2000,
    stencil_width=2,
)

hist = result["hist"]
ell_edges = result["ell_bin_edges"]
delta_edges = result["delta_bin_edges"][Channel.D_V]
```

## Output format (`*.npz`)

Combined results (and per-node partials) are self-describing. Typical keys:

- `hist`: `(N_CHANNELS, n_ell, n_theta, n_phi, n_delta)` integer counts
- `channels`: list of channel names in `hist` order
- `ell_bin_edges`, `theta_bin_edges`, `phi_bin_edges`
- `delta_bin_edges`: object array of per-channel Δ edges
- `metadata`: dict with run parameters (plus `node_info`/`node_infos` for distributed runs)

Example: compute ⟨|δv|²⟩(ℓ) from binned counts:
```python
import numpy as np
from sfunctor.core.histograms import Channel

data = np.load("sf_results_slice.npz", allow_pickle=True)
hist = data["hist"]
ell_edges = data["ell_bin_edges"]
delta_edges = np.asarray(data["delta_bin_edges"][Channel.D_V], dtype=float)

counts = hist[Channel.D_V].sum(axis=(1, 2))  # (n_ell, n_delta)
delta_centers = np.sqrt(delta_edges[:-1] * delta_edges[1:])
s2 = (counts * delta_centers**2).sum(axis=-1) / counts.sum(axis=-1)
ell_centers = 0.5 * (ell_edges[:-1] + ell_edges[1:])
```

## Channels

Channel definitions live in `sfunctor/core/histograms.py` (`Channel`, `N_CHANNELS=26`). The saved `channels` list matches that enum order.

## Slice extraction

The extraction helper `sfunctor.io.extract.extract_2d_slice()` depends on an external AthenaK reader module `bin_convert_new`.
For a minimal CLI wrapper, see `scripts/production/extractor.py`.

## Documentation

- `docs/README.md:1` (index)
- `docs/ANISOTROPIC_STRUCTURE_FUNCTIONS.md:1`
- `docs/MEASURING_ANISOTROPY.md:1`
- `docs/THEORY_COMPARISONS.md:1`
- `docs/HISTOGRAM_ANALYSIS_GUIDE.md:1`
- `docs/WORKED_EXAMPLE_ANISOTROPY.md:1`
