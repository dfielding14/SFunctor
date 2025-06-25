# Frontier Turbulence Structure-Function Analysis

This repository contains a moderately high-performance Python pipeline for
computing anisotropic, angle–resolved structure functions (SFs) from 2D
slices of large 3D MHD simulations run with AthenaK. We use 2D slices so 
that we can make the most of the periodic boundary conditions by rolling 
the arrays, and so that we can do >2 point stencils with ease.

Key features
------------
* **24-channel histograms** capturing both perpendicular (⊥) and full-vector
  statistics for velocity, magnetic field, Alfvén speed, Elsasser variables,
  vorticity and current.
* **MPI + shared-memory multiprocessing** – embarrassingly parallel across
  slices / nodes while avoiding in-node RAM duplication.
* **Numba-accelerated kernels** (see `sf_histograms.py`) 
* **Self-describing outputs** (`.npz`) with edges, centres and provenance meta.
* **Configuration file support** – YAML configs with profiles for reproducible analysis.
* Quick-look contour/heat-map plots for slice sanity checks.

---

Installation
============
```bash
# create & activate an isolated Python environment (no conda required)
python3 -m venv .venv
source .venv/bin/activate

# upgrade core tooling
python -m pip install --upgrade pip wheel

# install runtime dependencies
pip install -r requirements.txt  # optional: edit versions first
```
Additional runtime dependencies:
* `mpi4py` compiled against the system MPI (Open MPI, MPICH, etc.).
* A BLAS/LAPACK backend (OpenBLAS, MKL) – pulled in automatically by NumPy.

A minimal `requirements.txt` is:
```
numpy>=1.23
numba>=0.58
matplotlib>=3.8
cmasher>=1.7
mpi4py>=3.1
```
Feel free to regenerate with `pip freeze > requirements.txt` once your
environment is stable.

---

Quick start
===========

Using configuration files (recommended)
```bash
# Create a configuration file
python create_config.py

# Run with configuration
python run_sf.py --config sfunctor.yaml --file_name slice_data/slice_0000.npz

# Use a specific profile for production runs
python run_sf.py --config sfunctor.yaml --profile production --file_name slice_data/slice_0000.npz
```

Single slice with command-line arguments
```bash
python run_sf.py \
  --file_name slice_data/slice_x1_-0.375_Turb_5120_beta25_dedt025_plm_0024.npz \
  --stride 2 --n_disp_total 20000 --N_random_subsamples 2000
```
Multiple slices on an HPC cluster (Slurm + OpenMPI)
```bash
mpirun -n 64 python run_sf.py \
  --slice_list slices.txt \
  --stride 4 \
  --n_disp_total 2e5 \
  --N_random_subsamples 1e4 \
  --n_processes 8
```
`slices.txt` is a plain file containing one path per line; rank *i* processes
line *i*.

Outputs land in `slice_data/` as
```
hist_ALL_<slice-stem>_ndisp<...>_Nsub<...>.npz
```
and include the histogram plus all bin edges/centres and JSON-like metadata.

---

Code organisation
=================
```
|-- run_sf.py           # MPI entry-point
|-- sf_cli.py           # Argument parsing & RunConfig dataclass
|-- sf_io.py            # Slice loader + filename metadata helpers
|-- sf_displacements.py # ℓ-bin edges & displacement generation
|-- sf_physics.py       # Derived fields (v_A, z±, ...)
|-- sf_histograms.py    # Numba kernels & channel enumeration
|-- sf_parallel.py      # Shared-memory multiprocessing
|-- structure_function_2D_slices.py  # legacy/dev notebook script
```
Module import cost is kept minimal so that only what is needed is loaded for
worker processes.

---

Development notes
=================
* The channel list lives in `sf_histograms.Channel`; add new diagnostics there.
* Unit-testing with synthetic fields is planned (see issue #1).
* Performance tips:
  – Use `stencil_width=2` for production unless dissipation scales are
    explicitly required.
  – Start with `--N_random_subsamples 500` and scale up until histograms
    converge.

---

License
=======
The code is released under the MIT License (see `LICENSE`).

<!-- Please cite **Drummond et al. (2024)** if this package contributed to your research.  -->