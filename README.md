# Frontier Turbulence Structure-Function Analysis

This repository contains a moderately high-performance Python pipeline for
computing anisotropic, angle–resolved structure functions (SFs) from 2D
slices of large 3D MHD simulations run with AthenaK. We use 2D slices so
that we can make the most of the periodic boundary conditions by rolling
the arrays, and so that we can do >2 point stencils with ease.

## Key features

- **24-channel histograms** capturing both perpendicular (⊥) and full-vector
  statistics for velocity, magnetic field, Alfvén speed, Elsasser variables,
  vorticity and current.
- **Multiprocessing by default** – Process multiple slices in parallel on a single
  node using Python's multiprocessing.
- **Optional MPI support** – Scale across multiple nodes on HPC clusters when
  installed with `pip install sfunctor[mpi]`.
- **Shared-memory optimization** – Avoids in-node RAM duplication when processing
  large arrays.
- **Numba-accelerated kernels** (see `sf_histograms.py`)
- **Self-describing outputs** (`.npz`) with edges, centres and provenance meta.
- Quick-look contour/heat-map plots for slice sanity checks.

---

# Installation

### From source (recommended for development)

```bash
# Clone the repository
git clone https://github.com/yourusername/sfunctor.git
cd sfunctor

# Create & activate an isolated Python environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Upgrade core tooling
python -m pip install --upgrade pip wheel

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

### Standard installation

```bash
# Create & activate an isolated Python environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package (without MPI support)
pip install sfunctor

# Or install with MPI support (requires MPI libraries)
pip install sfunctor[mpi]
```

### Using requirements.txt (legacy method)

```bash
# Install runtime dependencies only
pip install -r requirements.txt
```

### System Dependencies

- **For MPI support**: You need MPI libraries installed on your system:
  - macOS: `brew install mpich` or `brew install openmpi`
  - Ubuntu/Debian: `sudo apt-get install mpich` or `sudo apt-get install openmpi-bin libopenmpi-dev`
  - RedHat/CentOS: `sudo yum install mpich-devel` or `sudo yum install openmpi-devel`
- **BLAS/LAPACK backend**: Pulled in automatically by NumPy (OpenBLAS, MKL, etc.)

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

# Quick start

## Using the installed package commands

Single slice

```bash
sfunctor \
  --file_name slice_data/slice_x1_-0.375_Turb_5120_beta25_dedt025_plm_0024.npz \
  --stride 2 --n_disp_total 20000 --N_random_subsamples 2000
```

Multiple slices with multiprocessing (default, single node)

```bash
sfunctor \
  --slice_list slices.txt \
  --stride 4 \
  --n_disp_total 2e5 \
  --N_random_subsamples 1e4 \
  --n_processes 8
```

Multiple slices on an HPC cluster (requires MPI)

```bash
mpirun -n 64 sfunctor \
  --slice_list slices.txt \
  --stride 4 \
  --n_disp_total 2e5 \
  --N_random_subsamples 1e4 \
  --n_processes 8
```

## Using the scripts directly (without installation)

Single slice

```bash
python run_sf.py \
  --file_name slice_data/slice_x1_-0.375_Turb_5120_beta25_dedt025_plm_0024.npz \
  --stride 2 --n_disp_total 20000 --N_random_subsamples 2000
```

Multiple slices with multiprocessing (default, single node)

```bash
python run_sf.py \
  --slice_list slices.txt \
  --stride 4 \
  --n_disp_total 2e5 \
  --N_random_subsamples 1e4 \
  --n_processes 8
```

Multiple slices on an HPC cluster (requires MPI)

```bash
mpirun -n 64 python run_sf.py \
  --slice_list slices.txt \
  --stride 4 \
  --n_disp_total 2e5 \
  --N_random_subsamples 1e4 \
  --n_processes 8
```

`slices.txt` is a plain file containing one path per line.

- With multiprocessing (default): slices are distributed among worker processes
- With MPI: rank _i_ processes slices where `i % size == rank`

Outputs land in `slice_data/` as

```
hist_ALL_<slice-stem>_ndisp<...>_Nsub<...>.npz
```

and include the histogram plus all bin edges/centres and JSON-like metadata.

---

# Code organisation

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

# Development notes

## Code Quality Tools

This project uses several tools to maintain code quality:

### Quick Start

```bash
# Install development dependencies
pip install -e ".[dev]"

# Format all code
make format

# Run all checks
make check

# See all available commands
make help
```

### Available Make Commands

- `make format` - Auto-format code with black and isort
- `make format-check` - Check formatting without changing files
- `make lint` - Run flake8 linting
- `make type-check` - Run mypy type checking
- `make check` - Run all checks (lint + type-check)
- `make clean` - Remove generated files and caches

### Pre-commit Hooks (optional)

```bash
# Install pre-commit hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

## Technical Notes

- The channel list lives in `sf_histograms.Channel`; add new diagnostics there.
- Unit-testing with synthetic fields is planned (see issue #1).
- Performance tips:
  – Use `stencil_width=2` for production unless dissipation scales are
  explicitly required.
  – Start with `--N_random_subsamples 500` and scale up until histograms
  converge.

---

# License

No license has been assigned to this code yet.

<!-- Please cite **Drummond et al. (2024)** if this package contributed to your research.  -->
