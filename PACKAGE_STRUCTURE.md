# SFunctor Package Structure

After restructuring, the SFunctor codebase is now organized as a proper Python package with clear separation of concerns.

## Directory Structure

```
SFunctor/
├── sfunctor/                    # Main package
│   ├── __init__.py              # Package initialization
│   ├── core/                    # Core computational modules
│   │   ├── __init__.py
│   │   ├── histograms.py        # Numba-accelerated histograms
│   │   ├── physics.py           # MHD physics calculations
│   │   └── parallel.py          # Shared-memory parallelization
│   ├── analysis/                # Analysis pipelines
│   │   ├── __init__.py
│   │   ├── batch.py             # MPI batch processing
│   │   ├── single_slice.py      # Single slice analysis
│   │   └── simple.py            # Simplified analysis (no Numba)
│   ├── io/                      # Input/output utilities
│   │   ├── __init__.py
│   │   ├── slice_io.py          # Slice data loading
│   │   └── extract.py           # 2D slice extraction from 3D data
│   ├── visualization/           # Plotting and visualization
│   │   ├── __init__.py
│   │   └── plots.py             # Structure function plots
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       ├── cli.py               # Command-line interface
│       └── displacements.py     # Displacement vector generation
├── tests/                       # Unit tests
│   ├── __init__.py
│   ├── test_sf_io.py
│   ├── test_sf_physics.py
│   └── test_sf_displacements.py
├── setup.py                     # Package installation script
├── requirements.txt             # Core dependencies
├── requirements-dev.txt         # Development dependencies
├── MANIFEST.in                  # Package file inclusion rules
└── run_analysis.py             # Backward compatibility wrapper
```

## Module Organization

### Core (`sfunctor.core`)
Performance-critical components with minimal dependencies:
- **histograms.py**: Numba JIT-compiled histogram computation
- **physics.py**: MHD physics calculations (Alfvén velocity, Elsasser variables)
- **parallel.py**: Shared-memory multiprocessing for single-node parallelism

### Analysis (`sfunctor.analysis`)
High-level analysis pipelines:
- **batch.py**: MPI-capable batch processing for clusters
- **single_slice.py**: Simple interface for analyzing one slice
- **simple.py**: Pure Python implementation for testing/education

### I/O (`sfunctor.io`)
Data loading and format conversion:
- **slice_io.py**: NPZ file loading with validation
- **extract.py**: Extract 2D slices from 3D AthenaK binary files

### Visualization (`sfunctor.visualization`)
Plotting and result visualization:
- **plots.py**: Structure function plots and diagnostics

### Utils (`sfunctor.utils`)
Helper functions and utilities:
- **cli.py**: Command-line argument parsing
- **displacements.py**: Displacement vector generation

## Import Examples

```python
# High-level imports from package root
from sfunctor import load_slice_npz, analyze_slice, plot_structure_functions

# Subpackage imports
from sfunctor.core import compute_vA, compute_z_plus_minus
from sfunctor.analysis import batch_analyze
from sfunctor.utils import build_displacement_list

# Direct module imports
from sfunctor.core.histograms import Channel, compute_histogram_for_disp_2D
from sfunctor.io.slice_io import parse_slice_metadata
```

## Installation

The package can now be installed in development mode:

```bash
pip install -e .
```

Or for production:

```bash
pip install .
```

With development dependencies:

```bash
pip install -e ".[dev]"
```

## Command-Line Entry Points

After installation, the following commands are available:

- `sfunctor`: Main batch analysis (replaces `run_sf.py`)
- `sfunctor-simple`: Simplified analysis without Numba
- `sfunctor-viz`: Visualization tool

## Benefits of New Structure

1. **Clear Organization**: Related functionality is grouped together
2. **Easier Testing**: Each module can be tested independently
3. **Better Imports**: No more path manipulation needed
4. **Pip Installable**: Can be installed like any Python package
5. **Dependency Management**: Clear separation of core vs optional dependencies
6. **Extensibility**: Easy to add new analysis methods or I/O formats
7. **Documentation**: Structure supports automatic API doc generation

## Migration from Old Structure

| Old File | New Location |
|----------|--------------|
| `sf_histograms.py` | `sfunctor/core/histograms.py` |
| `sf_physics.py` | `sfunctor/core/physics.py` |
| `sf_parallel.py` | `sfunctor/core/parallel.py` |
| `sf_io.py` | `sfunctor/io/slice_io.py` |
| `sf_cli.py` | `sfunctor/utils/cli.py` |
| `sf_displacements.py` | `sfunctor/utils/displacements.py` |
| `run_sf.py` | `sfunctor/analysis/batch.py` |
| `simple_sf_analysis.py` | `sfunctor/analysis/simple.py` |
| `visualize_sf_results.py` | `sfunctor/visualization/plots.py` |
| `extract_2d_slice.py` | `sfunctor/io/extract.py` |

Backward compatibility is maintained through wrapper scripts in the root directory.