# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SFunctor is a high-performance Python pipeline for computing anisotropic, angle-resolved structure functions from 2D slices of 3D magnetohydrodynamic (MHD) simulations. It's designed for analyzing AthenaK simulation outputs with a focus on turbulence analysis.

## Common Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install numpy numba matplotlib cmasher mpi4py
```

### Running the Analysis
```bash
# Single slice analysis (no MPI)
python run_sf.py --file_name slice_data/slice_0000.npz --stride 2

# Multi-slice analysis with MPI
mpirun -n 64 python run_sf.py --slice_list slice_list.txt --stride 2

# Simplified analysis (for testing, no MPI/Numba)
python simple_sf_analysis.py --file_name slice_data/slice_0000.npz

# Extract 2D slice from 3D data
python extract_2d_slice.py --input_file data/Turb.hydro.00100.bin --output_file slice.npz --dimension y --slice_value 0

# Visualize results
python visualize_sf_results.py results_sf.npz
```

### Testing Approach
There's no formal test suite. For testing changes:
1. Use `demo_pipeline.py` for end-to-end workflow testing
2. Use `simple_sf_analysis.py` to test without MPI/Numba dependencies
3. Compare outputs between full and simplified versions
4. Test data available in `slice_data/` directory

## High-Level Architecture

### Core Pipeline Flow
1. **Data Input**: 2D slices from 3D MHD simulations (.npz format)
2. **Physics Computation**: Derive fields (vorticity, current, Alfvén variables)
3. **Displacement Generation**: Random vectors with configurable binning
4. **Structure Function Calculation**: 24-channel histograms via Numba kernels
5. **Output**: Self-describing .npz files with complete metadata

### Parallelization Strategy
- **Level 1**: MPI for multi-node distribution (embarrassingly parallel across slices)
- **Level 2**: Shared-memory multiprocessing within nodes
- Graceful fallback when MPI not available

### Key Modules and Their Roles
- `run_sf.py`: Main entry point with MPI support
- `sf_cli.py`: Configuration management and argument parsing
- `sf_physics.py`: Physics calculations (must maintain consistency with AthenaK)
- `sf_histograms.py`: Performance-critical Numba kernels (24 channels)
- `sf_parallel.py`: Shared-memory parallelization logic
- `sf_displacements.py`: Displacement vector generation and binning

### Performance Considerations
- Numba JIT compilation for histogram kernels (first run slower)
- Minimal imports in worker processes to reduce overhead
- Configurable stencil widths (2, 3, or 5 points) affect memory/accuracy trade-off
- Memory usage scales with number of displacement vectors and bins

### Physics Channels Computed
The pipeline computes 24 structure function channels including:
- Velocity, magnetic field, density increments
- Elsasser variables (z+ and z-)
- Vorticity and current density
- Various cross products and angle-resolved statistics

When modifying physics calculations, ensure consistency with the channel definitions in `sf_histograms.py`.