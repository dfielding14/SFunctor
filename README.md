# SFunctor - Structure Function Analysis for MHD Turbulence

A high-performance Python pipeline for computing anisotropic, angle-resolved structure functions from 2D slices of 3D magnetohydrodynamic (MHD) simulations. Designed for analyzing AthenaK simulation outputs with a focus on turbulence analysis.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
  - [Step 1: Extract 2D Slices](#step-1-extract-2d-slices)
  - [Step 2: Compute Structure Functions](#step-2-compute-structure-functions)
  - [Step 3: Visualize Results](#step-3-visualize-results)
- [Physics and Channels](#physics-and-channels)
- [Configuration](#configuration)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)
- [Examples](#examples)

## Overview

SFunctor computes structure functions - statistical measures of field increments at different spatial separations - which are fundamental tools for analyzing turbulence. The pipeline:

1. Extracts 2D slices from 3D MHD simulation data
2. Computes derived fields (vorticity, current density, Elsasser variables, magnetic curvature, density gradients)
3. Calculates structure functions using efficient histogram methods
4. Provides comprehensive visualization tools

## Features

- **High Performance**: Numba JIT compilation for critical loops, MPI support for multi-node parallelization
- **Comprehensive Physics**: 10 magnitude channels and 18 cross-product channels including:
  - Velocity, magnetic field, and density increments
  - Perpendicular components for anisotropy studies
  - Elsasser variables (z+ and z-)
  - Vorticity and current density from neighboring slices
  - Magnetic curvature (b·∇)b
  - Density gradients ∇ρ
- **Angle-Resolved Analysis**: Full 3D displacement vector binning (magnitude, polar angle, azimuthal angle)
- **Flexible Configuration**: YAML-based configuration with profiles for different use cases
- **Robust I/O**: Handles AthenaK binary formats and self-describing NPZ files
- **Memory Efficient**: Shared memory multiprocessing to avoid data duplication

## Installation

### Prerequisites
- Python 3.8 or higher
- C compiler (for Numba)
- MPI implementation (optional, for multi-node runs)

### Option 1: Install from source (recommended)
```bash
# Clone the repository
git clone <repository-url>
cd SFunctor

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode (this installs all dependencies automatically)
pip install -e .
```

This command installs the package in "editable" mode and automatically installs all required dependencies listed in requirements.txt, including numpy, numba, matplotlib, scipy, pyyaml, etc.

### Option 2: Install dependencies manually
If you prefer to install dependencies separately without installing the package:
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages individually
pip install numpy numba matplotlib scipy pyyaml
pip install mpi4py  # Optional, for MPI support
pip install cmasher h5py  # For visualization and HDF5 support

# Then add SFunctor to your Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}"
```

### Option 3: Using requirements.txt
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Extract a single 2D slice
```bash
python run_extract_slice.py \
    --sim_name Turb_320_beta100_dedt025_plm \
    --axis 3 \
    --offset 0.0 \
    --file_numbers 0
```

### 2. Compute structure functions
```bash
# Single slice analysis
python run_analysis.py \
    --file_name slice_data/Turb_320_beta100_dedt025_plm_axis3_slice0_file0000.npz \
    --stride 2 \
    --n_disp_total 1000 \
    --N_random_subsamples 1000

# Multiple slices with MPI (if available)
mpirun -n 8 python run_analysis.py \
    --slice_list slice_list.txt \
    --stride 2
```

### 3. Visualize results
```bash
python visualize_sf_results.py results/sf_results_*.npz
```

## Detailed Usage

### Step 1: Extract 2D Slices

The extraction step reads 3D simulation data and extracts 2D slices at specified positions. It also computes derived fields that require information from neighboring slices.

#### Basic extraction:
```bash
python run_extract_slice.py \
    --sim_name <simulation_name> \
    --axis <1|2|3> \
    --offset <position> \
    --file_numbers <start>-<end>
```

#### Parameters:
- `--sim_name`: Base name of simulation files (without .bin extension)
- `--axis`: Slice orientation
  - 1 = yz-plane (x-normal)
  - 2 = xz-plane (y-normal)
  - 3 = xy-plane (z-normal)
- `--offset`: Position along the normal direction [-0.5, 0.5]
- `--file_numbers`: File numbers to process (e.g., "0-10" or "5,10,15")

#### Advanced options:
```bash
# Extract multiple slices at different positions
python run_extract_slice.py \
    --sim_name Turb_320_beta100_dedt025_plm \
    --axis 1,2,3 \
    --offset -0.25,0.0,0.25 \
    --file_numbers 0-100

# Specify data directory and output location
python run_extract_slice.py \
    --sim_name MySim \
    --data_dir /path/to/simulation/data \
    --cache_dir /path/to/output/slices \
    --axis 3 \
    --offset 0.0

# Extract with specific options
python run_extract_slice.py \
    --sim_name Turb_5120_beta25_dedt025_plm \
    --axis 1 \
    --offset 0.0 \
    --file_numbers 50 \
    --neighbor_offsets 0.002,0.004  # For derivative calculations
```

#### Extracted fields:
Each slice NPZ file contains:
- **Primary fields**: 
  - `velx`, `vely`, `velz` - Velocity components
  - `bcc1`, `bcc2`, `bcc3` - Magnetic field components
  - `dens` - Density
- **Derived fields**: 
  - `vortx`, `vorty`, `vortz` - Vorticity components (∇ × v)
  - `currx`, `curry`, `currz` - Current density components (∇ × B)
  - `curvx`, `curvy`, `curvz` - Magnetic curvature components ((b·∇)b)
  - `grad_rho_x`, `grad_rho_y`, `grad_rho_z` - Density gradient components (∇ρ)

### Step 2: Compute Structure Functions

The analysis step computes structure function histograms from extracted slices.

#### Single slice analysis:
```bash
python run_analysis.py \
    --file_name path/to/slice.npz \
    --n_disp_total 2000 \
    --n_ell_bins 64 \
    --N_random_subsamples 2000 \
    --stride 1 \
    --n_processes 4
```

#### Batch analysis with configuration file:
```bash
# Create configuration file
python create_config.py

# Using configuration profiles
python run_analysis.py \
    --config sfunctor.yaml \
    --profile high_resolution \
    --slice_list my_slices.txt

# Override configuration parameters
python run_analysis.py \
    --config sfunctor.yaml \
    --profile default \
    --n_disp_total 5000 \
    --stride 2
```

#### Key parameters:
- `--n_disp_total`: Total number of displacement vectors to sample
- `--n_ell_bins`: Number of displacement magnitude bins
- `--N_random_subsamples`: Random positions sampled per displacement
- `--stride`: Downsampling factor (1=full resolution, 2=every other point)
- `--n_processes`: Number of parallel processes (default: CPU count - 2)
- `--stencil_width`: Finite difference stencil (2, 3, or 5 points)
- `--ell_min_factor`: Minimum displacement as fraction of stride
- `--ell_max_factor`: Maximum displacement as fraction of box size

#### MPI parallel analysis:
```bash
# Create a list of slice files
find slice_data -name "*.npz" > slice_list.txt

# Run with MPI
mpirun -n 64 python run_analysis.py \
    --slice_list slice_list.txt \
    --config sfunctor.yaml \
    --profile production

# Hybrid MPI + shared memory
mpirun -n 8 python run_analysis.py \
    --slice_list slice_list.txt \
    --n_processes 4  # 4 processes per MPI rank
```

### Step 3: Visualize Results

#### Basic visualization:
```bash
# Plot structure functions from a single result file
python visualize_sf_results.py results/sf_results_axis3_slice0.npz

# Compare multiple results
python visualize_sf_results.py results/sf_results_*.npz --compare

# Save plots to specific directory
python visualize_sf_results.py results/*.npz --output_dir plots/
```

#### Custom visualization script:
```python
import numpy as np
import matplotlib.pyplot as plt
from sfunctor.visualization import plot_structure_functions

# Load results
data = np.load('results/sf_results.npz')

# Access histograms
hist_mag = data['hist_mag']  # Shape: (10, n_ell, n_theta, n_phi, n_sf)
hist_other = data['hist_other']  # Shape: (18, n_ell, n_product)

# Channel names
mag_channels = data['mag_channels']
other_channels = data['other_channels']

# Plot specific channel
channel_idx = list(mag_channels).index('D_VEL')
sf_velocity = np.sum(hist_mag[channel_idx], axis=(1,2,3))

ell_centers = 0.5 * (data['ell_bin_edges'][:-1] + data['ell_bin_edges'][1:])
plt.loglog(ell_centers, sf_velocity, 'o-')
plt.xlabel('ℓ')
plt.ylabel('S₂(ℓ)')
plt.title('Velocity Structure Function')
plt.grid(True, alpha=0.3)
plt.savefig('velocity_sf.png')
```

## Physics and Channels

### Magnitude Channels (10 total)

1. **D_VEL** - Velocity increment magnitude |δv|
2. **D_MAG** - Magnetic field increment magnitude |δB|
3. **D_RHO** - Density increment magnitude |δρ|
4. **D_VEL_PERP** - Perpendicular velocity increment |δv_⊥|
5. **D_MAG_PERP** - Perpendicular magnetic field increment |δB_⊥|
6. **D_ZPLUS** - Elsasser variable z+ = v + vA increment
7. **D_ZMINUS** - Elsasser variable z- = v - vA increment
8. **D_VORT** - Vorticity increment |δω|
9. **D_CURR** - Current density increment |δj|
10. **D_CURV** - Magnetic curvature increment |δ((b·∇)b)|
11. **D_GRAD_RHO** - Density gradient increment |δ(∇ρ)|

### Cross Product Channels (18 total)

Cross products and magnitude products for computing angles between field increments:

- **D_VEL_CROSS_D_MAG** - |δv × δB| (numerator for angle)
- **D_VEL_D_MAG_MAG** - |δv| × |δB| (denominator for angle)
- **D_ZPLUS_CROSS_ZMINUS** - |δz+ × δz-|
- **D_ZPLUS_ZMINUS_MAG** - |δz+| × |δz-|
- **D_CURV_CROSS_GRAD_RHO** - |δ(curv) × δ(∇ρ)|
- **D_CURV_D_GRAD_RHO_MAG** - |δ(curv)| × |δ(∇ρ)|
- And more for all relevant field combinations...

### Angle Calculation

To compute the angle between two field increments:
```python
# Example: angle between velocity and magnetic field increments
cross_idx = other_channels.index('D_VEL_CROSS_D_MAG')
mag_idx = other_channels.index('D_VEL_D_MAG_MAG')

numerator = hist_other[cross_idx]    # |δv × δB|
denominator = hist_other[mag_idx]     # |δv| × |δB|

# sin(θ) = |δv × δB| / (|δv| × |δB|)
sin_theta = numerator / (denominator + 1e-10)  # Add small value to avoid division by zero
theta = np.arcsin(np.clip(sin_theta, -1, 1))
```

### Physical Interpretations

- **Elsasser Variables**: z± = v ± vA represent counter-propagating Alfvén wave packets
- **Magnetic Curvature**: (b·∇)b measures field line bending, important for reconnection
- **Density Gradient**: ∇ρ captures compressible effects and density structures
- **Perpendicular Components**: δ⊥ measures increments perpendicular to displacement vector, revealing anisotropy

## Configuration

### Configuration file structure (sfunctor.yaml):
```yaml
# Default parameters
defaults:
  n_disp_total: 2000
  n_ell_bins: 50
  ell_min_factor: 0.5      # Min displacement = ell_min_factor * stride
  ell_max_factor: 0.25     # Max displacement = ell_max_factor * box_size
  N_random_subsamples: 1000
  stride: 1
  stencil_width: 2         # 2, 3, or 5 point stencil
  n_theta_bins: 18         # Polar angle bins
  n_phi_bins: 18           # Azimuthal angle bins
  n_sf_bins: 127           # Structure function value bins
  n_product_bins: 127      # Cross product value bins
  sf_min: 1e-8
  sf_max: 100.0
  n_processes: null        # Auto-detect

# Profiles for different use cases
profiles:
  quick:
    n_disp_total: 500
    n_ell_bins: 32
    N_random_subsamples: 500
    stride: 4
    
  production:
    n_disp_total: 10000
    n_ell_bins: 100
    N_random_subsamples: 5000
    stride: 1
    
  high_resolution:
    n_disp_total: 20000
    n_ell_bins: 128
    N_random_subsamples: 10000
    stride: 1
    n_theta_bins: 36
    n_phi_bins: 36
    
  low_memory:
    n_disp_total: 1000
    N_random_subsamples: 500
    stride: 4
    n_processes: 2
```

### Using profiles:
```bash
# Quick test run
python run_analysis.py --config sfunctor.yaml --profile quick --file_name slice.npz

# Production run
python run_analysis.py --config sfunctor.yaml --profile production --slice_list all_slices.txt

# Custom parameters with profile
python run_analysis.py \
    --config sfunctor.yaml \
    --profile quick \
    --n_disp_total 1000 \  # Override profile setting
    --stride 2
```

## Performance Optimization

### Memory Considerations

Memory usage scales with:
- Number of processes: `n_processes × slice_size × n_fields`
- Slice size: Quadratic in grid resolution
- Number of bins: Linear in `n_ell_bins × n_theta_bins × n_phi_bins × n_sf_bins`

For large slices (>1024²):
- Use `stride > 1` to downsample
- Reduce `n_processes`
- Use the `low_memory` profile

### Speed Optimization

1. **First run is slower**: Numba JIT compilation occurs on first execution
2. **Optimal process count**: Usually `n_processes = n_cores - 2`
3. **Stride parameter**: Reduces data size quadratically (stride=2 → 4× speedup)
4. **Batch processing**: Process multiple slices together for better efficiency
5. **Stencil width**: Use 2-point stencil unless higher accuracy needed

### Scaling Examples

```bash
# Single node (shared memory only)
python run_analysis.py --slice_list slices.txt --n_processes 8

# Multi-node with MPI (distributed memory)
mpirun -n 64 python run_analysis.py --slice_list slices.txt

# Hybrid MPI + shared memory (recommended for clusters)
mpirun -n 8 python run_analysis.py --slice_list slices.txt --n_processes 4
# This runs 8 MPI ranks with 4 processes each = 32 total processes
```

### Performance Benchmarks

Typical performance on a modern HPC node (dual-socket, 32 cores):
- 512² slice: ~10 seconds
- 1024² slice: ~40 seconds  
- 2048² slice: ~3 minutes
- 4096² slice: ~15 minutes

(Times for n_disp_total=2000, N_random_subsamples=1000, stride=1)

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'yaml'**
   ```bash
   pip install pyyaml
   ```

2. **ImportError: scipy 0.16+ is required**
   ```bash
   pip install scipy
   ```

3. **Memory errors with large slices**
   ```bash
   # Increase stride
   python run_analysis.py --file_name slice.npz --stride 4
   
   # Reduce processes
   python run_analysis.py --file_name slice.npz --n_processes 2
   
   # Use low memory profile
   python run_analysis.py --config sfunctor.yaml --profile low_memory
   ```

4. **MPI errors**
   ```bash
   # Check MPI installation
   which mpirun
   mpirun --version
   
   # Test mpi4py
   python -c "from mpi4py import MPI; print(f'MPI rank {MPI.COMM_WORLD.rank}/{MPI.COMM_WORLD.size}')"
   ```

5. **Slow first run**
   - This is normal due to Numba JIT compilation
   - Subsequent runs will be much faster
   - Consider running a small test first to trigger compilation

6. **No data in histograms**
   - Check that slice contains valid data
   - Verify displacement parameters are reasonable
   - Ensure stride doesn't skip all data points

### Debug Mode

```bash
# Enable verbose output
python run_analysis.py --file_name slice.npz --verbose

# Check slice contents
python -c "
import numpy as np
data = np.load('slice.npz')
print('Fields:', list(data.keys()))
print('Shape:', data['velx'].shape)
print('Density range:', data['dens'].min(), '-', data['dens'].max())
"

# Test with minimal parameters
python run_analysis.py \
    --file_name slice.npz \
    --n_disp_total 10 \
    --N_random_subsamples 10 \
    --n_ell_bins 8 \
    --stride 8
```

## API Reference

### Core Modules

#### sfunctor.io.extract
```python
from sfunctor.io.extract import extract_2d_slice

# Extract a single slice
slice_data = extract_2d_slice(
    sim_name="MySim",
    axis=3,              # xy-plane
    slice_value=0.0,     # Center of domain
    file_number=100,
    data_dir="/path/to/data",
    save=True,
    cache_dir="/path/to/output"
)
```

#### sfunctor.analysis.single_slice
```python
from sfunctor.analysis.single_slice import analyze_slice
from sfunctor.io.slice_io import load_slice_npz

# Load slice
slice_data = load_slice_npz("slice.npz", stride=2)

# Analyze
result = analyze_slice(
    slice_data,
    n_displacements=1000,
    n_ell_bins=50,
    n_random_subsamples=1000,
    stencil_width=2,
    n_processes=4,
    axis=3
)

# Access results
hist_mag = result['hist_mag']      # Magnitude channels
hist_other = result['hist_other']   # Cross products
```

#### sfunctor.visualization
```python
from sfunctor.visualization import (
    plot_magnitude_channels,
    plot_cross_products,
    plot_angle_distributions,
    compare_structure_functions
)

# Plot all magnitude channels
fig = plot_magnitude_channels(result)

# Plot specific channels
fig = plot_cross_products(
    result, 
    channels=['D_VEL_CROSS_D_MAG', 'D_CURV_CROSS_GRAD_RHO']
)

# Compare multiple results
fig = compare_structure_functions(
    [result1, result2, result3],
    labels=['Run 1', 'Run 2', 'Run 3'],
    channel='D_VEL'
)
```

### Utility Functions

#### Configuration Management
```python
from sfunctor.utils.config import load_config, merge_configs

# Load configuration with profile
config = load_config('sfunctor.yaml', profile='production')

# Override specific parameters
config = merge_configs(config, {'stride': 2, 'n_processes': 8})

# Create default config
from sfunctor.utils.config import create_default_config
create_default_config('my_config.yaml')
```

#### Displacement Generation
```python
from sfunctor.utils.displacements import (
    find_ell_bin_edges,
    build_displacement_list
)

# Generate logarithmic bins
ell_edges = find_ell_bin_edges(
    ell_min=1.0,
    ell_max=100.0,
    n_ell_bins=50
)

# Build displacement vectors
displacements = build_displacement_list(
    ell_bin_edges=ell_edges,
    n_disp_total=1000
)
```

## Examples

### Example 1: Basic Single Slice Analysis
```bash
# Extract slice
python run_extract_slice.py \
    --sim_name Turb_320_beta100_dedt025_plm \
    --axis 3 \
    --offset 0.0 \
    --file_numbers 100

# Analyze
python run_analysis.py \
    --file_name slice_data/Turb_320_beta100_dedt025_plm_axis3_slice0_file0100.npz \
    --n_disp_total 2000 \
    --N_random_subsamples 1000

# Visualize
python visualize_sf_results.py results/sf_results_*.npz
```

### Example 2: Multi-Slice Production Run
```bash
# Extract slices at multiple positions
for offset in -0.25 0.0 0.25; do
    python run_extract_slice.py \
        --sim_name Turb_1024_beta10_dedt025_plm \
        --axis 1,2,3 \
        --offset $offset \
        --file_numbers 50-150
done

# Create slice list
find slice_data -name "*.npz" | sort > all_slices.txt

# Run analysis with MPI
mpirun -n 32 python run_analysis.py \
    --slice_list all_slices.txt \
    --config sfunctor.yaml \
    --profile production

# Combine and visualize results
python combine_results.py results/hist_ALL_*.npz --output combined_results.npz
python visualize_sf_results.py combined_results.npz
```

### Example 3: Custom Analysis Script
```python
#!/usr/bin/env python3
"""Custom analysis of structure functions."""

import numpy as np
import matplotlib.pyplot as plt
from sfunctor.io.slice_io import load_slice_npz
from sfunctor.analysis.single_slice import analyze_slice

# Load and analyze multiple slices
results = []
for i in range(3):
    # Load slice
    slice_data = load_slice_npz(f'slice_{i}.npz', stride=2)
    
    # Run analysis
    result = analyze_slice(
        slice_data,
        n_displacements=5000,
        n_ell_bins=64,
        n_random_subsamples=2000,
        axis=3
    )
    results.append(result)

# Compare velocity structure functions
fig, ax = plt.subplots(figsize=(8, 6))

for i, result in enumerate(results):
    # Extract velocity SF
    vel_idx = list(result['mag_channels']).index('D_VEL')
    sf_vel = np.sum(result['hist_mag'][vel_idx], axis=(1,2,3))
    
    # Get ell values
    ell_edges = result['ell_bin_edges']
    ell_centers = 0.5 * (ell_edges[:-1] + ell_edges[1:])
    
    # Plot
    ax.loglog(ell_centers, sf_vel, 'o-', label=f'Slice {i}')

# Add reference scaling
ell_ref = np.logspace(0, 2, 50)
ax.loglog(ell_ref, 10 * ell_ref**(2/3), 'k--', alpha=0.5, label='ℓ^(2/3)')

ax.set_xlabel('ℓ')
ax.set_ylabel('S₂(ℓ)')
ax.set_title('Velocity Structure Functions')
ax.legend()
ax.grid(True, alpha=0.3)

plt.savefig('velocity_sf_comparison.png', dpi=150)
```

## Best Practices

1. **Start Small**: Test with reduced parameters before production runs
2. **Check Convergence**: Increase N_random_subsamples until results stabilize
3. **Save Metadata**: Use descriptive filenames and keep configuration files
4. **Monitor Memory**: Use `htop` or similar to watch memory usage
5. **Batch Processing**: Group similar analyses for efficiency
6. **Version Control**: Track configuration files and analysis scripts

## Contributing

We welcome contributions! Areas for improvement:
- Additional physics channels
- Performance optimizations
- Visualization enhancements
- Documentation improvements

Please see CONTRIBUTING.md for guidelines.

## Citation

If you use SFunctor in your research, please cite:
```bibtex
@software{sfunctor,
  title = {SFunctor: Structure Function Analysis for MHD Turbulence},
  author = {Your Author List},
  year = {2024},
  url = {https://github.com/yourusername/sfunctor}
}
```

## License

This project is licensed under the MIT License - see LICENSE file for details.