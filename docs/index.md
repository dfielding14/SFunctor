# SFunctor Documentation

## GPU-Accelerated Structure Function Analysis for MHD Turbulence

SFunctor is a high-performance Python package for computing anisotropic, angle-resolved structure functions from magnetohydrodynamic (MHD) turbulence simulations. With GPU acceleration, it achieves 100-1000x speedups over CPU implementations.

## Features

- 🚀 **GPU Acceleration**: Automatic GPU detection and usage (NVIDIA/AMD/CPU fallback)
- 📊 **24 Structure Function Channels**: Comprehensive turbulence analysis
- 🔬 **Multiple Stencil Widths**: 2, 3, and 5-point stencils for different analyses
- 🖥️ **HPC Ready**: Optimized for supercomputers like Frontier
- 📈 **Scalable**: Handles simulations from 64³ to 10240³ and beyond

## Quick Start

### Installation

```bash
# Basic installation
pip install sfunctor

# With GPU support (NVIDIA)
pip install sfunctor[gpu]

# With MPI support
pip install sfunctor[mpi]

# Everything
pip install sfunctor[all]
```

### Basic Usage

```python
import sfunctor

# Analyze a single slice
results = sfunctor.analyze_slice(
    "slice_data.npz",
    stride=2,           # Downsample by 2x
    n_disp_total=1000,  # Number of displacements
    n_random_subsamples=10000  # Samples per displacement
)

# Access structure functions
velocity_sf = results['hist_mag'][0]  # Velocity structure function
magnetic_sf = results['hist_mag'][1]  # Magnetic structure function
```

### GPU Acceleration

GPU acceleration is automatic when available:

```python
# Check GPU status
from sfunctor.core.histograms_gpu import get_gpu_info
info = get_gpu_info()
print(f"GPU Available: {info['available']}")
print(f"GPU Type: {info['backend']}")  # cuda, rocm, or cpu
```

To force CPU usage:
```bash
export SFUNCTOR_USE_CPU=1
python your_script.py
```

## Performance

| Platform | Configuration | Speedup vs CPU |
|----------|--------------|----------------|
| MacBook M4 Max | Apple Silicon GPU | 2-3x |
| Workstation | NVIDIA RTX 4090 | 10-30x |
| Frontier Node | 8 AMD MI250X GCDs | 100-300x |

## Advanced Scientific Features

SFunctor now includes comprehensive turbulence analysis capabilities:

- **[Time-Series Analysis](scientific_features.md#time-series-analysis)**: Track turbulence evolution across snapshots
- **[Scale-Dependent Anisotropy](scientific_features.md#scale-dependent-anisotropy)**: Multiple anisotropy metrics and spectrograms
- **[Cross-Correlation Analysis](scientific_features.md#cross-correlation-analysis)**: Field correlations and energy transfer
- **[Wavelet Decomposition](scientific_features.md#wavelet-decomposition)**: Multi-scale analysis and coherent structures

## Documentation

- [Installation Guide](installation.md)
- [User Guide](user_guide.md)
- [Scientific Features](scientific_features.md) ⭐ NEW
- [API Reference](api_reference.md)
- [Examples](examples.md)
- [HPC Usage](hpc_guide.md)
- [Contributing](contributing.md)