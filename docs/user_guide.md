# User Guide

## Understanding Structure Functions

Structure functions are statistical measures of turbulence that quantify how field quantities vary across different spatial scales. For a field `f`, the p-th order structure function is:

```
SF_p(ℓ) = ⟨|f(x + ℓ) - f(x)|^p⟩
```

SFunctor computes these for multiple MHD fields simultaneously, with angle-resolved bins to capture anisotropy.

## Input Data Format

SFunctor expects 2D slices from 3D MHD simulations in NumPy `.npz` format with these fields:

### Required Fields
- `dens`: Density (ρ)
- `velx`, `vely`, `velz`: Velocity components
- `bcc1`, `bcc2`, `bcc3`: Magnetic field components

### Optional Fields (computed if missing)
- `vortx`, `vorty`, `vortz`: Vorticity
- `currx`, `curry`, `currz`: Current density
- `curvx`, `curvy`, `curvz`: Magnetic curvature
- `grad_rho_x`, `grad_rho_y`, `grad_rho_z`: Density gradient

## Structure Function Channels

SFunctor computes 24 different structure function channels:

### Magnitude Channels (11)
1. **D_V**: Velocity increments |δv|
2. **D_B**: Magnetic field increments |δB|
3. **D_RHO**: Density increments |δρ|
4. **D_VA**: Alfvén velocity increments |δv_A|
5. **D_ZPLUS**: Elsässer variable z⁺ = v + v_A
6. **D_ZMINUS**: Elsässer variable z⁻ = v - v_A
7. **D_OMEGA**: Vorticity increments |δω|
8. **D_J**: Current density increments |δj|
9. **D_CURV**: Magnetic curvature increments |δ(b·∇b)|
10. **D_GRAD_RHO**: Density gradient increments |δ(∇ρ)|
11. **D_B_over_Bmean**: Normalized magnetic increments

### Cross Products & Correlations (13)
12-24. Various cross products and correlations between field components

## Analysis Parameters

### Key Parameters

```python
analyze_slice(
    slice_file,
    # Spatial sampling
    stride=1,              # Downsample factor (1 = full resolution)
    
    # Displacement configuration
    n_disp_per_decade=8,   # Displacement samples per decade in ℓ
    n_disp_per_bin=10,     # Displacements per ℓ bin
    n_disp_total=1000,     # Total number of displacements
    ell_min=1,             # Minimum displacement
    ell_max=100,           # Maximum displacement
    
    # Random sampling
    n_random_subsamples=10000,  # Random points per displacement
    
    # Angular binning
    n_theta_bins=16,       # Bins in polar angle θ
    n_phi_bins=32,         # Bins in azimuthal angle φ
    
    # Computational
    stencil_width=2,       # Finite difference stencil (2, 3, or 5)
    n_processes=1,         # CPU processes (ignored with GPU)
)
```

### Parameter Selection Guidelines

#### For Quick Testing
```python
config_test = {
    'stride': 8,
    'n_disp_total': 100,
    'n_random_subsamples': 1000,
    'n_theta_bins': 8,
    'n_phi_bins': 16,
}
```

#### For Production Runs
```python
config_production = {
    'stride': 1,
    'n_disp_total': 10000,
    'n_random_subsamples': 50000,
    'n_theta_bins': 32,
    'n_phi_bins': 64,
}
```

#### Memory-Constrained Systems
```python
config_lowmem = {
    'stride': 4,
    'n_disp_total': 1000,
    'n_random_subsamples': 5000,
    'n_theta_bins': 16,
    'n_phi_bins': 32,
}
```

## Output Format

Results are returned as a dictionary:

```python
results = {
    'hist_mag': ndarray,      # Shape: (11, n_ell, n_theta, n_phi, n_sf)
    'hist_other': ndarray,     # Shape: (13, n_ell, n_product)
    'ell_bin_edges': ndarray,  # Displacement bin edges
    'theta_bin_edges': ndarray,# Polar angle bins
    'phi_bin_edges': ndarray,  # Azimuthal angle bins
    'sf_bin_edges': ndarray,   # Structure function value bins
    'displacements': ndarray,  # Actual (dx, dy) pairs used
}
```

## Visualization

### Basic Plotting

```python
import matplotlib.pyplot as plt
import numpy as np

# Load results
results = np.load('results.npz')

# Extract velocity structure function
velocity_sf = results['hist_mag'][0]  # Channel 0 is velocity

# Average over angles for isotropic SF
sf_isotropic = velocity_sf.mean(axis=(2, 3))  # Average over θ and φ

# Get bin centers
ell_centers = 0.5 * (results['ell_bin_edges'][:-1] + results['ell_bin_edges'][1:])
sf_centers = 0.5 * (results['sf_bin_edges'][:-1] + results['sf_bin_edges'][1:])

# Plot 2D histogram
plt.figure(figsize=(10, 6))
plt.pcolormesh(ell_centers, sf_centers, sf_isotropic.T, shading='auto')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Displacement ℓ')
plt.ylabel('|δv|')
plt.title('Velocity Structure Function')
plt.colorbar(label='Counts')
plt.show()
```

### Anisotropy Analysis

```python
# Compute anisotropy: parallel vs perpendicular to B
theta_centers = 0.5 * (results['theta_bin_edges'][:-1] + results['theta_bin_edges'][1:])

# Parallel: θ ≈ 0 or π
parallel_mask = (theta_centers < np.pi/6) | (theta_centers > 5*np.pi/6)
# Perpendicular: θ ≈ π/2
perp_mask = (theta_centers > np.pi/3) & (theta_centers < 2*np.pi/3)

sf_parallel = velocity_sf[:, parallel_mask, :, :].mean(axis=(1, 2))
sf_perp = velocity_sf[:, perp_mask, :, :].mean(axis=(1, 2))

# Plot anisotropy ratio
plt.figure(figsize=(8, 6))
for ell_idx in range(0, len(ell_centers), 10):
    ratio = sf_parallel[ell_idx] / sf_perp[ell_idx]
    plt.plot(sf_centers, ratio, label=f'ℓ = {ell_centers[ell_idx]:.1f}')
plt.xscale('log')
plt.xlabel('|δv|')
plt.ylabel('SF_∥ / SF_⊥')
plt.title('Anisotropy Ratio')
plt.legend()
plt.show()
```

## Best Practices

1. **Start with small tests**: Use high stride and low sample counts to verify setup
2. **Monitor memory usage**: GPU memory is limited (typically 8-64 GB)
3. **Use appropriate stencil width**:
   - 2-point: Standard increments, fastest
   - 3-point: Second-order differences
   - 5-point: Fourth-order accuracy, slowest
4. **Save intermediate results**: For large runs, save checkpoints
5. **Validate with known cases**: Compare isotropic limits with theory

## Troubleshooting

### GPU Not Detected
```bash
# Check CUDA/ROCm installation
nvidia-smi  # For NVIDIA
rocm-smi    # For AMD

# Force CPU mode if needed
export SFUNCTOR_USE_CPU=1
```

### Out of Memory
- Increase stride (downsample)
- Reduce n_random_subsamples
- Process in smaller batches
- Use memory profiling:
```python
import cupy as cp
mempool = cp.get_default_memory_pool()
print(f"GPU memory used: {mempool.used_bytes() / 1e9:.2f} GB")
```

### Numerical Differences GPU vs CPU
Small differences (<1%) are normal due to:
- Different random number generators
- Floating-point rounding
- Parallel reduction order

For exact reproducibility, use CPU mode with fixed seed:
```python
np.random.seed(42)
```