# Scientific Analysis Features

SFunctor provides advanced scientific analysis capabilities for studying MHD turbulence beyond basic structure functions. These features enable comprehensive turbulence characterization including temporal evolution, anisotropy quantification, field correlations, and multi-scale decomposition.

## Time-Series Analysis

Analyze the temporal evolution of turbulence properties across multiple snapshots.

### Features
- Track structure function evolution over time
- Compute time-dependent anisotropy measures
- Analyze spectral evolution and scaling exponents
- Detect transitions and intermittency changes

### Usage

```python
from sfunctor.analysis.time_series import analyze_time_series
from pathlib import Path

# Get snapshot files in temporal order
slice_files = sorted(Path("data").glob("*.npz"))

# Analyze time series
results = analyze_time_series(
    slice_files,
    config={
        'stride': 2,  # Downsample by 2x
        'n_disp_total': 1000,
        'n_random_subsamples': 10000
    },
    output_dir="time_series_output"
)

# Access results
times = results['times']
hist_evolution = results['hist_mag_series']  # Shape: (n_times, n_channels, ...)
anisotropy = results['anisotropy_series']
energy = results['energy_series']
scaling = results['scaling_exponents']
```

### Individual Analysis Functions

```python
from sfunctor.analysis.time_series import (
    compute_anisotropy_measures,
    compute_energy_content,
    compute_scaling_exponents,
    detect_transition_times
)

# Compute anisotropy for single snapshot
aniso = compute_anisotropy_measures(single_result)
# Returns: parallel/perp ratio, alignment angle, anisotropy index

# Compute energy in each channel
energy = compute_energy_content(single_result)
# Returns: energy content for velocity, magnetic, density, etc.

# Compute scaling exponents ζ_p
scaling = compute_scaling_exponents(single_result, p_values=[1, 2, 3])
# Returns: scaling exponents for different moments

# Detect transition times in time series
transitions = detect_transition_times(time_series_results, threshold=0.1)
```

## Scale-Dependent Anisotropy

Quantify anisotropic properties as a function of scale with multiple metrics.

### Features
- Four anisotropy measures: ratio, variance, entropy, alignment
- Scale-by-scale analysis
- Preferential direction identification
- Anisotropy spectrograms
- SVD-based mode decomposition

### Usage

```python
from sfunctor.analysis.anisotropy import (
    compute_scale_dependent_anisotropy,
    create_anisotropy_spectrogram,
    decompose_anisotropy_modes
)

# Compute anisotropy with different methods
aniso_ratio = compute_scale_dependent_anisotropy(results, method='ratio')
aniso_var = compute_scale_dependent_anisotropy(results, method='variance')
aniso_entropy = compute_scale_dependent_anisotropy(results, method='entropy')

# With reference direction (e.g., mean B field)
ref_direction = np.array([0, 0, 1])  # Along z
aniso_align = compute_scale_dependent_anisotropy(
    results, 
    method='alignment',
    reference_direction=ref_direction
)

# Create anisotropy spectrogram
spectrogram = create_anisotropy_spectrogram(results, channel=0)  # Velocity

# Decompose into principal modes
modes = decompose_anisotropy_modes(results, n_modes=3)
print(f"Variance explained: {modes['variance_explained']}")
```

### Anisotropy Metrics

1. **Parallel/Perpendicular Ratio**: SF∥/SF⊥ at each scale
2. **Angular Variance**: Spread of angular distribution
3. **Angular Entropy**: Ordering measure (0=ordered, 1=random)
4. **Alignment**: Correlation with reference direction
5. **Strength**: Max/min directional ratio
6. **Preferential Directions**: Dominant θ, φ at each scale

## Cross-Correlation Analysis

Analyze relationships and energy transfer between different MHD fields.

### Features
- Field-to-field correlations
- Scale-dependent energy transfer
- Nonlinear triadic interactions
- Conditional statistics
- Mutual information

### Usage

```python
from sfunctor.analysis.cross_correlation import (
    compute_field_correlations,
    compute_scale_dependent_transfer,
    compute_nonlinear_coupling,
    compute_conditional_statistics,
    compute_mutual_information
)

# Field correlations (velocity-magnetic, velocity-density, etc.)
correlations = compute_field_correlations(
    results,
    field_pairs=[(0, 1), (0, 2), (1, 2)],
    normalize=True,
    compute_phase=True
)

# Energy transfer functions
transfer = compute_scale_dependent_transfer(
    results,
    source_channel=0,  # Velocity
    target_channel=1,  # Magnetic
    method='spectral'  # or 'flux', 'cascade'
)

# Nonlinear coupling (three-wave interactions)
coupling = compute_nonlinear_coupling(
    results,
    triplet=(0, 1, 7),  # Velocity, Magnetic, Current
    compute_bispectrum=True
)

# Conditional statistics (e.g., velocity conditioned on density)
conditional = compute_conditional_statistics(
    results,
    condition_channel=2,  # Density
    target_channel=0,     # Velocity
    n_bins=5,
    percentile_bins=True
)

# Mutual information (nonlinear dependencies)
mi = compute_mutual_information(
    results,
    channel_pairs=[(0, 1), (0, 2)],
    n_bins=20
)
```

### Transfer Methods
- **Spectral**: Transfer function and coherence
- **Flux**: Forward/backward energy flux
- **Cascade**: Energy cascade rates

## Wavelet Decomposition

Multi-scale analysis using wavelet transforms for identifying coherent structures and intermittency.

### Features
- Discrete and continuous wavelet transforms
- Multi-scale decomposition
- Wavelet coherence between fields
- Multifractal analysis
- Coherent structure identification
- Intermittency detection
- **NEW: Spatial wavelet power spectrum mapping**

### Installation
Wavelet features require PyWavelets:
```bash
pip install PyWavelets
```

### Usage

```python
from sfunctor.analysis.wavelet import (
    wavelet_decompose_sf,
    continuous_wavelet_analysis,
    wavelet_coherence_analysis,
    multifractal_wavelet_analysis,
    identify_coherent_structures
)

# Discrete wavelet decomposition
decomp = wavelet_decompose_sf(
    results,
    channel=0,  # Velocity
    wavelet='db4',  # Daubechies-4
    level=None  # Auto-detect max level
)

# Access decomposition
approx = decomp['approx']  # Low-frequency approximation
details_h = decomp['details_horizontal']  # Horizontal details
details_v = decomp['details_vertical']    # Vertical details
details_d = decomp['details_diagonal']    # Diagonal details
energy = decomp['energy_per_scale']       # Energy distribution

# Continuous wavelet transform
cwt = continuous_wavelet_analysis(
    results,
    channel=0,
    scales=np.logspace(0, 2, 32),  # 32 scales from 1 to 100
    wavelet='morl'  # Morlet wavelet
)

scalogram = cwt['scalogram']  # Power at each scale/position
dominant = cwt['dominant_scales']  # Identified dominant scales
intermittency = cwt['intermittency_factor']  # Scale-dependent intermittency

# Wavelet coherence between fields
coherence = wavelet_coherence_analysis(
    results,
    channel1=0,  # Velocity
    channel2=1,  # Magnetic
    wavelet='morl'
)

# Multifractal analysis
multifractal = multifractal_wavelet_analysis(
    results,
    channel=0,
    q_values=np.linspace(-5, 5, 21),  # Moment orders
    n_scales=16
)

alpha = multifractal['alpha']  # Singularity spectrum
f_alpha = multifractal['f_alpha']  # Multifractal spectrum
width = multifractal['width']  # Multifractal width

# Identify coherent structures
structures = identify_coherent_structures(
    decomp,
    threshold_percentile=95.0,  # Top 5% intensities
    min_size=3  # Minimum structure size
)
```

### Wavelet Types
- **Discrete**: 'db4' (Daubechies), 'sym5' (Symlets), 'coif3' (Coiflets)
- **Continuous**: 'morl' (Morlet), 'mexh' (Mexican hat), 'gaus8' (Gaussian)

## Wavelet Power Spectrum Mapping

Create spatial maps of turbulence properties by computing local wavelet power spectra and fitting power laws. This reveals how turbulence characteristics vary across the spatial domain.

### Features
- Local wavelet power spectrum at each spatial location
- Power law fitting to extract amplitude and spectral index
- Multi-scale spatial analysis
- Anisotropic turbulence mapping
- Cross-scale correlation maps
- Automatic region identification

### Usage

```python
from sfunctor.analysis.wavelet_maps import (
    WaveletMapConfig,
    compute_wavelet_power_maps,
    compute_multiscale_maps,
    compute_anisotropic_wavelet_maps,
    identify_turbulence_regions,
    visualize_wavelet_maps
)

# Configure analysis
config = WaveletMapConfig(
    outer_scale=64.0,      # Maximum scale to analyze (pixels)
    inner_scale=2.0,       # Minimum scale
    n_scales=32,           # Number of scales
    window_size=64,        # Local window size
    window_overlap=0.5,    # Overlap between windows
    wavelet='morl',        # Morlet wavelet
    detrend='linear',      # Remove trends
    fit_method='robust'    # Robust power law fitting
)

# Compute wavelet power maps
results = compute_wavelet_power_maps(field, config)

# Access results
amplitude_map = results['amplitude_map']  # Power law amplitude
slope_map = results['slope_map']          # Spectral index
r_squared_map = results['r_squared_map']  # Fit quality
energy_map = results['energy_map']        # Total energy
peak_scale_map = results['peak_scale_map'] # Dominant scale
intermittency_map = results['intermittency_map']  # Local intermittency

# Visualize
visualize_wavelet_maps(results, field)
```

### Multi-Scale Analysis

Analyze different scale ranges to study scale-dependent spatial variations:

```python
# Define scale ranges
scale_ranges = [
    (2, 8),     # Small scales (dissipation)
    (8, 32),    # Intermediate scales (inertial)
    (32, 128)   # Large scales (injection)
]

# Compute maps for each range
multiscale = compute_multiscale_maps(field, scale_ranges, config)

# Compare spectral slopes across scales
for key, results in multiscale.items():
    slope = results['slope_map']
    print(f"{key}: mean slope = {np.nanmean(slope):.2f}")
```

### Anisotropic Analysis

Study directional variations in turbulence properties:

```python
# Compute directional wavelet maps
aniso_results = compute_anisotropic_wavelet_maps(
    field,
    config,
    angles=[0, 45, 90, 135]  # degrees
)

# Access anisotropy measures
anisotropy_ratio = aniso_results['anisotropy_ratio']
preferred_direction = aniso_results['preferred_direction']
alignment_strength = aniso_results['alignment_strength']
```

### Region Identification

Automatically identify regions with different turbulence characteristics:

```python
# Identify turbulence regions
regions = identify_turbulence_regions(
    results,
    slope_threshold=(-2.0, -1.3),  # Spectral slope range
    r_squared_threshold=0.8         # Minimum fit quality
)

# Access region masks
kolmogorov_regions = regions['regions']['kolmogorov']
dissipation_regions = regions['regions']['dissipation']
injection_regions = regions['regions']['injection']

# Get statistics
for name, stats in regions['statistics'].items():
    print(f"{name}: {stats['fraction']*100:.1f}% of area")
    print(f"  Mean slope: {stats['mean_slope']:.2f}")
```

### Cross-Scale Correlations

Analyze energy transfer between scales:

```python
# Compute correlation between two scales
cross_scale = compute_cross_scale_correlation_map(
    field,
    scale1=4.0,   # Small scale
    scale2=32.0,  # Large scale
    config=config
)

correlation_map = cross_scale['correlation_map']
coherence_map = cross_scale['coherence_map']
phase_map = cross_scale['phase_map']
```

### Interpretation

**Spectral Slope Values:**
- `-5/3 ≈ -1.67`: Kolmogorov turbulence (inertial range)
- `-3`: Steep spectrum (dissipation or magnetic dominated)
- `-1`: Shallow spectrum (energy injection or inverse cascade)

**Amplitude Map:**
- High values: Regions of intense turbulence
- Low values: Quiescent or laminar regions

**R² Map:**
- Values > 0.8: Good power law fit (well-developed turbulence)
- Values < 0.5: Poor fit (transitional or non-turbulent)

**Intermittency Map:**
- High values: Intermittent, bursty turbulence
- Low values: Steady, homogeneous turbulence

## Complete Example

```python
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from sfunctor.io.slice_io import load_slice_npz
from sfunctor.analysis.single_slice import analyze_slice
from sfunctor.analysis.anisotropy import compute_scale_dependent_anisotropy
from sfunctor.analysis.cross_correlation import compute_field_correlations

# Load and analyze single slice
slice_data = load_slice_npz("data/slice_0000.npz", stride=2)
results = analyze_slice(
    slice_data,
    n_displacements=1000,
    n_random_subsamples=10000
)

# Compute anisotropy
aniso = compute_scale_dependent_anisotropy(results, method='ratio')

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot anisotropy ratio vs scale
ax = axes[0]
ell = aniso['ell_centers']
for ch in range(3):  # Velocity, Magnetic, Density
    ratio = aniso['parallel_perp_ratio'][ch]
    ax.semilogx(ell, ratio, label=f'Channel {ch}')
ax.axhline(1.0, color='k', linestyle=':', alpha=0.5)
ax.set_xlabel('Scale ℓ')
ax.set_ylabel('SF∥ / SF⊥')
ax.set_title('Scale-Dependent Anisotropy')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot field correlations
correlations = compute_field_correlations(results)
ax = axes[1]
ax.semilogx(ell, correlations['velocity_magnetic_scale'], label='Vel-Mag')
ax.semilogx(ell, correlations['velocity_density_scale'], label='Vel-Dens')
ax.set_xlabel('Scale ℓ')
ax.set_ylabel('Correlation')
ax.set_title('Field Correlations')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Performance Considerations

### Memory Usage
- Time-series analysis stores results for all snapshots
- Use `save_intermediate=True` for large datasets
- Wavelet transforms can be memory-intensive for large arrays

### Computation Time
- Anisotropy and correlation analyses are fast (seconds)
- Wavelet decomposition scales with array size
- Multifractal analysis is computationally intensive

### Parallelization
- Time-series analysis can process snapshots in parallel
- Most functions are NumPy-vectorized for efficiency
- GPU acceleration available for core histogram computation

## Scientific Applications

### Turbulence Characterization
- Quantify anisotropy induced by magnetic fields
- Measure energy cascade rates
- Identify dominant scales and structures

### Intermittency Analysis
- Detect intermittent events via wavelets
- Compute multifractal spectra
- Analyze flatness and higher moments

### Field Interactions
- Study velocity-magnetic coupling
- Quantify cross-helicity dynamics
- Analyze density-velocity correlations

### Temporal Evolution
- Track decay/growth of turbulence
- Identify phase transitions
- Study dynamo action

## References

The scientific methods implemented are based on:

1. **Anisotropy Analysis**: Cho & Vishniac (2000), Beresnyak & Lazarian (2009)
2. **Structure Functions**: Politano & Pouquet (1998), Biskamp (2003)
3. **Wavelet Methods**: Farge (1992), Torrence & Compo (1998)
4. **Multifractal Analysis**: Frisch (1995), Muzy et al. (1991)
5. **MHD Turbulence**: Goldreich & Sridhar (1995), Boldyrev (2006)