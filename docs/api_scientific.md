# Scientific Features API Reference

## Time-Series Analysis Module

### `sfunctor.analysis.time_series`

#### `analyze_time_series`
```python
analyze_time_series(
    slice_files: List[Union[str, Path]],
    time_values: Optional[np.ndarray] = None,
    config: Optional[Dict] = None,
    output_dir: Optional[Path] = None,
    save_intermediate: bool = True,
    verbose: bool = True,
) -> Dict[str, np.ndarray]
```

Analyze structure functions across multiple time snapshots.

**Parameters:**
- `slice_files`: List of paths to slice files in temporal order
- `time_values`: Optional array of time values for each snapshot
- `config`: Analysis configuration dictionary
- `output_dir`: Directory to save intermediate results
- `save_intermediate`: Whether to save results for each snapshot
- `verbose`: Print progress information

**Returns:**
Dictionary containing:
- `times`: Time values for each snapshot
- `hist_mag_series`: Structure functions vs time
- `hist_other_series`: Cross products vs time
- `anisotropy_series`: Anisotropy measures vs time
- `energy_series`: Energy in different channels vs time
- `scaling_exponents`: Scaling exponents vs time

---

#### `compute_anisotropy_measures`
```python
compute_anisotropy_measures(results: Dict) -> Dict[str, np.ndarray]
```

Compute various anisotropy measures from structure function results.

**Returns:**
- `parallel_perp_ratio`: Ratio of parallel to perpendicular SF
- `alignment_angle`: Mean alignment angle with B field
- `anisotropy_index`: Scalar anisotropy measure

---

#### `compute_energy_content`
```python
compute_energy_content(results: Dict) -> Dict[str, float]
```

Compute energy content in different channels.

**Returns:**
Dictionary with energy measures for each channel (velocity, magnetic, density, etc.)

---

#### `compute_scaling_exponents`
```python
compute_scaling_exponents(
    results: Dict, 
    p_values: List[float] = [1, 2, 3]
) -> Dict[str, np.ndarray]
```

Compute scaling exponents ζ_p for different moment orders.

**Parameters:**
- `results`: Output from analyze_slice
- `p_values`: List of moment orders to compute

**Returns:**
Dictionary with scaling exponents for each channel and order

---

#### `detect_transition_times`
```python
detect_transition_times(
    time_series_results: Dict,
    threshold: float = 0.1
) -> List[float]
```

Detect times when turbulence properties undergo significant changes.

**Parameters:**
- `time_series_results`: Output from analyze_time_series
- `threshold`: Relative change threshold for detection

**Returns:**
List of transition times

---

## Anisotropy Analysis Module

### `sfunctor.analysis.anisotropy`

#### `compute_scale_dependent_anisotropy`
```python
compute_scale_dependent_anisotropy(
    results: Dict,
    method: str = 'ratio',
    reference_direction: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]
```

Compute anisotropy measures as a function of scale.

**Parameters:**
- `results`: Output from analyze_slice
- `method`: Anisotropy measure ('ratio', 'variance', 'entropy', 'alignment')
- `reference_direction`: Reference direction for alignment (e.g., mean B field)

**Returns:**
Dictionary with scale-dependent anisotropy measures:
- `ell_centers`: Scale values
- `theta_centers`: Polar angle values
- `phi_centers`: Azimuthal angle values
- Method-specific measures (parallel_perp_ratio, angular_variance, etc.)
- `strength`: Overall anisotropy strength
- `preferential_theta/phi`: Dominant directions

---

#### `create_anisotropy_spectrogram`
```python
create_anisotropy_spectrogram(
    results: Dict,
    channel: int = 0,
    cmap: str = 'RdBu_r'
) -> Dict[str, np.ndarray]
```

Create a spectrogram showing anisotropy as a function of scale and angle.

**Parameters:**
- `results`: Output from analyze_slice
- `channel`: Which channel to analyze (0=velocity, 1=magnetic, etc.)
- `cmap`: Colormap name for visualization

**Returns:**
- `spectrogram`: Angular distribution at each scale
- `anisotropy_ratio`: Normalized anisotropy measure
- `ell_centers`: Scale values
- `theta_centers`: Angle values

---

#### `decompose_anisotropy_modes`
```python
decompose_anisotropy_modes(
    results: Dict,
    n_modes: int = 3
) -> Dict[str, np.ndarray]
```

Decompose anisotropy into principal modes using SVD.

**Parameters:**
- `results`: Output from analyze_slice
- `n_modes`: Number of modes to extract

**Returns:**
- `spatial_modes`: Scale/channel patterns
- `angular_modes`: Angular patterns
- `singular_values`: Mode strengths
- `variance_explained`: Fraction of variance per mode

---

## Cross-Correlation Module

### `sfunctor.analysis.cross_correlation`

#### `compute_field_correlations`
```python
compute_field_correlations(
    results: Dict,
    field_pairs: Optional[List[Tuple[int, int]]] = None,
    normalize: bool = True,
    compute_phase: bool = False
) -> Dict[str, np.ndarray]
```

Compute cross-correlations between different field channels.

**Parameters:**
- `results`: Output from analyze_slice
- `field_pairs`: List of (channel1, channel2) tuples to correlate
- `normalize`: Whether to normalize correlations to [-1, 1]
- `compute_phase`: Whether to compute phase relationships

**Returns:**
Dictionary with correlation measures for each field pair

---

#### `compute_scale_dependent_transfer`
```python
compute_scale_dependent_transfer(
    results: Dict,
    source_channel: int = 0,
    target_channel: int = 1,
    method: str = 'spectral'
) -> Dict[str, np.ndarray]
```

Compute scale-dependent energy transfer between fields.

**Parameters:**
- `results`: Output from analyze_slice
- `source_channel`: Source field channel index
- `target_channel`: Target field channel index
- `method`: Transfer calculation method ('spectral', 'flux', 'cascade')

**Returns:**
Dictionary with transfer functions and rates

---

#### `compute_nonlinear_coupling`
```python
compute_nonlinear_coupling(
    results: Dict,
    triplet: Tuple[int, int, int] = (0, 1, 7),
    compute_bispectrum: bool = True
) -> Dict[str, np.ndarray]
```

Compute nonlinear coupling between three field quantities.

**Parameters:**
- `results`: Output from analyze_slice
- `triplet`: Three channel indices for coupling analysis
- `compute_bispectrum`: Whether to compute bispectral measures

**Returns:**
- `triadic_strength`: Coupling strength at each scale
- `bicoherence`: Phase coupling measure (if requested)
- `exchange_*`: Energy exchange rates between fields

---

#### `compute_conditional_statistics`
```python
compute_conditional_statistics(
    results: Dict,
    condition_channel: int = 2,
    target_channel: int = 0,
    n_bins: int = 5,
    percentile_bins: bool = True
) -> Dict[str, np.ndarray]
```

Compute statistics of one field conditioned on another.

**Parameters:**
- `results`: Output from analyze_slice
- `condition_channel`: Channel to condition on
- `target_channel`: Channel to compute statistics for
- `n_bins`: Number of conditioning bins
- `percentile_bins`: Use percentile-based bins (True) or uniform (False)

**Returns:**
- `mean`: Conditional mean at each scale and bin
- `std`: Conditional standard deviation
- `skewness`: Conditional skewness
- `pdf`: Full conditional PDFs

---

#### `compute_mutual_information`
```python
compute_mutual_information(
    results: Dict,
    channel_pairs: Optional[List[Tuple[int, int]]] = None,
    n_bins: int = 20
) -> Dict[str, np.ndarray]
```

Compute mutual information between field pairs.

**Parameters:**
- `results`: Output from analyze_slice
- `channel_pairs`: List of channel pairs to analyze
- `n_bins`: Number of bins for entropy estimation

**Returns:**
Dictionary with mutual information and normalized MI for each pair

---

## Wavelet Analysis Module

### `sfunctor.analysis.wavelet`

#### `wavelet_decompose_sf`
```python
wavelet_decompose_sf(
    results: Dict,
    channel: int = 0,
    wavelet: str = 'db4',
    level: Optional[int] = None,
    mode: str = 'symmetric'
) -> Dict[str, np.ndarray]
```

Decompose structure functions using wavelet transform.

**Parameters:**
- `results`: Output from analyze_slice
- `channel`: Which channel to decompose
- `wavelet`: Wavelet type (e.g., 'db4', 'sym5', 'coif3')
- `level`: Decomposition level (None for maximum)
- `mode`: Signal extension mode

**Returns:**
- `approx`: Low-frequency approximation
- `details_horizontal/vertical/diagonal`: Detail coefficients
- `energy_per_scale`: Energy distribution
- `intermittency_map`: Detected intermittent regions
- `reconstructions`: Reconstructed signals at each level

---

#### `continuous_wavelet_analysis`
```python
continuous_wavelet_analysis(
    results: Dict,
    channel: int = 0,
    scales: Optional[np.ndarray] = None,
    wavelet: str = 'morl',
    angle_idx: Optional[int] = None
) -> Dict[str, np.ndarray]
```

Perform continuous wavelet transform for scale analysis.

**Parameters:**
- `results`: Output from analyze_slice
- `channel`: Which channel to analyze
- `scales`: Scales for CWT (None for automatic)
- `wavelet`: Mother wavelet ('morl', 'mexh', 'gaus8', etc.)
- `angle_idx`: Specific angle to analyze (None for average)

**Returns:**
- `coefficients`: CWT coefficients
- `scalogram`: Power at each scale/position
- `dominant_scales`: Identified dominant scales
- `intermittency_factor`: Scale-dependent intermittency
- `ridges`: Detected ridge lines

---

#### `wavelet_coherence_analysis`
```python
wavelet_coherence_analysis(
    results: Dict,
    channel1: int = 0,
    channel2: int = 1,
    scales: Optional[np.ndarray] = None,
    wavelet: str = 'morl'
) -> Dict[str, np.ndarray]
```

Compute wavelet coherence between two channels.

**Parameters:**
- `results`: Output from analyze_slice
- `channel1`: First channel
- `channel2`: Second channel
- `scales`: CWT scales
- `wavelet`: Mother wavelet

**Returns:**
- `coherence`: Coherence values (0 to 1)
- `phase_difference`: Phase lag between fields
- `significant_regions`: Regions of high coherence
- `mean_phase_lag`: Average phase at each scale

---

#### `multifractal_wavelet_analysis`
```python
multifractal_wavelet_analysis(
    results: Dict,
    channel: int = 0,
    q_values: Optional[np.ndarray] = None,
    n_scales: int = 16
) -> Dict[str, np.ndarray]
```

Perform multifractal analysis using wavelet leaders.

**Parameters:**
- `results`: Output from analyze_slice
- `channel`: Channel to analyze
- `q_values`: Moment orders for multifractal spectrum
- `n_scales`: Number of scales for analysis

**Returns:**
- `q_values`: Moment orders used
- `tau_q`: Scaling exponents
- `alpha`: Singularity spectrum
- `f_alpha`: Multifractal spectrum
- `width`: Multifractal width
- `alpha_min/max/peak`: Characteristic singularities

---

#### `identify_coherent_structures`
```python
identify_coherent_structures(
    wavelet_decomp: Dict,
    threshold_percentile: float = 95.0,
    min_size: int = 3
) -> Dict[str, np.ndarray]
```

Identify coherent structures from wavelet decomposition.

**Parameters:**
- `wavelet_decomp`: Output from wavelet_decompose_sf
- `threshold_percentile`: Percentile for thresholding coefficients
- `min_size`: Minimum structure size

**Returns:**
- `structures_per_level`: List of structures at each wavelet level
- `structure_maps`: Labeled structure maps
- `total_count`: Total number of structures
- `mean_size`: Average structure size
- `size_distribution`: Histogram of structure sizes

---

## Channel Indices

The following channel indices are used throughout the API:

| Index | Channel | Description |
|-------|---------|-------------|
| 0 | Velocity | Velocity magnitude increments |
| 1 | Magnetic | Magnetic field increments |
| 2 | Density | Density increments |
| 3 | Alfvén | Alfvén velocity increments |
| 4 | z⁺ | Elsässer variable (v + vA) |
| 5 | z⁻ | Elsässer variable (v - vA) |
| 6 | Vorticity | Vorticity increments |
| 7 | Current | Current density increments |
| 8 | Curvature | Magnetic curvature increments |
| 9 | Grad ρ | Density gradient increments |
| 10 | B/⟨B⟩ | Normalized magnetic increments |

## Error Handling

All functions include comprehensive error checking:

```python
try:
    results = analyze_time_series(files)
except ValueError as e:
    print(f"Invalid parameters: {e}")
except IOError as e:
    print(f"File error: {e}")
except ImportError as e:
    print(f"Missing dependency: {e}")
```

## Performance Tips

1. **Memory Management**
   - Use `stride` parameter to downsample large datasets
   - Enable `save_intermediate` for time-series analysis
   - Process in batches for very large datasets

2. **Computation Speed**
   - Reduce `n_displacements` for faster testing
   - Use smaller `n_random_subsamples` for quick analysis
   - Leverage GPU acceleration when available

3. **Parallel Processing**
   - Time-series analysis benefits from multiple cores
   - Set `n_processes` appropriately for your system
   - Consider MPI for cluster-scale analysis