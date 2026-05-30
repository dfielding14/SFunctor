# SFunctor Next Steps

## Critical Insight: GPU Strategy for Frontier

**IMPORTANT**: Frontier uses AMD MI250X GPUs, not NVIDIA GPUs. CuPy won't work there.

### GPU Acceleration Options for AMD (in order of preference):

1. **PyTorch with ROCm** (RECOMMENDED)
   - PyTorch has official ROCm support for AMD GPUs
   - Can use torch tensors instead of CuPy arrays
   - Installation on Frontier: `module load pytorch/rocm`
   - Minimal code changes needed

2. **JAX with ROCm**
   - JAX also supports AMD GPUs via ROCm
   - Good for functional programming style
   - XLA compilation can provide excellent performance

3. **Direct HIP Programming**
   - AMD's CUDA-equivalent 
   - Would require rewriting kernels in HIP C++
   - Most performant but most work

4. **Numba with ROCm**
   - Numba has experimental ROCm support
   - Could keep existing Numba kernels
   - Less mature than PyTorch/JAX options

### Recommended Approach

```python
# In sfunctor/core/gpu_backend.py
import numpy as np

# Try different GPU backends in order of preference
GPU_BACKEND = None
GPU_AVAILABLE = False

# Try PyTorch with ROCm/CUDA/MPS
try:
    import torch
    if torch.cuda.is_available():  # Works for both CUDA and ROCm
        GPU_BACKEND = 'torch_cuda'
        GPU_AVAILABLE = True
    elif torch.backends.mps.is_available():  # Apple Silicon
        GPU_BACKEND = 'torch_mps'
        GPU_AVAILABLE = True
except ImportError:
    pass

# Try CuPy for NVIDIA systems
if not GPU_AVAILABLE:
    try:
        import cupy as cp
        GPU_BACKEND = 'cupy'
        GPU_AVAILABLE = True
    except ImportError:
        pass

# Try JAX
if not GPU_AVAILABLE:
    try:
        import jax
        import jax.numpy as jnp
        if len(jax.devices('gpu')) > 0:
            GPU_BACKEND = 'jax'
            GPU_AVAILABLE = True
    except ImportError:
        pass

def to_gpu(array):
    """Transfer array to GPU using available backend."""
    if GPU_BACKEND == 'torch_cuda' or GPU_BACKEND == 'torch_mps':
        return torch.from_numpy(array).to('cuda' if 'cuda' in GPU_BACKEND else 'mps')
    elif GPU_BACKEND == 'cupy':
        return cp.asarray(array)
    elif GPU_BACKEND == 'jax':
        return jnp.array(array)
    else:
        return array  # CPU fallback

def from_gpu(array):
    """Transfer array from GPU to CPU."""
    if hasattr(array, 'cpu'):  # PyTorch
        return array.cpu().numpy()
    elif hasattr(array, 'get'):  # CuPy
        return array.get()
    elif GPU_BACKEND == 'jax':
        return np.array(array)
    else:
        return array  # Already CPU
```

## Immediate Tasks

### 1. Fix Test 02 (Physics Calculations)
- [ ] Check actual function names in `sfunctor/core/physics.py`
- [ ] Update imports in test 02 to match actual API
- [ ] Run test with high-resolution data

### 2. Complete Remaining Tests

**Test 05: Cross-Correlation**
- [ ] Test field-field correlations
- [ ] Test energy transfer analysis
- [ ] Test coherence analysis
- [ ] Create correlation matrix plots

**Test 06: Wavelet Decomposition**
- [ ] Install PyWavelets: `pip install PyWavelets`
- [ ] Test multi-scale decomposition
- [ ] Test wavelet coherence
- [ ] Generate wavelet scalograms

**Test 07: Wavelet Power Maps**
- [ ] Test spatial power spectrum mapping
- [ ] Generate amplitude and slope maps
- [ ] Test intermittency maps
- [ ] Validate against known power laws

**Test 08: GPU Acceleration**
- [ ] Implement GPU backend abstraction
- [ ] Test with PyTorch (most portable)
- [ ] Benchmark CPU vs GPU performance
- [ ] Profile memory usage

**Test 09: Visualization Gallery**
- [ ] Generate all plot types from `plot_structure_functions.py`
- [ ] Create publication-ready figures
- [ ] Add colorbars and proper labels
- [ ] Test different colormaps

### 3. Memory Optimization for Large Datasets

**Priority Areas:**
1. **Streaming/Chunking** in `sfunctor/io/slice_io.py`
   ```python
   def load_slice_npz_chunked(filename, chunk_size=1024):
       """Load slice in chunks for memory efficiency."""
       # Implementation needed
   ```

2. **Lazy Histogram Accumulation** in `sfunctor/core/histograms.py`
   - Process displacements in batches
   - Accumulate histograms incrementally
   - Use memory mapping for very large histograms

3. **Shared Memory Optimization** in `sfunctor/core/parallel.py`
   - Only create shared memory for active fields
   - Release shared memory after processing
   - Use memory pools for allocation

### 4. Performance Profiling

```bash
# Profile memory usage
mprof run python run_sf.py --file_name large_slice.npz
mprof plot

# Profile computation time
python -m cProfile -o profile.stats run_sf.py --file_name large_slice.npz
python -m pstats profile.stats

# Line-by-line profiling
kernprof -l -v run_sf.py --file_name large_slice.npz
```

### 5. Frontier-Specific Optimizations

**Module Setup on Frontier:**
```bash
module load python/3.9
module load rocm/5.4
module load pytorch/2.0-rocm5.4
module load cray-hdf5-parallel
module load cray-netcdf-parallel
```

**SLURM Script Template:**
```bash
#!/bin/bash
#SBATCH -A [project]
#SBATCH -J sfunctor
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -N 4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=8

module load python/3.9
module load rocm/5.4
module load pytorch/2.0-rocm5.4

srun -n 32 python run_sf.py --slice_list slices.txt --use_gpu
```

## Testing Priority Order

1. **Fix Test 02** - Physics calculations are fundamental
2. **Run Test 05** - Cross-correlations are scientifically important
3. **Implement GPU Backend** - Critical for Frontier performance
4. **Run Test 08** - Validate GPU acceleration
5. **Memory Optimization** - Needed for production runs
6. **Tests 06-07** - Wavelet features (if PyWavelets available)
7. **Test 09** - Final visualization gallery

## Performance Targets

- **2560³ slice processing**: < 5 minutes on single GPU
- **Memory usage**: < 32 GB for 2560² slice
- **Weak scaling**: 90% efficiency up to 128 nodes
- **Strong scaling**: 70% efficiency up to 32 GPUs

## Documentation Updates Needed

1. Add GPU backend selection to README
2. Document Frontier-specific setup
3. Add memory optimization guide
4. Create performance tuning guide
5. Update CLAUDE.md with AMD GPU info

## Key Decisions to Make

1. **Primary GPU Backend**: PyTorch (most portable) vs JAX (potentially faster)
2. **Memory Strategy**: Chunking vs memory mapping vs distributed arrays
3. **Histogram Precision**: float32 vs float64 for bins
4. **Parallelization**: MPI+GPU vs pure GPU with NCCL

## Success Metrics

- [ ] All tests pass with high-resolution data
- [ ] GPU acceleration working on AMD hardware
- [ ] Memory usage under 32GB for production runs
- [ ] Processing time < 5 min for 2560² slices
- [ ] Publication-quality plots generated
- [ ] Reproducible results across platforms
