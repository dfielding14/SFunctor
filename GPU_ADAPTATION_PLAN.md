# GPU Adaptation Plan for SFunctor

## Executive Summary
Transform SFunctor's simplified codebase into a GPU-accelerated pipeline, targeting 10-100x speedup for structure function computation while maintaining CPU compatibility.

## Framework Analysis & Selection

### Option 1: CuPy (RECOMMENDED)
**Pros:**
- Drop-in NumPy replacement - minimal code changes
- Mature, stable, well-documented
- Direct CUDA kernel support when needed
- Good memory management tools
- Works on all NVIDIA GPUs

**Cons:**
- NVIDIA-only (but that's 90% of HPC)
- Some NumPy functions not implemented

**Decision: PRIMARY CHOICE** - Best balance of ease and performance

### Option 2: Numba CUDA
**Pros:**
- Already using Numba JIT
- Can write CUDA kernels in Python
- Fine-grained control

**Cons:**
- Steeper learning curve
- More complex memory management
- Harder to debug

**Decision: SECONDARY** - Use for specific kernels if needed

### Option 3: JAX
**Pros:**
- Functional programming model
- XLA compilation
- Works on TPUs

**Cons:**
- Complete rewrite needed
- Different programming paradigm
- Overkill for our use case

**Decision: NO** - Too much change

## Performance Bottleneck Analysis

### Current Hotspots (from profiling):
1. **Histogram computation (75% of runtime)**
   - Random sampling: Highly parallel
   - Bin finding: Binary search can be optimized
   - Accumulation: Needs atomic operations

2. **Physics calculations (15% of runtime)**
   - Element-wise operations: Perfect for GPU
   - No complex dependencies

3. **Data loading (10% of runtime)**
   - I/O bound, limited GPU benefit
   - Can overlap with computation

## GPU Implementation Strategy

### Phase 1: Core Histogram Kernel (Week 1)
```python
# Current CPU code (simplified)
def compute_histogram_unified(...):
    for idx in range(N_random_subsamples):
        i, j = random_points[idx]
        # Compute differences
        # Find bins
        # Accumulate
        
# GPU version
def compute_histogram_gpu(...):
    # Each thread handles one random sample
    # Use shared memory for bin edges
    # Atomic operations for accumulation
```

**Implementation Steps:**
1. Convert arrays to CuPy
2. Replace np.random with cupy.random
3. Implement custom CUDA kernel for core loop
4. Use atomic operations for histogram accumulation

### Phase 2: Physics Calculations (Week 2)
```python
# Simple replacements
def compute_vA_gpu(B_x, B_y, B_z, rho):
    # Just change np to cp
    import cupy as cp
    B_mag_sq = B_x**2 + B_y**2 + B_z**2
    return B_x/cp.sqrt(rho), B_y/cp.sqrt(rho), B_z/cp.sqrt(rho)
```

### Phase 3: Memory Optimization (Week 3)
- Implement memory pooling
- Use pinned memory for transfers
- Overlap computation with I/O using streams
- Batch processing to minimize transfers

### Phase 4: Multi-GPU Support (Week 4)
- Distribute slices across GPUs
- Use NCCL for efficient reduction
- Implement GPU-aware MPI

## Detailed Implementation Plan

### Week 1: Core Histogram GPU Kernel

**Day 1-2: Environment Setup**
```bash
# Create GPU development environment
pip install cupy-cuda12x
pip install cupyx
pip install pynvml  # For GPU monitoring
```

**Day 3-4: Basic CuPy Migration**
```python
# sfunctor/core/histograms_gpu.py
import cupy as cp
import numpy as np

def compute_histogram_gpu(
    fields_gpu,  # Already on GPU
    displacements,
    ...
):
    # Custom CUDA kernel for the hot loop
    kernel = cp.RawKernel(r'''
    extern "C" __global__
    void histogram_kernel(
        const float* field_data,
        const int* random_indices,
        const float* bin_edges,
        long long* histogram,
        int n_samples,
        int n_bins
    ) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= n_samples) return;
        
        // Load sample position
        int i = random_indices[idx * 2];
        int j = random_indices[idx * 2 + 1];
        
        // Compute differences (simplified)
        float diff = field_data[j * width + i] - field_data[...];
        
        // Binary search for bin (use shared memory)
        int bin = binary_search_shared(diff, bin_edges, n_bins);
        
        // Atomic accumulation
        atomicAdd(&histogram[bin], 1);
    }
    ''', 'histogram_kernel')
```

**Day 5: Testing & Validation**
- Compare GPU vs CPU results
- Benchmark performance
- Profile and optimize

### Week 2: Physics & Field Calculations

**Day 1-2: Port compute_vA and compute_z_plus_minus**
```python
def compute_physics_gpu(fields):
    """GPU version of physics calculations."""
    # Most operations are element-wise, easy port
    import cupy as cp
    
    # Transfer once if needed
    if not isinstance(fields['B_x'], cp.ndarray):
        fields = {k: cp.asarray(v) for k, v in fields.items()}
    
    # Compute derived fields
    fields['vA_x'], fields['vA_y'], fields['vA_z'] = compute_vA_gpu(...)
    return fields
```

**Day 3-4: Optimize memory patterns**
- Ensure coalesced access
- Use texture memory for bin edges
- Optimize block/grid dimensions

**Day 5: Integration testing**

### Week 3: Memory & I/O Optimization

**Day 1-2: Implement memory pools**
```python
# Reuse memory allocations
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

def process_with_memory_pool(data):
    with cp.cuda.Device(0):
        mempool.set_limit(size=8*1024**3)  # 8GB limit
        # Process data
        result = compute_histogram_gpu(data)
    return result
```

**Day 3-4: Stream-based overlap**
```python
def process_slices_gpu_streamed(slices):
    streams = [cp.cuda.Stream() for _ in range(2)]
    
    for i, slice_data in enumerate(slices):
        stream = streams[i % 2]
        with stream:
            # Async copy to GPU
            gpu_data = cp.asarray(slice_data)
            # Compute
            result = compute_histogram_gpu(gpu_data)
            # Async copy back
            cpu_result = cp.asnumpy(result)
        
        # Previous stream finishes while current processes
```

**Day 5: Benchmark I/O vs computation overlap**

### Week 4: Multi-GPU & Production

**Day 1-2: Multi-GPU distribution**
```python
def distribute_across_gpus(data, n_gpus):
    """Distribute computation across multiple GPUs."""
    import cupy as cp
    from cupy.cuda import nccl
    
    # Create NCCL communicator
    comm = nccl.NcclCommunicator(n_gpus)
    
    # Split data
    chunks = np.array_split(data, n_gpus)
    
    # Process on each GPU
    results = []
    for gpu_id, chunk in enumerate(chunks):
        with cp.cuda.Device(gpu_id):
            result = compute_histogram_gpu(chunk)
            results.append(result)
    
    # Reduce across GPUs
    final_result = nccl_reduce(results, comm)
    return final_result
```

**Day 3-4: Production integration**
- Add GPU detection and fallback
- Environment variable controls
- Logging and monitoring

**Day 5: Final benchmarks and documentation**

## Memory Management Strategy

### GPU Memory Hierarchy Utilization
1. **Global Memory**: Main field arrays (largest, slowest)
2. **Shared Memory**: Bin edges (frequently accessed)
3. **Registers**: Loop variables, temporary calculations
4. **Constant Memory**: Physical constants

### Transfer Optimization
```python
# Bad: Many small transfers
for field in fields:
    gpu_field = cp.asarray(field)
    process(gpu_field)

# Good: One large transfer
gpu_fields = {k: cp.asarray(v) for k, v in fields.items()}
process_all(gpu_fields)
```

## Performance Targets

### Expected Speedups
- **Histogram kernel**: 50-100x (highly parallel)
- **Physics calculations**: 10-20x (memory bandwidth limited)
- **End-to-end pipeline**: 10-30x (including I/O)

### Benchmarking Metrics
```python
def benchmark_gpu():
    """Compare GPU vs CPU performance."""
    sizes = [1024, 2048, 4096, 8192]
    
    for size in sizes:
        data = generate_test_data(size)
        
        # CPU timing
        t_cpu = time_cpu_version(data)
        
        # GPU timing (excluding first run compilation)
        _ = compute_histogram_gpu(data)  # Warmup
        t_gpu = time_gpu_version(data)
        
        print(f"Size {size}: CPU={t_cpu:.3f}s, GPU={t_gpu:.3f}s, "
              f"Speedup={t_cpu/t_gpu:.1f}x")
```

## Risk Mitigation

### Technical Risks
1. **Memory limitations**: Use chunking for large datasets
2. **GPU availability**: Always maintain CPU fallback
3. **Numerical differences**: Validate against CPU results
4. **Debugging complexity**: Extensive logging and checkpoints

### Implementation Risks
1. **Scope creep**: Stick to phases, don't optimize prematurely
2. **Compatibility**: Test on multiple GPU architectures
3. **Dependencies**: Minimize external GPU libraries

## Code Organization

```
sfunctor/
├── core/
│   ├── histograms.py       # CPU version
│   ├── histograms_gpu.py   # GPU version
│   ├── kernels/            # CUDA kernels
│   │   ├── histogram.cu
│   │   └── physics.cu
│   └── gpu_utils.py        # GPU helpers
├── backends/
│   ├── cpu.py             # CPU backend
│   ├── cupy_backend.py    # CuPy backend
│   └── selector.py        # Runtime selection
```

## Testing Strategy

### Unit Tests
```python
def test_histogram_gpu_vs_cpu():
    """Verify GPU produces same results as CPU."""
    data = generate_test_data()
    
    result_cpu = compute_histogram_cpu(data)
    result_gpu = compute_histogram_gpu(data)
    
    np.testing.assert_allclose(result_cpu, result_gpu, rtol=1e-5)
```

### Performance Tests
```python
def test_gpu_speedup():
    """Ensure GPU is actually faster."""
    data = generate_large_test_data()
    
    t_cpu = measure_time(compute_histogram_cpu, data)
    t_gpu = measure_time(compute_histogram_gpu, data)
    
    assert t_gpu < t_cpu * 0.5  # At least 2x speedup
```

## Success Criteria

### Week 1
- [ ] Basic histogram kernel running on GPU
- [ ] 10x speedup on histogram computation
- [ ] Tests passing with <0.1% numerical difference

### Week 2
- [ ] All physics calculations on GPU
- [ ] Integrated GPU pipeline
- [ ] 5x end-to-end speedup

### Week 3
- [ ] Memory optimizations implemented
- [ ] I/O overlap working
- [ ] 10x end-to-end speedup

### Week 4
- [ ] Multi-GPU support
- [ ] Production ready with fallbacks
- [ ] Documentation complete
- [ ] 20x speedup on multi-GPU systems

## Next Steps

1. **Immediate**: Install CuPy and create test environment
2. **Day 1**: Create `histograms_gpu.py` with basic structure
3. **Day 2**: Implement first GPU kernel
4. **Day 3**: Benchmark and iterate

## Notes

- Start simple, optimize later
- Always maintain CPU compatibility
- Test frequently with real data
- Document GPU-specific assumptions
- Consider power efficiency, not just speed