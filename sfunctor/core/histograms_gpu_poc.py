"""GPU-accelerated histogram computation using CuPy - Proof of Concept.

This module demonstrates how to adapt the simplified histogram computation
for GPU execution using CuPy as a drop-in NumPy replacement.

Key optimizations:
- Batch operations to minimize kernel launches
- Vectorized operations where possible
- Memory pooling for repeated allocations
- Option to keep data on GPU between operations
"""

import numpy as np
from typing import Tuple, Optional, List
import warnings

# Try to import CuPy, fall back to NumPy if not available
try:
    import cupy as cp
    HAS_GPU = True
    
    # Set up memory pool for efficient allocation
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
except ImportError:
    cp = np  # Fallback to NumPy
    HAS_GPU = False
    warnings.warn("CuPy not available, falling back to CPU implementation")

# Import CPU version for fallback and validation
from sfunctor.core.histograms import (
    N_MAG_CHANNELS,
    N_OTHER_CHANNELS,
    compute_histogram_unified as compute_histogram_cpu
)


def transfer_to_gpu(fields: dict, device_id: int = 0) -> dict:
    """Transfer field arrays to GPU memory.
    
    Args:
        fields: Dictionary of field arrays
        device_id: GPU device to use
        
    Returns:
        Dictionary with CuPy arrays on GPU
    """
    if not HAS_GPU:
        return fields
    
    with cp.cuda.Device(device_id):
        gpu_fields = {}
        for key, arr in fields.items():
            if not isinstance(arr, cp.ndarray):
                gpu_fields[key] = cp.asarray(arr)
            else:
                gpu_fields[key] = arr
    return gpu_fields


def compute_histogram_gpu_simple(
    # Field arrays (should already be on GPU)
    v_x, v_y, v_z, B_x, B_y, B_z, rho,
    vA_x, vA_y, vA_z, zp_x, zp_y, zp_z,
    zm_x, zm_y, zm_z, omega_x, omega_y, omega_z,
    J_x, J_y, J_z, curv_x, curv_y, curv_z,
    grad_rho_x, grad_rho_y, grad_rho_z,
    # Parameters
    delta_i, delta_j, slice_axis,
    N_random_subsamples,
    ell_bin_edges, theta_bin_edges, phi_bin_edges,
    sf_channel_bin_edges, product_bin_edges,
    stencil_width
):
    """GPU-accelerated histogram computation using CuPy.
    
    This is a simple proof-of-concept that uses CuPy's NumPy-compatible
    interface for easy migration. Further optimization with custom kernels
    is possible for production use.
    """
    if not HAS_GPU:
        # Fall back to CPU version
        return compute_histogram_cpu(
            v_x, v_y, v_z, B_x, B_y, B_z, rho,
            vA_x, vA_y, vA_z, zp_x, zp_y, zp_z,
            zm_x, zm_y, zm_z, omega_x, omega_y, omega_z,
            J_x, J_y, J_z, curv_x, curv_y, curv_z,
            grad_rho_x, grad_rho_y, grad_rho_z,
            delta_i, delta_j, slice_axis,
            N_random_subsamples,
            ell_bin_edges, theta_bin_edges, phi_bin_edges,
            sf_channel_bin_edges, product_bin_edges,
            stencil_width
        )
    
    # Ensure arrays are on GPU
    xp = cp.get_array_module(v_x)  # Will be cp if on GPU, np if on CPU
    
    N, M = v_x.shape
    n_ell_bins = len(ell_bin_edges) - 1
    n_theta_bins = len(theta_bin_edges) - 1  
    n_phi_bins = len(phi_bin_edges) - 1
    n_sf_bins = len(sf_channel_bin_edges[0]) - 1
    n_product_bins = len(product_bin_edges) - 1
    
    # Transfer bin edges to GPU if needed
    if xp == cp:
        ell_bin_edges = cp.asarray(ell_bin_edges)
        theta_bin_edges = cp.asarray(theta_bin_edges)
        phi_bin_edges = cp.asarray(phi_bin_edges)
        product_bin_edges = cp.asarray(product_bin_edges)
        sf_channel_bin_edges = [cp.asarray(edges) for edges in sf_channel_bin_edges]
    
    # Initialize histograms on GPU
    hist_mag = xp.zeros((N_MAG_CHANNELS, n_ell_bins, n_theta_bins, n_phi_bins, n_sf_bins), dtype=xp.int64)
    hist_other = xp.zeros((N_OTHER_CHANNELS, n_ell_bins, n_product_bins), dtype=xp.int64)
    
    # Determine displacement vector
    if slice_axis == 1:
        dx, dy, dz = 0, delta_i, delta_j
    elif slice_axis == 2:
        dx, dy, dz = delta_i, 0, delta_j
    else:
        dx, dy, dz = delta_i, delta_j, 0
    
    r = (delta_i * delta_i + delta_j * delta_j) ** 0.5
    
    # Generate random samples on GPU
    flat_indices = xp.random.choice(M * N, size=N_random_subsamples, replace=False)
    random_points_y = flat_indices // M
    random_points_x = flat_indices % M
    
    # Vectorized computation where possible
    # Note: Full vectorization is limited by the histogram accumulation
    # which requires atomic operations. A custom CUDA kernel would be
    # more efficient for this part.
    
    # For proof of concept, we'll process in batches to balance memory and speed
    batch_size = min(1000, N_random_subsamples)
    
    for batch_start in range(0, N_random_subsamples, batch_size):
        batch_end = min(batch_start + batch_size, N_random_subsamples)
        batch_idx = slice(batch_start, batch_end)
        
        # Get batch of random points
        i_batch = random_points_x[batch_idx]
        j_batch = random_points_y[batch_idx]
        
        # Compute indices
        jp = (j_batch + delta_j) % N
        ip = (i_batch + delta_i) % M
        
        # Compute differences for all fields (vectorized)
        if stencil_width == 2:
            # Simple 2-point differences
            dvx = v_x[jp, ip] - v_x[j_batch, i_batch]
            dvy = v_y[jp, ip] - v_y[j_batch, i_batch]
            dvz = v_z[jp, ip] - v_z[j_batch, i_batch]
            # ... similar for other fields
            
        # Compute magnitudes (vectorized)
        dv_mag = xp.sqrt(dvx**2 + dvy**2 + dvz**2)
        
        # Binning (this part would benefit from custom kernel)
        # For now, using searchsorted which is reasonably efficient on GPU
        ell_indices = xp.searchsorted(ell_bin_edges, xp.full(len(i_batch), r), side='right') - 1
        
        # Accumulate into histograms
        # Note: This is the main bottleneck that would benefit from custom kernels
        # CuPy doesn't have great support for arbitrary histogram accumulation
        # For POC, we'll do a simple loop (not optimal)
        for k in range(len(i_batch)):
            if ell_indices[k] >= 0 and ell_indices[k] < n_ell_bins:
                # Simplified - just incrementing one bin for demonstration
                # Real implementation would compute all channels
                hist_mag[0, ell_indices[k], 0, 0, 0] += 1
    
    # Transfer results back to CPU if needed
    if xp == cp:
        hist_mag = cp.asnumpy(hist_mag)
        hist_other = cp.asnumpy(hist_other)
    
    return hist_mag, hist_other


def compute_histogram_gpu_optimized(
    fields_gpu: dict,
    displacements: np.ndarray,
    axis: int,
    N_random_subsamples: int,
    ell_bin_edges: np.ndarray,
    theta_bin_edges: np.ndarray,
    phi_bin_edges: np.ndarray,
    sf_channel_bin_edges: List[np.ndarray],
    product_bin_edges: np.ndarray,
    stencil_width: int = 2,
    device_id: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Optimized GPU histogram computation with custom kernels.
    
    This version would include:
    - Custom CUDA kernels for the hot loops
    - Shared memory for bin edges
    - Atomic operations for histogram accumulation
    - Stream-based overlap of computation and data transfer
    
    For the POC, this just wraps the simple version.
    """
    
    if not HAS_GPU:
        # Fallback to CPU
        from sfunctor.core.parallel import compute_histograms_chunked
        return compute_histograms_chunked(
            fields_gpu, displacements, axis, N_random_subsamples,
            ell_bin_edges, theta_bin_edges, phi_bin_edges,
            sf_channel_bin_edges, product_bin_edges, stencil_width
        )
    
    with cp.cuda.Device(device_id):
        # Transfer fields to GPU once
        gpu_fields = transfer_to_gpu(fields_gpu, device_id)
        
        # Initialize combined histograms
        n_ell = len(ell_bin_edges) - 1
        n_theta = len(theta_bin_edges) - 1
        n_phi = len(phi_bin_edges) - 1
        n_sf = len(sf_channel_bin_edges[0]) - 1
        n_prod = len(product_bin_edges) - 1
        
        hist_mag_total = cp.zeros(
            (N_MAG_CHANNELS, n_ell, n_theta, n_phi, n_sf), 
            dtype=cp.int64
        )
        hist_other_total = cp.zeros(
            (N_OTHER_CHANNELS, n_ell, n_prod),
            dtype=cp.int64
        )
        
        # Process each displacement
        for dx, dy in displacements:
            hist_mag, hist_other = compute_histogram_gpu_simple(
                gpu_fields["v_x"], gpu_fields["v_y"], gpu_fields["v_z"],
                gpu_fields["B_x"], gpu_fields["B_y"], gpu_fields["B_z"],
                gpu_fields["rho"],
                gpu_fields["vA_x"], gpu_fields["vA_y"], gpu_fields["vA_z"],
                gpu_fields["zp_x"], gpu_fields["zp_y"], gpu_fields["zp_z"],
                gpu_fields["zm_x"], gpu_fields["zm_y"], gpu_fields["zm_z"],
                gpu_fields["omega_x"], gpu_fields["omega_y"], gpu_fields["omega_z"],
                gpu_fields["j_x"], gpu_fields["j_y"], gpu_fields["j_z"],
                gpu_fields["curv_x"], gpu_fields["curv_y"], gpu_fields["curv_z"],
                gpu_fields["grad_rho_x"], gpu_fields["grad_rho_y"], gpu_fields["grad_rho_z"],
                int(dx), int(dy), axis,
                N_random_subsamples,
                ell_bin_edges, theta_bin_edges, phi_bin_edges,
                sf_channel_bin_edges, product_bin_edges,
                stencil_width
            )
            
            # Accumulate on GPU
            hist_mag_total += cp.asarray(hist_mag)
            hist_other_total += cp.asarray(hist_other)
        
        # Transfer final results back to CPU
        hist_mag_cpu = cp.asnumpy(hist_mag_total)
        hist_other_cpu = cp.asnumpy(hist_other_total)
        
        # Clear GPU memory pool
        mempool.free_all_blocks()
        
    return hist_mag_cpu, hist_other_cpu


def benchmark_gpu_vs_cpu():
    """Simple benchmark comparing GPU and CPU performance."""
    import time
    
    print("GPU Benchmark (Proof of Concept)")
    print("=" * 50)
    
    if not HAS_GPU:
        print("CuPy not installed - cannot run GPU benchmark")
        print("Install with: pip install cupy-cuda12x")
        return
    
    # Create test data
    N, M = 512, 512
    print(f"Testing with {N}x{M} grid")
    
    # Create dummy fields
    fields = {}
    for var in ["v", "B", "vA", "zp", "zm", "omega", "j", "curv", "grad_rho"]:
        for comp in ["x", "y", "z"]:
            fields[f"{var}_{comp}"] = np.random.randn(N, M).astype(np.float32)
    fields["rho"] = np.ones((N, M), dtype=np.float32)
    
    # Test parameters
    displacements = np.array([[10, 0], [0, 10], [10, 10]])
    N_samples = 10000
    
    # Bin edges
    ell_bins = np.logspace(0, 2, 20)
    theta_bins = np.linspace(0, np.pi, 10)
    phi_bins = np.linspace(-np.pi, np.pi, 20)
    sf_bins = [np.logspace(-2, 1, 50)] * N_MAG_CHANNELS
    prod_bins = np.logspace(-2, 2, 50)
    
    print(f"Random samples per displacement: {N_samples}")
    print(f"Number of displacements: {len(displacements)}")
    
    # CPU timing
    print("\nCPU Version:")
    from sfunctor.core.parallel import compute_histograms_chunked
    
    t0 = time.time()
    hist_mag_cpu, hist_other_cpu = compute_histograms_chunked(
        fields, displacements, 3, N_samples,
        ell_bins, theta_bins, phi_bins,
        sf_bins, prod_bins, 2
    )
    cpu_time = time.time() - t0
    print(f"  Time: {cpu_time:.3f} seconds")
    print(f"  Histogram shape: {hist_mag_cpu.shape}")
    print(f"  Total counts: {hist_mag_cpu.sum()}")
    
    # GPU timing
    print("\nGPU Version:")
    
    # Warmup (for JIT compilation)
    print("  Warming up GPU...")
    _ = compute_histogram_gpu_optimized(
        fields, displacements[:1], 3, 100,
        ell_bins, theta_bins, phi_bins,
        sf_bins, prod_bins, 2
    )
    
    t0 = time.time()
    hist_mag_gpu, hist_other_gpu = compute_histogram_gpu_optimized(
        fields, displacements, 3, N_samples,
        ell_bins, theta_bins, phi_bins,
        sf_bins, prod_bins, 2
    )
    gpu_time = time.time() - t0
    print(f"  Time: {gpu_time:.3f} seconds")
    print(f"  Histogram shape: {hist_mag_gpu.shape}")
    print(f"  Total counts: {hist_mag_gpu.sum()}")
    
    # Compare results
    print(f"\nSpeedup: {cpu_time/gpu_time:.1f}x")
    print(f"Results match: {np.allclose(hist_mag_cpu, hist_mag_gpu, rtol=0.01)}")
    
    print("\nNote: This is a proof-of-concept. Production version would be faster with:")
    print("  - Custom CUDA kernels for histogram accumulation")
    print("  - Shared memory for bin edges")
    print("  - Better memory access patterns")
    print("  - Stream-based overlap")


if __name__ == "__main__":
    benchmark_gpu_vs_cpu()