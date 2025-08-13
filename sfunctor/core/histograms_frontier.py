"""GPU implementation optimized for Frontier supercomputer (AMD MI250X GPUs).

Frontier specifications:
- 8 AMD GCDs (Graphics Compute Dies) per node (think of as 8 GPUs)
- 64 GB HBM2E per GCD (512 GB total GPU memory per node)
- 200 GB/s bandwidth between GCDs on same MI250X
- 36+36 GB/s H2D and D2H bandwidth
- ROCm/HIP for AMD GPU programming

This module provides:
1. Automatic detection of AMD vs NVIDIA GPUs
2. Optimized memory patterns for HBM2E
3. Multi-GCD support for Frontier nodes
4. Fallback to CuPy (NVIDIA) or CPU as needed
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Union
import warnings
import os
import platform

# Detect GPU backend
GPU_AVAILABLE = False
GPU_BACKEND = "none"
GPU_COUNT = 0

# Try ROCm/HIP for AMD GPUs (Frontier)
try:
    import rocm_smi
    # Check for AMD GPUs
    rocm_smi.initializeRsmi()
    GPU_COUNT = rocm_smi.getDeviceCount()
    if GPU_COUNT > 0:
        import cupy as cp  # CuPy supports ROCm too!
        GPU_AVAILABLE = True
        GPU_BACKEND = "rocm"
        print(f"ROCm backend detected with {GPU_COUNT} AMD GPUs")
except ImportError:
    pass

# Try CuPy for NVIDIA GPUs
if not GPU_AVAILABLE:
    try:
        import cupy as cp
        GPU_AVAILABLE = True
        GPU_BACKEND = "cuda"
        GPU_COUNT = cp.cuda.runtime.getDeviceCount()
    except ImportError:
        pass

# Fallback to NumPy
if not GPU_AVAILABLE:
    cp = np
    GPU_BACKEND = "cpu"

# Import CPU version for fallback
from sfunctor.core.histograms import (
    N_MAG_CHANNELS,
    N_OTHER_CHANNELS,
    compute_histogram_unified as compute_histogram_cpu,
)


def get_frontier_topology():
    """Get Frontier node topology if available."""
    topology = {
        'node_gpus': 8,  # 8 GCDs per node
        'gpus_per_mi250x': 2,  # 2 GCDs per MI250X
        'gpu_memory_gb': 64,  # Per GCD
        'cpu_cores': 64,
        'cpu_memory_gb': 512,
        'h2d_bandwidth_gb': 36,
        'gcd_bandwidth_gb': 200,  # Between GCDs on same MI250X
    }
    
    # Check if we're on Frontier
    hostname = platform.node()
    if 'frontier' in hostname.lower() or 'OLCF' in os.environ.get('LMOD_SYSTEM_NAME', ''):
        topology['on_frontier'] = True
        topology['actual_gpus'] = GPU_COUNT
    else:
        topology['on_frontier'] = False
        topology['actual_gpus'] = 0
    
    return topology


def distribute_work_frontier(n_tasks, n_gcds=8):
    """Distribute work optimally across Frontier GCDs.
    
    Frontier has 4 MI250X, each with 2 GCDs. We want to:
    1. Balance work across all 8 GCDs
    2. Minimize communication between MI250X units
    3. Exploit 200 GB/s bandwidth within MI250X pairs
    """
    if n_tasks < n_gcds:
        # Few tasks - use only as many GCDs as needed
        distribution = [(i, [i]) for i in range(n_tasks)]
    else:
        # Distribute tasks to minimize cross-MI250X communication
        tasks_per_gcd = n_tasks // n_gcds
        remainder = n_tasks % n_gcds
        
        distribution = []
        task_idx = 0
        for gcd in range(n_gcds):
            n_gcd_tasks = tasks_per_gcd + (1 if gcd < remainder else 0)
            gcd_tasks = list(range(task_idx, task_idx + n_gcd_tasks))
            distribution.append((gcd, gcd_tasks))
            task_idx += n_gcd_tasks
    
    return distribution


def compute_histogram_frontier_multi_gcd(
    fields: Dict[str, np.ndarray],
    displacements: np.ndarray,
    axis: int,
    N_random_subsamples: int,
    ell_bin_edges: np.ndarray,
    theta_bin_edges: np.ndarray,
    phi_bin_edges: np.ndarray,
    sf_channel_bin_edges: List[np.ndarray],
    product_bin_edges: np.ndarray,
    stencil_width: int = 2,
    use_gcds: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute histograms using multiple Frontier GCDs.
    
    This version is optimized for Frontier's architecture:
    - Distributes displacements across GCDs
    - Minimizes H2D transfers (36 GB/s bottleneck)
    - Exploits high bandwidth within MI250X pairs (200 GB/s)
    - Uses HBM2E efficiently (64 GB per GCD)
    """
    
    topology = get_frontier_topology()
    
    if not GPU_AVAILABLE or GPU_BACKEND == "cpu":
        print("No GPU available, using CPU fallback")
        from sfunctor.core.parallel import compute_histograms_chunked
        return compute_histograms_chunked(
            fields, displacements, axis, N_random_subsamples,
            ell_bin_edges, theta_bin_edges, phi_bin_edges,
            sf_channel_bin_edges, product_bin_edges, stencil_width
        )
    
    # Determine which GCDs to use
    if use_gcds is None:
        available_gcds = min(GPU_COUNT, topology['node_gpus'])
        use_gcds = list(range(available_gcds))
    
    n_gcds = len(use_gcds)
    print(f"Using {n_gcds} GCDs for computation")
    
    # Distribute displacements across GCDs
    distribution = distribute_work_frontier(len(displacements), n_gcds)
    
    # Initialize output histograms
    n_ell_bins = len(ell_bin_edges) - 1
    n_theta_bins = len(theta_bin_edges) - 1
    n_phi_bins = len(phi_bin_edges) - 1
    n_sf_bins = len(sf_channel_bin_edges[0]) - 1
    n_product_bins = len(product_bin_edges) - 1
    
    # Create histogram arrays on each GCD
    hist_mag_per_gcd = []
    hist_other_per_gcd = []
    
    for gcd_id in use_gcds:
        with cp.cuda.Device(gcd_id):
            hist_mag = cp.zeros(
                (N_MAG_CHANNELS, n_ell_bins, n_theta_bins, n_phi_bins, n_sf_bins),
                dtype=cp.int64
            )
            hist_other = cp.zeros(
                (N_OTHER_CHANNELS, n_ell_bins, n_product_bins),
                dtype=cp.int64
            )
            hist_mag_per_gcd.append(hist_mag)
            hist_other_per_gcd.append(hist_other)
    
    # Transfer fields to each GCD once (minimize H2D transfers)
    fields_per_gcd = []
    for gcd_id in use_gcds:
        with cp.cuda.Device(gcd_id):
            fields_gpu = {k: cp.asarray(v, dtype=np.float32) for k, v in fields.items()}
            fields_per_gcd.append(fields_gpu)
            
            # Report memory usage
            mempool = cp.get_default_memory_pool()
            used_gb = mempool.used_bytes() / (1024**3)
            print(f"  GCD {gcd_id}: {used_gb:.2f} GB used of 64 GB")
    
    # Process displacements on assigned GCDs
    from sfunctor.core.histograms_gpu import compute_histogram_gpu
    
    for gcd_idx, (gcd_id, task_indices) in enumerate(distribution):
        if gcd_idx >= n_gcds:
            break
            
        with cp.cuda.Device(use_gcds[gcd_idx]):
            for task_idx in task_indices:
                if task_idx >= len(displacements):
                    break
                    
                dx, dy = displacements[task_idx]
                
                # Compute histogram on this GCD
                hist_mag, hist_other = compute_histogram_gpu(
                    fields_per_gcd[gcd_idx]["v_x"],
                    fields_per_gcd[gcd_idx]["v_y"],
                    fields_per_gcd[gcd_idx]["v_z"],
                    fields_per_gcd[gcd_idx]["B_x"],
                    fields_per_gcd[gcd_idx]["B_y"],
                    fields_per_gcd[gcd_idx]["B_z"],
                    fields_per_gcd[gcd_idx]["rho"],
                    fields_per_gcd[gcd_idx]["vA_x"],
                    fields_per_gcd[gcd_idx]["vA_y"],
                    fields_per_gcd[gcd_idx]["vA_z"],
                    fields_per_gcd[gcd_idx]["zp_x"],
                    fields_per_gcd[gcd_idx]["zp_y"],
                    fields_per_gcd[gcd_idx]["zp_z"],
                    fields_per_gcd[gcd_idx]["zm_x"],
                    fields_per_gcd[gcd_idx]["zm_y"],
                    fields_per_gcd[gcd_idx]["zm_z"],
                    fields_per_gcd[gcd_idx]["omega_x"],
                    fields_per_gcd[gcd_idx]["omega_y"],
                    fields_per_gcd[gcd_idx]["omega_z"],
                    fields_per_gcd[gcd_idx]["j_x"],
                    fields_per_gcd[gcd_idx]["j_y"],
                    fields_per_gcd[gcd_idx]["j_z"],
                    fields_per_gcd[gcd_idx]["curv_x"],
                    fields_per_gcd[gcd_idx]["curv_y"],
                    fields_per_gcd[gcd_idx]["curv_z"],
                    fields_per_gcd[gcd_idx]["grad_rho_x"],
                    fields_per_gcd[gcd_idx]["grad_rho_y"],
                    fields_per_gcd[gcd_idx]["grad_rho_z"],
                    int(dx), int(dy), axis,
                    N_random_subsamples,
                    ell_bin_edges, theta_bin_edges, phi_bin_edges,
                    sf_channel_bin_edges, product_bin_edges,
                    stencil_width
                )
                
                # Accumulate on GCD
                hist_mag_per_gcd[gcd_idx] += cp.asarray(hist_mag)
                hist_other_per_gcd[gcd_idx] += cp.asarray(hist_other)
    
    # Reduce across GCDs
    # On Frontier, GCDs 0-1 are on MI250X #0, 2-3 on #1, etc.
    # We can use high-bandwidth links within pairs
    
    # First reduce within MI250X pairs (200 GB/s bandwidth)
    if n_gcds > 1:
        for mi250x in range(0, n_gcds, 2):
            if mi250x + 1 < n_gcds:
                # GCDs on same MI250X - use fast link
                with cp.cuda.Device(use_gcds[mi250x]):
                    hist_mag_per_gcd[mi250x] += hist_mag_per_gcd[mi250x + 1]
                    hist_other_per_gcd[mi250x] += hist_other_per_gcd[mi250x + 1]
    
    # Then reduce across MI250X units (slower)
    with cp.cuda.Device(use_gcds[0]):
        hist_mag_total = hist_mag_per_gcd[0]
        hist_other_total = hist_other_per_gcd[0]
        
        for mi250x in range(2, n_gcds, 2):
            if mi250x < n_gcds:
                hist_mag_total += hist_mag_per_gcd[mi250x]
                hist_other_total += hist_other_per_gcd[mi250x]
    
    # Transfer final results to CPU
    hist_mag_cpu = cp.asnumpy(hist_mag_total)
    hist_other_cpu = cp.asnumpy(hist_other_total)
    
    # Clean up GPU memory
    for gcd_id in use_gcds:
        with cp.cuda.Device(gcd_id):
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
    
    return hist_mag_cpu, hist_other_cpu


def benchmark_frontier_multi_gcd():
    """Benchmark multi-GCD performance for Frontier."""
    import time
    
    print("=" * 60)
    print("Frontier Multi-GCD Benchmark")
    print("=" * 60)
    
    topology = get_frontier_topology()
    print(f"Frontier topology: {topology}")
    
    if not GPU_AVAILABLE:
        print("No GPUs available for benchmarking")
        return
    
    # Create test data
    N, M = 1024, 1024  # Larger for Frontier scale
    fields = {}
    for var in ["v", "B", "vA", "zp", "zm", "omega", "j", "curv", "grad_rho"]:
        for comp in ["x", "y", "z"]:
            fields[f"{var}_{comp}"] = np.random.randn(N, M).astype(np.float32)
    fields["rho"] = np.ones((N, M), dtype=np.float32)
    
    # Test parameters
    displacements = np.array([[i, j] for i in range(10) for j in range(10)])
    N_samples = 10000
    
    # Bin edges
    ell_bins = np.logspace(0, 2, 30)
    theta_bins = np.linspace(0, np.pi, 16)
    phi_bins = np.linspace(-np.pi, np.pi, 32)
    sf_bins = [np.logspace(-2, 1, 100)] * N_MAG_CHANNELS
    prod_bins = np.logspace(-2, 2, 100)
    
    print(f"\nTest configuration:")
    print(f"  Grid size: {N}x{M}")
    print(f"  Displacements: {len(displacements)}")
    print(f"  Samples per displacement: {N_samples}")
    
    # Test with different GCD counts
    for n_gcds in [1, 2, 4, 8]:
        if n_gcds > GPU_COUNT:
            break
            
        print(f"\nTesting with {n_gcds} GCDs:")
        use_gcds = list(range(n_gcds))
        
        t0 = time.time()
        hist_mag, hist_other = compute_histogram_frontier_multi_gcd(
            fields, displacements, 3, N_samples,
            ell_bins, theta_bins, phi_bins,
            sf_bins, prod_bins, 2,
            use_gcds=use_gcds
        )
        elapsed = time.time() - t0
        
        throughput = len(displacements) / elapsed
        print(f"  Time: {elapsed:.2f} seconds")
        print(f"  Throughput: {throughput:.1f} displacements/second")
        print(f"  Total histogram counts: {hist_mag.sum()}")
        
        # Estimate Frontier full-node performance
        if topology['on_frontier']:
            node_throughput = throughput * (8 / n_gcds)
            print(f"  Estimated full-node throughput: {node_throughput:.1f} disp/s")


# Backwards compatibility
compute_histograms_shared = compute_histogram_frontier_multi_gcd

if __name__ == "__main__":
    benchmark_frontier_multi_gcd()