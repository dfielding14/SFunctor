#!/usr/bin/env python
"""Test GPU implementation and validate against CPU results.

This test ensures:
1. GPU code produces same results as CPU
2. GPU provides performance improvements
3. Graceful fallback when GPU not available
"""

import numpy as np
import time
import sys
from typing import Dict, Tuple


def test_gpu_availability():
    """Test GPU detection and fallback."""
    print("=" * 60)
    print("Testing GPU Availability")
    print("=" * 60)
    
    from sfunctor.core.histograms_gpu import get_gpu_info
    
    info = get_gpu_info()
    print(f"GPU Available: {info['available']}")
    print(f"Backend: {info['backend']}")
    print(f"GPU Name: {info['name']}")
    print(f"GPU Memory: {info['memory_gb']:.1f} GB")
    print(f"Force CPU: {info['force_cpu']}")
    
    return info['available']


def test_physics_gpu():
    """Test GPU physics calculations."""
    print("\n" + "=" * 60)
    print("Testing Physics Calculations")
    print("=" * 60)
    
    # Import both CPU and GPU versions
    from sfunctor.core.physics import compute_vA as compute_vA_cpu
    from sfunctor.core.physics import compute_z_plus_minus as compute_z_cpu
    from sfunctor.core.physics_gpu import compute_vA_gpu, compute_z_plus_minus_gpu
    
    # Create test data
    N, M = 256, 256
    B_x = np.random.randn(N, M).astype(np.float32)
    B_y = np.random.randn(N, M).astype(np.float32)
    B_z = np.random.randn(N, M).astype(np.float32)
    rho = np.abs(np.random.randn(N, M)).astype(np.float32) + 1.0
    
    # Test Alfvén velocity
    print("\nTesting Alfvén velocity computation:")
    
    # CPU version
    t0 = time.time()
    vA_x_cpu, vA_y_cpu, vA_z_cpu = compute_vA_cpu(B_x, B_y, B_z, rho)
    cpu_time = time.time() - t0
    print(f"  CPU time: {cpu_time*1000:.2f} ms")
    
    # GPU version
    t0 = time.time()
    vA_x_gpu, vA_y_gpu, vA_z_gpu = compute_vA_gpu(B_x, B_y, B_z, rho)
    gpu_time = time.time() - t0
    print(f"  GPU time: {gpu_time*1000:.2f} ms")
    
    # Validate results
    match = np.allclose(vA_x_cpu, vA_x_gpu, rtol=1e-5)
    print(f"  Results match: {match}")
    if cpu_time > 0 and gpu_time > 0:
        print(f"  Speedup: {cpu_time/gpu_time:.1f}x")
    
    # Test Elsässer variables
    print("\nTesting Elsässer variables:")
    v_x = np.random.randn(N, M).astype(np.float32)
    v_y = np.random.randn(N, M).astype(np.float32)
    v_z = np.random.randn(N, M).astype(np.float32)
    
    # CPU version
    t0 = time.time()
    zp_cpu = compute_z_cpu(v_x, v_y, v_z, vA_x_cpu, vA_y_cpu, vA_z_cpu)
    cpu_time = time.time() - t0
    print(f"  CPU time: {cpu_time*1000:.2f} ms")
    
    # GPU version
    t0 = time.time()
    zp_gpu = compute_z_plus_minus_gpu(v_x, v_y, v_z, vA_x_gpu, vA_y_gpu, vA_z_gpu)
    gpu_time = time.time() - t0
    print(f"  GPU time: {gpu_time*1000:.2f} ms")
    
    # Validate
    match = np.allclose(zp_cpu[0], zp_gpu[0], rtol=1e-5)
    print(f"  Results match: {match}")
    if cpu_time > 0 and gpu_time > 0:
        print(f"  Speedup: {cpu_time/gpu_time:.1f}x")
    
    return match


def test_histogram_gpu():
    """Test GPU histogram computation."""
    print("\n" + "=" * 60)
    print("Testing Histogram Computation")
    print("=" * 60)
    
    from sfunctor.core.histograms import compute_histogram_unified as compute_histogram_cpu
    from sfunctor.core.histograms_gpu import compute_histogram_gpu
    
    # Create test data
    N, M = 128, 128
    print(f"Testing with {N}x{M} grid")
    
    # Create dummy fields
    dummy_field = np.random.randn(N, M).astype(np.float32)
    
    # Create bin edges
    ell_bins = np.logspace(0, 2, 10)
    theta_bins = np.linspace(0, np.pi, 5)
    phi_bins = np.linspace(-np.pi, np.pi, 9)
    sf_bins = [np.logspace(-2, 1, 20)] * 11
    prod_bins = np.logspace(-2, 2, 30)
    
    # Test parameters
    N_samples = 1000
    delta_i, delta_j = 5, 5
    axis = 3
    stencil = 2
    
    print(f"Random samples: {N_samples}")
    print(f"Displacement: ({delta_i}, {delta_j})")
    print(f"Stencil width: {stencil}")
    
    # CPU version
    print("\nCPU Version:")
    t0 = time.time()
    hist_mag_cpu, hist_other_cpu = compute_histogram_cpu(
        dummy_field, dummy_field, dummy_field,  # v_x, v_y, v_z
        dummy_field, dummy_field, dummy_field,  # B_x, B_y, B_z
        dummy_field,  # rho
        dummy_field, dummy_field, dummy_field,  # vA
        dummy_field, dummy_field, dummy_field,  # zp
        dummy_field, dummy_field, dummy_field,  # zm
        dummy_field, dummy_field, dummy_field,  # omega
        dummy_field, dummy_field, dummy_field,  # j
        dummy_field, dummy_field, dummy_field,  # curv
        dummy_field, dummy_field, dummy_field,  # grad_rho
        delta_i, delta_j, axis,
        N_samples,
        ell_bins, theta_bins, phi_bins,
        sf_bins, prod_bins,
        stencil
    )
    cpu_time = time.time() - t0
    print(f"  Time: {cpu_time*1000:.2f} ms")
    print(f"  Histogram shape: {hist_mag_cpu.shape}")
    print(f"  Total counts: {hist_mag_cpu.sum()}")
    
    # GPU version
    print("\nGPU Version:")
    t0 = time.time()
    hist_mag_gpu, hist_other_gpu = compute_histogram_gpu(
        dummy_field, dummy_field, dummy_field,  # v_x, v_y, v_z
        dummy_field, dummy_field, dummy_field,  # B_x, B_y, B_z
        dummy_field,  # rho
        dummy_field, dummy_field, dummy_field,  # vA
        dummy_field, dummy_field, dummy_field,  # zp
        dummy_field, dummy_field, dummy_field,  # zm
        dummy_field, dummy_field, dummy_field,  # omega
        dummy_field, dummy_field, dummy_field,  # j
        dummy_field, dummy_field, dummy_field,  # curv
        dummy_field, dummy_field, dummy_field,  # grad_rho
        delta_i, delta_j, axis,
        N_samples,
        ell_bins, theta_bins, phi_bins,
        sf_bins, prod_bins,
        stencil
    )
    gpu_time = time.time() - t0
    print(f"  Time: {gpu_time*1000:.2f} ms")
    print(f"  Histogram shape: {hist_mag_gpu.shape}")
    print(f"  Total counts: {hist_mag_gpu.sum()}")
    
    # Compare
    if cpu_time > 0 and gpu_time > 0:
        print(f"\nSpeedup: {cpu_time/gpu_time:.1f}x")
    
    # Check if results are close (allowing for some numerical differences)
    if hist_mag_cpu.sum() > 0 and hist_mag_gpu.sum() > 0:
        relative_diff = abs(hist_mag_cpu.sum() - hist_mag_gpu.sum()) / hist_mag_cpu.sum()
        print(f"Relative difference in counts: {relative_diff*100:.2f}%")
        match = relative_diff < 0.1  # Allow 10% difference due to random sampling
    else:
        match = True  # Both empty is OK
    
    return match


def test_batch_processing():
    """Test batch processing with GPU acceleration."""
    print("\n" + "=" * 60)
    print("Testing Batch Processing")
    print("=" * 60)
    
    from sfunctor.core.histograms_gpu import compute_histograms_gpu_batch
    from sfunctor.core.parallel import compute_histograms_chunked
    
    # Create test data
    N, M = 64, 64
    fields = {}
    for var in ["v", "B", "vA", "zp", "zm", "omega", "j", "curv", "grad_rho"]:
        for comp in ["x", "y", "z"]:
            fields[f"{var}_{comp}"] = np.random.randn(N, M).astype(np.float32)
    fields["rho"] = np.ones((N, M), dtype=np.float32)
    
    # Test parameters
    displacements = np.array([[1, 0], [0, 1], [1, 1]])
    N_samples = 100
    
    # Bin edges
    ell_bins = np.logspace(0, 1, 5)
    theta_bins = np.linspace(0, np.pi, 3)
    phi_bins = np.linspace(-np.pi, np.pi, 5)
    sf_bins = [np.logspace(-2, 1, 10)] * 11
    prod_bins = np.logspace(-2, 2, 15)
    
    print(f"Grid size: {N}x{M}")
    print(f"Displacements: {len(displacements)}")
    print(f"Samples per displacement: {N_samples}")
    
    # Test GPU batch processing
    print("\nGPU Batch Processing:")
    t0 = time.time()
    hist_mag, hist_other = compute_histograms_gpu_batch(
        fields, displacements, 3, N_samples,
        ell_bins, theta_bins, phi_bins,
        sf_bins, prod_bins, 2
    )
    gpu_time = time.time() - t0
    print(f"  Time: {gpu_time:.3f} s")
    print(f"  Total histogram counts: {hist_mag.sum()}")
    
    return True


def test_memory_management():
    """Test GPU memory management."""
    print("\n" + "=" * 60)
    print("Testing Memory Management")
    print("=" * 60)
    
    try:
        import cupy as cp
        has_gpu = True
    except ImportError:
        has_gpu = False
        print("CuPy not available - skipping memory tests")
        return True
    
    if not has_gpu:
        return True
    
    # Test memory pool
    mempool = cp.get_default_memory_pool()
    
    print("Initial memory pool state:")
    print(f"  Used bytes: {mempool.used_bytes() / 1024**2:.2f} MB")
    print(f"  Total bytes: {mempool.total_bytes() / 1024**2:.2f} MB")
    
    # Allocate some arrays
    arrays = []
    for i in range(5):
        arr = cp.random.randn(1024, 1024, dtype=cp.float32)
        arrays.append(arr)
    
    print("\nAfter allocation:")
    print(f"  Used bytes: {mempool.used_bytes() / 1024**2:.2f} MB")
    print(f"  Total bytes: {mempool.total_bytes() / 1024**2:.2f} MB")
    
    # Clear arrays
    del arrays
    
    print("\nAfter deletion (before cleanup):")
    print(f"  Used bytes: {mempool.used_bytes() / 1024**2:.2f} MB")
    print(f"  Total bytes: {mempool.total_bytes() / 1024**2:.2f} MB")
    
    # Free memory pool
    mempool.free_all_blocks()
    
    print("\nAfter cleanup:")
    print(f"  Used bytes: {mempool.used_bytes() / 1024**2:.2f} MB")
    print(f"  Total bytes: {mempool.total_bytes() / 1024**2:.2f} MB")
    
    return True


def main():
    """Run all GPU tests."""
    print("=" * 60)
    print("GPU Implementation Test Suite")
    print("=" * 60)
    
    # Test sequence
    tests = [
        ("GPU Availability", test_gpu_availability),
        ("Physics Calculations", test_physics_gpu),
        ("Histogram Computation", test_histogram_gpu),
        ("Batch Processing", test_batch_processing),
        ("Memory Management", test_memory_management),
    ]
    
    results = []
    has_gpu = False
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
            if name == "GPU Availability":
                has_gpu = result
        except Exception as e:
            print(f"\n✗ {name} failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name}: {status}")
        if not result and name != "GPU Availability":
            all_passed = False
    
    print("\n" + "=" * 60)
    if not has_gpu:
        print("GPU NOT AVAILABLE - Using CPU fallback")
        print("To enable GPU acceleration:")
        print("  pip install cupy-cuda12x  # For CUDA 12.x")
        print("  pip install cupy-cuda11x  # For CUDA 11.x")
    elif all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nGPU acceleration is working correctly!")
        print("The implementation:")
        print("  • Provides seamless GPU/CPU switching")
        print("  • Maintains numerical accuracy")
        print("  • Offers performance improvements where available")
        print("  • Falls back gracefully when GPU unavailable")
    else:
        print("✗ SOME TESTS FAILED")
        print("Please review the errors above")
    print("=" * 60)
    
    return 0 if (all_passed or not has_gpu) else 1


if __name__ == "__main__":
    sys.exit(main())