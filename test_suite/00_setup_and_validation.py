#!/usr/bin/env python
"""
Test 00: Setup and Data Validation
Tests basic data loading and field validation.
"""

import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sfunctor.io.slice_io import load_slice_npz, parse_slice_metadata

# Test configuration
DEFAULT_TEST_FILE = "slice_data/Turb_2560_beta25_dedt025_plm_axis3_slicem0p375_file0024.npz"
FALLBACK_TEST_FILE = "slice_data/Turb_320_beta100_dedt025_plm_axis2_slice0_file0000.npz"
RESULTS_DIR = Path("test_suite/results")
PLOTS_DIR = RESULTS_DIR / "plots"
DATA_DIR = RESULTS_DIR / "data"

# Create directories
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

def test_setup(test_file: str | None = None):
    """Test 00: Setup and Data Validation"""
    
    print("="*60)
    print("TEST 00: SETUP AND DATA VALIDATION")
    print("="*60)
    
    # 1. Check file existence
    print("\n1. Checking test file...")
    selected = test_file or DEFAULT_TEST_FILE
    test_path = Path(selected)
    if not test_path.exists():
        test_path = Path(FALLBACK_TEST_FILE)
    if not test_path.exists():
        print(f"  ✗ Test file not found: {selected} (fallback: {FALLBACK_TEST_FILE})")
        return False
    
    print(f"  ✓ Found: {test_path}")
    print(f"  File size: {test_path.stat().st_size / 1e9:.2f} GB")
    
    # 2. Parse metadata
    print("\n2. Parsing metadata...")
    axis, beta = parse_slice_metadata(test_path)
    print(f"  ✓ Axis: {axis}")
    print(f"  ✓ Beta: {beta}")
    
    # 3. Load data with different strides
    print("\n3. Loading data with different strides...")
    
    strides = [1, 2, 4, 8]
    load_times = []
    data_shapes = []
    
    for stride in strides:
        import time
        start = time.time()
        data = load_slice_npz(test_path, stride=stride)
        elapsed = time.time() - start
        load_times.append(elapsed)
        
        shape = data['rho'].shape
        data_shapes.append(shape)
        print(f"  Stride {stride}: {shape} in {elapsed:.2f}s")
    
    # 4. Validate fields
    print("\n4. Validating fields...")
    data = load_slice_npz(test_path, stride=8)  # Use downsampled for speed
    
    required_fields = ['rho', 'v_x', 'v_y', 'v_z', 'B_x', 'B_y', 'B_z']
    optional_fields = ['omega_x', 'omega_y', 'omega_z', 'j_x', 'j_y', 'j_z',
                      'curv_x', 'curv_y', 'curv_z', 'grad_rho_x', 'grad_rho_y', 'grad_rho_z']
    
    print("  Required fields:")
    for field in required_fields:
        if field in data:
            print(f"    ✓ {field}: shape={data[field].shape}, "
                  f"mean={np.mean(data[field]):.3e}, std={np.std(data[field]):.3e}")
        else:
            print(f"    ✗ {field}: MISSING")
            return False
    
    print("  Optional fields:")
    for field in optional_fields:
        if field in data:
            print(f"    ✓ {field}: present")
        else:
            print(f"    - {field}: not present (will be computed)")
    
    # 5. Check data statistics
    print("\n5. Data statistics...")
    
    # Density
    print(f"  Density:")
    print(f"    Range: [{np.min(data['rho']):.3f}, {np.max(data['rho']):.3f}]")
    print(f"    Mean: {np.mean(data['rho']):.3f}")
    
    # Velocity
    v_mag = np.sqrt(data['v_x']**2 + data['v_y']**2 + data['v_z']**2)
    print(f"  Velocity magnitude:")
    print(f"    Range: [{np.min(v_mag):.3f}, {np.max(v_mag):.3f}]")
    print(f"    RMS: {np.sqrt(np.mean(v_mag**2)):.3f}")
    
    # Magnetic field
    B_mag = np.sqrt(data['B_x']**2 + data['B_y']**2 + data['B_z']**2)
    print(f"  Magnetic field magnitude:")
    print(f"    Range: [{np.min(B_mag):.3f}, {np.max(B_mag):.3f}]")
    print(f"    RMS: {np.sqrt(np.mean(B_mag**2)):.3f}")
    
    # Plasma beta
    v_rms = np.sqrt(np.mean(v_mag**2))
    B_rms = np.sqrt(np.mean(B_mag**2))
    rho_mean = np.mean(data['rho'])
    beta_computed = 2 * rho_mean * v_rms**2 / B_rms**2
    print(f"  Computed β ≈ {beta_computed:.1f} (expected: {beta})")
    
    # 6. Visualize fields
    print("\n6. Creating field visualizations...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    fields_to_plot = [
        ('rho', data['rho'], 'Density', 'RdBu_r'),
        ('v_mag', v_mag, 'Velocity Magnitude', 'viridis'),
        ('B_mag', B_mag, 'Magnetic Field Magnitude', 'plasma'),
        ('v_x', data['v_x'], 'Velocity X', 'RdBu_r'),
        ('v_y', data['v_y'], 'Velocity Y', 'RdBu_r'),
        ('v_z', data['v_z'], 'Velocity Z', 'RdBu_r'),
        ('B_x', data['B_x'], 'Magnetic Field X', 'RdBu_r'),
        ('B_z', data['B_z'], 'Magnetic Field Z', 'RdBu_r'),
    ]
    
    for ax, (name, field, title, cmap) in zip(axes.flat, fields_to_plot):
        im = ax.imshow(field, cmap=cmap, origin='lower')
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle(f'Field Visualization (stride=8)', fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '00_field_visualization.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {PLOTS_DIR}/00_field_visualization.png")
    
    # 7. Check memory usage
    print("\n7. Memory usage...")
    
    # Estimate memory for different strides
    for stride, shape in zip(strides, data_shapes):
        n_fields = len(required_fields) + len([f for f in optional_fields if f in data])
        mem_gb = n_fields * np.prod(shape) * 8 / 1e9  # 8 bytes per float64
        print(f"  Stride {stride}: ~{mem_gb:.2f} GB for all fields")
    
    # 8. Save test configuration
    print("\n8. Saving test configuration...")
    
    config = {
        'test_file': str(test_path),
        'axis': int(axis),
        'beta': float(beta),
        'full_shape': data_shapes[0],
        'n_required_fields': len(required_fields),
        'n_optional_fields': len([f for f in optional_fields if f in data]),
        'v_rms': float(v_rms),
        'B_rms': float(B_rms),
        'rho_mean': float(rho_mean),
        'beta_computed': float(beta_computed)
    }
    
    import json
    with open(DATA_DIR / '00_test_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Saved: {DATA_DIR}/00_test_config.json")
    
    print("\n" + "="*60)
    print("SETUP VALIDATION COMPLETE")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = test_setup()
    sys.exit(0 if success else 1)
