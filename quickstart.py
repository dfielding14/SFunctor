#!/usr/bin/env python3
"""Quick start script to verify installation and run a simple example."""

import sys
import os
from pathlib import Path

def check_installation():
    """Check if all required packages are installed."""
    print("Checking installation...")
    
    required_packages = {
        'numpy': 'numpy',
        'numba': 'numba', 
        'matplotlib': 'matplotlib',
        'scipy': 'scipy',
        'yaml': 'pyyaml',
        'cmasher': 'cmasher',
        'h5py': 'h5py'
    }
    
    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    # Check optional MPI
    try:
        import mpi4py
        print(f"✅ mpi4py (optional)")
    except ImportError:
        print(f"⚠️  mpi4py (optional, needed for multi-node runs)")
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    print("\n✅ All required packages installed!")
    return True

def check_data():
    """Check if example data exists."""
    print("\nChecking for example data...")
    
    data_dir = Path("data")
    slice_dir = Path("slice_data")
    
    if data_dir.exists() and list(data_dir.glob("*.bin")):
        print(f"✅ Found simulation data in {data_dir}/")
        return True
    elif slice_dir.exists() and list(slice_dir.glob("*.npz")):
        print(f"✅ Found slice data in {slice_dir}/")
        return True
    else:
        print("⚠️  No example data found")
        print("   Place .bin files in data/ or .npz slices in slice_data/")
        return False

def run_simple_example():
    """Run a simple example if data is available."""
    print("\nRunning simple example...")
    
    slice_files = list(Path("slice_data").glob("*.npz"))
    if not slice_files:
        print("⚠️  No slice files found to analyze")
        return
    
    # Use the first slice file
    slice_file = slice_files[0]
    print(f"Using slice: {slice_file}")
    
    # Create results directory
    Path("results").mkdir(exist_ok=True)
    
    # Run analysis with minimal parameters
    cmd = f"""python run_analysis.py \\
    --file_name {slice_file} \\
    --n_disp_total 100 \\
    --N_random_subsamples 100 \\
    --n_ell_bins 16 \\
    --stride 4 \\
    --n_processes 1"""
    
    print(f"\nRunning: {cmd}")
    print("\n(This is a quick test with minimal parameters)")
    
    os.system(cmd)

def main():
    """Main quickstart routine."""
    print("=== SFunctor Quick Start ===\n")
    
    # Check installation
    if not check_installation():
        print("\n❌ Please install missing packages before continuing")
        sys.exit(1)
    
    # Check for data
    has_data = check_data()
    
    # Print next steps
    print("\n=== Next Steps ===")
    
    if has_data:
        print("\n1. Run a quick test:")
        print("   python quickstart.py --run")
        
        print("\n2. Extract slices from simulation data:")
        print("   python run_extract_slice.py --sim_name <name> --axis 3 --offset 0.0 --file_numbers 0")
        
        print("\n3. Analyze slices:")
        print("   python run_analysis.py --file_name <slice.npz> --n_disp_total 1000")
        
        print("\n4. View results:")
        print("   python visualize_sf_results.py results/*.npz")
    else:
        print("\n1. Add simulation data:")
        print("   - Place .bin files in data/")
        print("   - Or place .npz slice files in slice_data/")
        
        print("\n2. Then run: python quickstart.py")
    
    print("\nFor more information, see README.md")
    
    # Run example if requested
    if len(sys.argv) > 1 and sys.argv[1] == "--run" and has_data:
        run_simple_example()

if __name__ == "__main__":
    main()