#!/usr/bin/env python3
"""Test that the sfunctor package can be imported correctly."""

import sys

def test_imports():
    """Test basic imports from the sfunctor package."""
    try:
        # Test main package import
        import sfunctor
        print("✓ Main package import successful")
        print(f"  Version: {sfunctor.__version__}")
        
        # Test subpackage imports
        from sfunctor import core, analysis, io, visualization, utils
        print("✓ All subpackages import successful")
        
        # Test key function imports
        from sfunctor import load_slice_npz, analyze_slice, plot_structure_functions
        print("✓ Key functions import successful")
        
        # Test core module imports
        from sfunctor.core import compute_vA, compute_z_plus_minus, Channel
        print("✓ Core module imports successful")
        
        # Test utils imports
        from sfunctor.utils import parse_cli, build_displacement_list
        print("✓ Utils imports successful")
        
        print("\n✅ All imports successful! Package structure is working correctly.")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("\nMake sure to install the package first:")
        print("  pip install -e .")
        return False

if __name__ == "__main__":
    # Add current directory to path for testing without installation
    sys.path.insert(0, '.')
    
    success = test_imports()
    sys.exit(0 if success else 1)