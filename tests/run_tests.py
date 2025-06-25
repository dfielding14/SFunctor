#!/usr/bin/env python3
"""Simple test runner for SFunctor tests."""

import sys
import subprocess
from pathlib import Path

def run_tests():
    """Run all tests and display results."""
    # Get the project root directory
    tests_dir = Path(__file__).parent
    project_root = tests_dir.parent
    
    # Add project root to Python path
    sys.path.insert(0, str(project_root))
    
    print("Running SFunctor tests...")
    print("=" * 60)
    
    # Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        str(tests_dir),
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "-p", "no:warnings",  # Suppress warnings in test output
    ]
    
    # Try to run with coverage if available
    try:
        import pytest_cov
        cmd.extend(["--cov=.", "--cov-report=term-missing"])
    except ImportError:
        print("Note: Install pytest-cov for coverage reports")
    
    # Run the tests
    result = subprocess.run(cmd, cwd=str(project_root))
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(run_tests())