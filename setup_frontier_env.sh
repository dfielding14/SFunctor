#!/bin/bash
# Setup script for creating a virtual environment on ORNL Frontier
# This ensures all dependencies are properly installed with system-specific configurations

echo "Setting up SFunctor environment for Frontier..."

# Load necessary modules
echo "Loading modules..."
module reset
module load PrgEnv-gnu

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv_frontier

# Activate environment
source venv_frontier/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install packages
echo "Installing Python packages..."
pip install numpy numba matplotlib cmasher scipy h5py

# Install mpi4py with Cray compiler wrapper
echo "Building mpi4py for Frontier's MPI..."
MPICC=cc pip install --no-binary mpi4py mpi4py --no-cache-dir

echo ""
echo "======================================"
echo "Environment setup complete!"
echo "======================================"
echo "To activate this environment in the future, run:"
echo "  module reset && module load PrgEnv-gnu"
echo "  source venv_frontier/bin/activate"
echo ""
echo "Installed packages:"
pip list | grep -E "numpy|scipy|numba|matplotlib|cmasher|mpi4py|h5py"