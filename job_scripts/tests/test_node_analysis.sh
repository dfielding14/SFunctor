#!/bin/bash
# Test script to diagnose node analysis issues

echo "Testing node analysis setup..."

# Load modules
echo "Loading modules..."
module reset
module load PrgEnv-gnu

# Activate virtual environment
echo "Activating virtual environment..."
source /autofs/nccs-svm1_home2/dfielding/SFunctor/venv_frontier/bin/activate

# Check Python and key dependencies
echo "Python version:"
which python
python --version

echo "Checking key imports..."
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import numba; print('numba:', numba.__version__)"
python -c "import scipy; print('scipy:', scipy.__version__)"

# Check if scripts/production/run_node_analysis.py can be imported
echo "Checking scripts/production/run_node_analysis.py..."
python -c "import sys; sys.path.insert(0, '/autofs/nccs-svm1_home2/dfielding/SFunctor'); from scripts.production import run_node_analysis; print('scripts/production/run_node_analysis.py imports successfully')"

# Try a minimal run with very small parameters
echo "Testing minimal run..."
# Use a smaller test file if available, or the 10240 file
SLICE_5120="/lustre/orion/ast207/proj-shared/dfielding/Production_plm/2d_slices_Turb_5120_beta6_dedt025_plm/Turb_5120_beta6_dedt025_plm_axis1_slice0p125_file0024.npz"
SLICE_10240="/lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results/slice_Turb_10240_beta25_dedt025_plm/Turb_10240_beta25_dedt025_plm_axis1_slice0p125_file0011.npz"

if [ -f "$SLICE_5120" ]; then
    SLICE_PATH="$SLICE_5120"
    GRID_RES=5120
    echo "Using 5120 resolution slice for testing"
elif [ -f "$SLICE_10240" ]; then
    SLICE_PATH="$SLICE_10240"
    GRID_RES=10240
    echo "Using 10240 resolution slice for testing"
else
    echo "ERROR: No slice file found"
    exit 1
fi

if [ -f "$SLICE_PATH" ]; then
    echo "Slice file exists: $SLICE_PATH"
    
    # Create a test displacement file with just a few vectors
    echo "Generating test displacements..."
    python /autofs/nccs-svm1_home2/dfielding/SFunctor/scripts/production/generate_displacements.py \
        --n_disp_total 100 \
        --n_ell_bins 10 \
        --Nres $GRID_RES \
        --stencil_width 2 \
        --output test_displacements.npz \
        --seed 42
    
    if [ ! -f "test_displacements.npz" ]; then
        echo "ERROR: Failed to create displacement file"
        exit 1
    fi
    echo "Test displacement file created"
    
    # Try to run node analysis with minimal parameters
    echo "Running node analysis test..."
    python /autofs/nccs-svm1_home2/dfielding/SFunctor/scripts/production/run_node_analysis.py \
        --slice "$SLICE_PATH" \
        --displacements test_displacements.npz \
        --node_id 0 \
        --total_nodes 1 \
        --output_dir test_output \
        --stride 8 \
        --N_random_subsamples 100 \
        --stencil_width 2 \
        --n_processes 1 \
        --N_delta_bin_edges 51
    
    # Check if output was created
    HISTOGRAM_COUNT=$(ls -1 test_output/histogram_*.npz 2>/dev/null | wc -l)
    if [ "$HISTOGRAM_COUNT" -gt 0 ]; then
        echo "SUCCESS: $HISTOGRAM_COUNT histogram file(s) created"
        ls -la test_output/
    else
        echo "FAILED: No histogram file created"
        if [ -d "test_output" ]; then
            echo "Output directory contents:"
            ls -la test_output/
        fi
    fi
    
    # Clean up
    rm -f test_displacements.npz
    rm -rf test_output
else
    echo "ERROR: Slice file not found: $SLICE_PATH"
fi

echo "Test complete"
