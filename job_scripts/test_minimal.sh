#!/bin/bash

# Minimal test to check if the code works at all

echo "Minimal functionality test"
echo "=========================="

# Use the small local test data
SLICE_FILE="./slice_data/Turb_320_beta100_dedt025_plm_axis3_slice0_file0000.npz"

if [ ! -f "$SLICE_FILE" ]; then
    echo "ERROR: Test slice not found"
    exit 1
fi

# Generate minimal displacements
echo "Generating minimal test displacements (10 displacements)..."
python generate_test_displacements.py \
    --n_disp_total 10 \
    --n_ell_bins 4 \
    --nres 320 \
    --output minimal_test_disp.npz

# Activate venv
source venv_frontier/bin/activate

echo ""
echo "Running minimal test (320x320 slice, 10 displacements, 10 random samples)..."
echo "This should complete in under 1 minute if working correctly"

time python run_node_analysis.py \
    --slice "$SLICE_FILE" \
    --displacements minimal_test_disp.npz \
    --node_id 0 \
    --total_nodes 1 \
    --output_dir ./test_minimal_output \
    --stride 1 \
    --n_processes 1 \
    --N_random_subsamples 10

if [ $? -eq 0 ]; then
    echo "✓ Minimal test passed!"
    echo "Basic functionality is working."
else
    echo "✗ Minimal test failed"
    echo "There's a fundamental issue with the code"
    exit 1
fi

# Clean up
rm -f minimal_test_disp.npz
rm -rf test_minimal_output

echo ""
echo "Now let's test what's taking so long with larger data..."
echo "Testing with stride=8 on 10240 slice (1280x1280 after stride)..."

# Generate small displacement set
python generate_test_displacements.py \
    --n_disp_total 100 \
    --n_ell_bins 8 \
    --nres 1280 \
    --output test_stride8_disp.npz

SLICE_10240="/lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results/slice_Turb_10240_beta25_dedt025_plm/Turb_10240_beta25_dedt025_plm_axis1_slice0p125_file0011.npz"

if [ -f "$SLICE_10240" ]; then
    echo "Starting stride=8 test (should complete in 2-3 minutes)..."
    
    time timeout 300 python run_node_analysis.py \
        --slice "$SLICE_10240" \
        --displacements test_stride8_disp.npz \
        --node_id 0 \
        --total_nodes 1 \
        --output_dir ./test_stride8_output \
        --stride 8 \
        --n_processes 1 \
        --N_random_subsamples 100
    
    if [ $? -eq 124 ]; then
        echo "✗ Test timed out after 5 minutes - code is stuck"
    elif [ $? -eq 0 ]; then
        echo "✓ Stride=8 test completed successfully"
    else
        echo "✗ Stride=8 test failed with error"
    fi
else
    echo "10240 slice not available, skipping large data test"
fi

# Clean up
rm -f test_stride8_disp.npz
rm -rf test_stride8_output

echo ""
echo "Test complete. Results:"
echo "- If minimal test passed: Basic code works, issue is with scale"
echo "- If stride=8 test timed out: Problem with larger data sizes"
echo "- If both passed: Original test parameters were too large"