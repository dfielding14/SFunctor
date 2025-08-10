#!/bin/bash
#SBATCH -A AST207
#SBATCH -J test_shared_mem
#SBATCH -o test_shared_mem_%j.out
#SBATCH -e test_shared_mem_%j.err
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56

# Test script to validate shared memory implementation on production data

echo "=============================================="
echo "Testing Shared Memory Implementation"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# Load modules
module reset
module load PrgEnv-gnu
module load python/3.11

# Activate virtual environment
cd /autofs/nccs-svm1_home2/dfielding/SFunctor
source venv_sfunctor/bin/activate

# Set test parameters
SLICE_DIR="/lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results/results_Turb_5120_beta6_dedt025_plm/ndisp10_000_nrand1_000_nell128_sw2_job3650342"
SLICE_FILE="${SLICE_DIR}/slice_0000.npz"
DISP_FILE="${SLICE_DIR}/displacements.npz"
OUTPUT_DIR="./test_shared_memory_output"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Test Configuration:"
echo "  Slice: $SLICE_FILE"
echo "  Displacements: $DISP_FILE"
echo "  Output: $OUTPUT_DIR"
echo ""

# Check if files exist
if [ ! -f "$SLICE_FILE" ]; then
    echo "ERROR: Slice file not found: $SLICE_FILE"
    exit 1
fi

if [ ! -f "$DISP_FILE" ]; then
    echo "ERROR: Displacements file not found: $DISP_FILE"
    exit 1
fi

echo "=============================================="
echo "Test 1: Single Process (baseline)"
echo "=============================================="

time python run_node_analysis.py \
    --slice "$SLICE_FILE" \
    --displacements "$DISP_FILE" \
    --node_id 0 \
    --total_nodes 1 \
    --output_dir "$OUTPUT_DIR/single_process" \
    --stride 4 \
    --n_processes 1 \
    --N_random_subsamples 100

if [ $? -eq 0 ]; then
    echo "✓ Single process test passed"
else
    echo "✗ Single process test failed"
    exit 1
fi

echo ""
echo "=============================================="
echo "Test 2: Multi-Process with Shared Memory"
echo "=============================================="

time python run_node_analysis.py \
    --slice "$SLICE_FILE" \
    --displacements "$DISP_FILE" \
    --node_id 0 \
    --total_nodes 1 \
    --output_dir "$OUTPUT_DIR/multi_process" \
    --stride 4 \
    --n_processes 14 \
    --N_random_subsamples 100

if [ $? -eq 0 ]; then
    echo "✓ Multi-process test passed"
else
    echo "✗ Multi-process test failed"
    exit 1
fi

echo ""
echo "=============================================="
echo "Test 3: Compare Results"
echo "=============================================="

python - <<EOF
import numpy as np

# Load results
single = np.load("$OUTPUT_DIR/single_process/histogram_slice_0000_node0000.npz")
multi = np.load("$OUTPUT_DIR/multi_process/histogram_slice_0000_node0000.npz")

# Compare histograms
hist_mag_match = np.allclose(single['hist_mag'], multi['hist_mag'], rtol=1e-5)
hist_other_match = np.allclose(single['hist_other'], multi['hist_other'], rtol=1e-5)

print(f"Histogram magnitude match: {hist_mag_match}")
print(f"Histogram other match: {hist_other_match}")

if hist_mag_match and hist_other_match:
    print("\n✓ Results match! Shared memory implementation validated.")
    exit(0)
else:
    print("\n✗ Results differ! Check implementation.")
    print(f"Max diff (mag): {np.max(np.abs(single['hist_mag'] - multi['hist_mag']))}")
    print(f"Max diff (other): {np.max(np.abs(single['hist_other'] - multi['hist_other']))}")
    exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "All Tests Passed Successfully!"
    echo "=============================================="
    echo ""
    echo "Memory Usage Summary:"
    echo "  Check test_shared_mem_${SLURM_JOB_ID}.out for memory details"
    echo ""
    echo "Next Steps:"
    echo "1. Review the test output for memory usage differences"
    echo "2. If tests pass, proceed with full-scale production run"
    echo "3. Monitor memory usage on larger slices (10240x10240)"
else
    echo ""
    echo "=============================================="
    echo "Tests Failed - Review Output"
    echo "=============================================="
    exit 1
fi