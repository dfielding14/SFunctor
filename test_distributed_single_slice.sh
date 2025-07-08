#!/bin/bash
# Simple test script to verify distributed analysis works on a single slice

echo "Testing distributed analysis pipeline"
echo "====================================="

# Set paths
SFUNCTOR_DIR="/autofs/nccs-svm1_home2/dfielding/SFunctor"
TEST_SLICE="/lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results/slices_Turb_640_beta25_dedt025_plm/Turb_640_beta25_dedt025_plm_axis1_slice0p375_file0024.npz"
WORK_DIR="./test_distributed_$(date +%Y%m%d_%H%M%S)"

# Configuration
N_NODES=4
N_DISP_TOTAL=1000  # Small for testing
N_RANDOM_SUBSAMPLES=100  # Small for testing

# Create working directory
mkdir -p $WORK_DIR
cd $WORK_DIR

# Step 1: Generate displacements
echo "1. Generating displacements..."
python ${SFUNCTOR_DIR}/generate_displacements.py \
    --n_disp_total $N_DISP_TOTAL \
    --n_ell_bins 32 \
    --output displacements.npz \
    --seed 42

# Step 2: Run analysis on each "node" (simulated)
echo ""
echo "2. Running analysis on $N_NODES simulated nodes..."
for node_id in $(seq 0 $((N_NODES - 1))); do
    echo "   - Node $node_id"
    python ${SFUNCTOR_DIR}/run_node_analysis.py \
        --slice "$TEST_SLICE" \
        --displacements displacements.npz \
        --node_id $node_id \
        --total_nodes $N_NODES \
        --output_dir . \
        --stride 8 \
        --N_random_subsamples $N_RANDOM_SUBSAMPLES \
        --n_processes 2 &
done

# Wait for all to complete
wait

# Step 3: Combine results
echo ""
echo "3. Combining histograms..."
python ${SFUNCTOR_DIR}/combine_histograms.py \
    --pattern "histogram_*.npz" \
    --output sf_results_combined.npz \
    --verbose

echo ""
echo "Test complete! Results in: $WORK_DIR"
ls -lh *.npz