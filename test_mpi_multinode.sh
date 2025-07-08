#!/bin/bash
#SBATCH -A AST207
#SBATCH -J SF_TEST_MULTINODE
#SBATCH -o sf_test_multinode.%j.out
#SBATCH -t 0:45:00
#SBATCH -p batch
#SBATCH -N 4
#SBATCH --ntasks-per-node=32

# Test script for 4+ node MPI runs with SFunctor run_analysis.py

echo "============================================"
echo "SFunctor Multi-Node (4+) MPI Scaling Test"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks: $SLURM_NTASKS"
echo "============================================"

# Load required modules
module reset
module load gcc/9.3.0 python/.3.11-anaconda3 openmpi/4.0.4 hdf5/1.10.7

# Set environment variables for MPI
export MPICC=$(which mpicc)
export CC=$MPICC
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

# Activate virtual environment
source /ccs/home/dfielding/SFunctor/venv_sfunctor/bin/activate

# Set paths
SFUNCTOR_DIR="/autofs/nccs-svm1_home2/dfielding/SFunctor"
SLICE_LIST="/lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results/slice_list_Turb_640_beta25_dedt025_plm.txt"
CONFIG_FILE="$HOME/SFunctor/examples/configs/profiles.yaml"
RESULTS_BASE="/lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results/test_mpi_multinode"

# Create results directory
mkdir -p $RESULTS_BASE

echo ""
echo "Test configuration:"
echo "- Config file: $CONFIG_FILE"
echo "- Slice list: $SLICE_LIST"
echo "- Total slices available: $(wc -l < $SLICE_LIST 2>/dev/null || echo 0)"
echo "- Results base: $RESULTS_BASE"
echo ""

# First, run the multi-node detection test
echo "============================================"
echo "Testing Multi-Node Detection (expecting 4 nodes)"
echo "============================================"
srun -n 8 python ${SFUNCTOR_DIR}/test_multinode_fix.py
echo ""

# Function to run scaling test
run_scaling_test() {
    local n_ranks=$1
    local test_name=$2
    local n_slices=$3
    local profile=$4
    
    echo "----------------------------------------"
    echo "Running scaling test: $test_name"
    echo "Number of MPI ranks: $n_ranks"
    echo "Number of slices: $n_slices"
    echo "Profile: $profile"
    echo "----------------------------------------"
    
    # Create subdirectory for this test
    test_dir="${RESULTS_BASE}/${test_name}"
    mkdir -p $test_dir
    cd $test_dir
    
    # Create slice list with specified number of slices
    head -${n_slices} $SLICE_LIST > test_slice_list.txt
    actual_slices=$(wc -l < test_slice_list.txt)
    echo "Using $actual_slices slices for test"
    
    # Run the analysis with timing
    start_time=$(date +%s)
    
    srun -n $n_ranks python ${SFUNCTOR_DIR}/run_analysis.py \
        --config $CONFIG_FILE \
        --profile $profile \
        --slice_list test_slice_list.txt \
        2>&1 | tee run_${test_name}.log
    
    exit_code=$?
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    
    echo ""
    echo "Test completed in $elapsed seconds"
    echo "Exit code: $exit_code"
    echo "Performance: $(echo "scale=2; $actual_slices / $elapsed" | bc) slices/second"
    
    # Check if output files were created
    n_output_files=$(ls sf_results_*.npz 2>/dev/null | wc -l)
    if [ $n_output_files -gt 0 ]; then
        echo "Output files created: $n_output_files"
        ls -lh sf_results_*.npz | head -5
        if [ $n_output_files -gt 5 ]; then
            echo "... and $(($n_output_files - 5)) more files"
        fi
    else
        echo "ERROR: No output files found!"
    fi
    
    cd $RESULTS_BASE
    echo ""
    
    return $exit_code
}

# Test 1: Weak scaling - increase slices with ranks
echo "============================================"
echo "WEAK SCALING TESTS"
echo "============================================"

# 32 ranks (1 node) - 8 slices
run_scaling_test 32 "weak_1node_32ranks" 8 "cluster"

# 64 ranks (2 nodes) - 16 slices
run_scaling_test 64 "weak_2nodes_64ranks" 16 "cluster"

# 128 ranks (4 nodes) - 32 slices
run_scaling_test 128 "weak_4nodes_128ranks" 32 "cluster"

# Test 2: Strong scaling - fixed workload
echo ""
echo "============================================"
echo "STRONG SCALING TESTS (24 slices)"
echo "============================================"

# 32 ranks (1 node)
run_scaling_test 32 "strong_1node_32ranks" 24 "cluster"

# 64 ranks (2 nodes)
run_scaling_test 64 "strong_2nodes_64ranks" 24 "cluster"

# 128 ranks (4 nodes)
run_scaling_test 128 "strong_4nodes_128ranks" 24 "cluster"

# Test 3: Production profile test with full node utilization
echo ""
echo "============================================"
echo "PRODUCTION PROFILE TEST"
echo "============================================"

# Use dev profile (compromise between test and production)
run_scaling_test 128 "production_4nodes_128ranks" 48 "dev"

# Test 4: Load imbalance test
echo ""
echo "============================================"
echo "LOAD IMBALANCE TEST"
echo "============================================"

# Test with prime number of slices to create imbalance
run_scaling_test 128 "imbalance_4nodes_128ranks" 47 "test"

# Test 5: Cluster profile test
echo ""
echo "============================================"
echo "CLUSTER PROFILE TEST"
echo "============================================"

# Test cluster profile optimized for multi-node
run_scaling_test 128 "cluster_4nodes_128ranks" 64 "cluster"

# Summary
echo ""
echo "============================================"
echo "Test Summary"
echo "============================================"
echo "Results saved in: $RESULTS_BASE"
echo ""
echo "MPI Information:"
echo "- SLURM_JOB_ID: $SLURM_JOB_ID"
echo "- SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "- SLURM_NTASKS: $SLURM_NTASKS"
echo ""

# Display performance summary
echo "Performance summary:"
echo ""
echo "Weak Scaling (slices/second):"
for test in weak_1node_32ranks weak_2nodes_64ranks weak_4nodes_128ranks; do
    if [ -f "${RESULTS_BASE}/${test}/run_${test}.log" ]; then
        perf=$(grep "Performance:" "${RESULTS_BASE}/${test}/run_${test}.log" | awk '{print $2}')
        if [ ! -z "$perf" ]; then
            echo "  $test: $perf slices/second"
        fi
    fi
done

echo ""
echo "Strong Scaling (total time for 24 slices):"
for test in strong_1node_32ranks strong_2nodes_64ranks strong_4nodes_128ranks; do
    if [ -f "${RESULTS_BASE}/${test}/run_${test}.log" ]; then
        elapsed=$(grep "Test completed in" "${RESULTS_BASE}/${test}/run_${test}.log" | awk '{print $4}')
        if [ ! -z "$elapsed" ]; then
            echo "  $test: ${elapsed}s"
        fi
    fi
done

echo ""
echo "Scaling efficiency can be calculated from these results."
echo "Use verify_results.py to ensure numerical consistency across all runs."
echo ""
echo "Multi-node specific checks:"
echo "- Check logs for multi-node detection messages"
echo "- Verify multiprocessing was disabled (n_processes=1)"
echo ""
echo "Checking for multiprocessing override warnings in logs:"
for dir in ${RESULTS_BASE}/*/; do
    if [ -f "${dir}run_*.log" ]; then
        test_name=$(basename "$dir")
        if grep -q "WARNING: Multi-node MPI detected" "${dir}run_*.log" 2>/dev/null; then
            echo "  $test_name: Multi-node warning found ✓"
        else
            echo "  $test_name: No multi-node warning (single node test?)"
        fi
    fi
done