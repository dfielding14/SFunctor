#!/bin/bash
#SBATCH -A AST207
#SBATCH -J SF_TEST_2NODES
#SBATCH -o sf_test_2nodes.%j.out
#SBATCH -t 0:30:00
#SBATCH -p batch
#SBATCH -N 2
#SBATCH --ntasks-per-node=32

# Test script for 2-node MPI runs with SFunctor run_analysis.py

echo "=========================================="
echo "SFunctor 2-Node MPI Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks: $SLURM_NTASKS"
echo "=========================================="

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
TEST_SLICE="/lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results/slices_Turb_640_beta25_dedt025_plm/Turb_640_beta25_dedt025_plm_axis1_slice0p375_file0024.npz"
SLICE_LIST="/lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results/slice_list_Turb_640_beta25_dedt025_plm.txt"
CONFIG_FILE="$HOME/SFunctor/examples/configs/profiles.yaml"
RESULTS_BASE="/lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results/test_mpi_2nodes"

# Create results directory
mkdir -p $RESULTS_BASE

echo ""
echo "Test configuration:"
echo "- Config file: $CONFIG_FILE"
echo "- Test slice: $TEST_SLICE"
echo "- Slice list: $SLICE_LIST"
echo "- Results base: $RESULTS_BASE"
echo ""

# First, run the multi-node detection test
echo "========================================="
echo "Testing Multi-Node Detection"
echo "========================================="
srun -n 4 python ${SFUNCTOR_DIR}/test_multinode_fix.py
echo ""

# Function to run MPI test
run_mpi_test() {
    local n_ranks=$1
    local test_name=$2
    local input_mode=$3  # "single" or "list"
    local profile=${4:-"test"}  # Default to "test" profile
    
    echo "----------------------------------------"
    echo "Running MPI test: $test_name"
    echo "Number of MPI ranks: $n_ranks"
    echo "Input mode: $input_mode"
    echo "Profile: $profile"
    echo "----------------------------------------"
    
    # Create subdirectory for this test
    test_dir="${RESULTS_BASE}/${test_name}"
    mkdir -p $test_dir
    cd $test_dir
    
    # Run the analysis with timing
    start_time=$(date +%s)
    
    if [ "$input_mode" == "single" ]; then
        srun -n $n_ranks python ${SFUNCTOR_DIR}/run_analysis.py \
            --config $CONFIG_FILE \
            --profile $profile \
            --file_name $TEST_SLICE \
            2>&1 | tee run_${test_name}.log
    else
        # Create a limited slice list for testing (first 8 slices)
        head -8 $SLICE_LIST > test_slice_list.txt
        echo "Using $(wc -l < test_slice_list.txt) slices for test"
        
        srun -n $n_ranks python ${SFUNCTOR_DIR}/run_analysis.py \
            --config $CONFIG_FILE \
            --profile $profile \
            --slice_list test_slice_list.txt \
            2>&1 | tee run_${test_name}.log
    fi
    
    exit_code=$?
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    
    echo ""
    echo "Test completed in $elapsed seconds"
    echo "Exit code: $exit_code"
    
    # Check if output file was created
    if ls sf_results_*.npz 1> /dev/null 2>&1; then
        echo "Output file(s) created successfully:"
        ls -lh sf_results_*.npz
    else
        echo "ERROR: No output file found!"
    fi
    
    cd $RESULTS_BASE
    echo ""
    
    return $exit_code
}

# Test 1: Single slice with 2 ranks (1 per node) - use cluster profile
run_mpi_test 2 "mpi_2ranks_single" "single" "cluster"

# Test 2: Single slice with 8 ranks (4 per node) - use cluster profile
run_mpi_test 8 "mpi_8ranks_single" "single" "cluster"

# Test 3: Single slice with 16 ranks (8 per node) - use test profile
run_mpi_test 16 "mpi_16ranks_single" "single" "test"

# Test 4: Single slice with 64 ranks (32 per node, full utilization) - use cluster profile
run_mpi_test 64 "mpi_64ranks_single" "single" "cluster"

# Test 5: Multiple slices with 8 ranks - use cluster profile
run_mpi_test 8 "mpi_8ranks_list" "list" "cluster"

# Test 6: Multiple slices with 64 ranks - use cluster profile
run_mpi_test 64 "mpi_64ranks_list" "list" "cluster"

# Test 7: Test with dev profile and 32 ranks
echo "=========================================="
echo "Testing with 'cluster' profile (optimized for multi-node)"
echo "=========================================="
test_dir="${RESULTS_BASE}/mpi_32ranks_cluster"
mkdir -p $test_dir
cd $test_dir

srun -n 32 python ${SFUNCTOR_DIR}/run_analysis.py \
    --config $CONFIG_FILE \
    --profile cluster \
    --file_name $TEST_SLICE \
    2>&1 | tee run_cluster_profile.log

cd $RESULTS_BASE

# Test 8: Test multiprocessing override warning
echo "=========================================="
echo "Testing multiprocessing override (should show warning)"
echo "=========================================="
test_dir="${RESULTS_BASE}/mpi_override_test"
mkdir -p $test_dir
cd $test_dir

# Use 4 MPI ranks with explicit n_processes=8 (should be overridden to 1)
srun -n 4 python ${SFUNCTOR_DIR}/run_analysis.py \
    --config $CONFIG_FILE \
    --profile test \
    --file_name $TEST_SLICE \
    --n_processes 8 \
    2>&1 | tee run_override_test.log

echo ""
echo "Checking for multiprocessing override warning:"
grep -i "WARNING: Multi-node MPI detected" run_override_test.log || echo "No warning found - might be single node?"

cd $RESULTS_BASE

# Summary
echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "Results saved in: $RESULTS_BASE"
echo ""
echo "MPI Information:"
echo "- SLURM_JOB_ID: $SLURM_JOB_ID"
echo "- SLURM_JOB_NUM_NODES: $SLURM_JOB_NUM_NODES"
echo "- SLURM_NTASKS: $SLURM_NTASKS"
echo ""

# Display timing summary
echo "Timing summary:"
for dir in ${RESULTS_BASE}/*/; do
    if [ -f "${dir}run_*.log" ]; then
        test_name=$(basename "$dir")
        elapsed=$(grep "Test completed in" "${dir}run_*.log" | awk '{print $4}')
        if [ ! -z "$elapsed" ]; then
            echo "  $test_name: ${elapsed}s"
        fi
    fi
done

echo ""
echo "To verify results consistency:"
echo "- Check that all tests produced output files"
echo "- Use verify_results.py to compare numerical results"
echo "- Look for any MPI-related errors in the log files"
echo ""
echo "Multi-node specific checks:"
echo "- Verify multi-node detection worked (check logs for node distribution)"
echo "- Confirm multiprocessing was disabled when running across nodes"
echo "- Check for 'WARNING: Multi-node MPI detected' messages in logs"