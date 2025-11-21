#!/bin/bash
#SBATCH -A AST207
#SBATCH -J sf_beta25_10240_sw2
#SBATCH -o sf_distributed_%j.out
#SBATCH -e sf_distributed_%j.err
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 64
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56

# Distributed structure function analysis for beta=25 10240 simulation
# Using fixed shared memory implementation and stencil width 2
# Starting with 10x less displacements/subsamples for initial test

echo "======================================"
echo "Starting Distributed SF Analysis"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "======================================"

# Load necessary modules for Frontier
module reset
module load PrgEnv-gnu

# Activate virtual environment with proper MPI
source /autofs/nccs-svm1_home2/dfielding/SFunctor/venv_frontier/bin/activate

# Configuration for beta=25 10240 simulation
SIM_NAME="Turb_10240_beta25_dedt025_plm"
BASE_DIR="/lustre/orion/ast207/proj-shared/dfielding/Production_plm"

# Parameters - starting with 10x less for initial test
# Target: 1,000,000 displacements, 100,000 subsamples
# Initial test: 100,000 displacements, 10,000 subsamples
N_DISP_TOTAL=100_000      # 10x less than target 1,000,000
N_ELL_BINS=128            # 128 ell bins as requested
N_RANDOM_SUBSAMPLES=10_000  # 10x less than target 100,000
STRIDE=1                  # Full resolution
STENCIL_WIDTH=2           # Stencil width 2 as requested
NRES=10240                # 10240 resolution
SEED=42

# Resume from previous run (optional)
# If RESUME_FROM_DIR not set, check for job 3659610's results
if [ -z "$RESUME_FROM_DIR" ]; then
    PREV_JOB_DIR="${BASE_DIR}/sfunctor_results/results_${SIM_NAME}/ndisp100_000_nrand10_000_nell128_sw2_job3659610"
    if [ -d "$PREV_JOB_DIR" ]; then
        echo "Found previous job 3659610 results, will resume from there"
        RESUME_FROM_DIR="$PREV_JOB_DIR"
    fi
fi

# Set paths
SFUNCTOR_DIR="/autofs/nccs-svm1_home2/dfielding/SFunctor"
export PYTHONPATH="${SFUNCTOR_DIR}:${PYTHONPATH}"
RUN_NAME="ndisp${N_DISP_TOTAL}_nrand${N_RANDOM_SUBSAMPLES}_nell${N_ELL_BINS}_sw${STENCIL_WIDTH}_job${SLURM_JOB_ID}"
WORK_DIR="${BASE_DIR}/sfunctor_results/results_${SIM_NAME}/${RUN_NAME}"
SLICE_LIST="${BASE_DIR}/sfunctor_results/slice_list_${SIM_NAME}.txt"

# Create working directory
mkdir -p $WORK_DIR
cd $WORK_DIR

# Check for resume mode and copy existing results
if [ -n "$RESUME_FROM_DIR" ] && [ -d "$RESUME_FROM_DIR" ]; then
    echo ""
    echo "======================================"
    echo "Resume Mode: Checking for existing results"
    echo "======================================"
    echo "Resume from: $RESUME_FROM_DIR"
    echo ""

    # Copy existing slice results
    EXISTING_RESULTS=($(ls ${RESUME_FROM_DIR}/sf_results_Turb*.npz 2>/dev/null || true))
    if [ ${#EXISTING_RESULTS[@]} -gt 0 ]; then
        echo "Found ${#EXISTING_RESULTS[@]} existing slice results. Copying..."
        for result in "${EXISTING_RESULTS[@]}"; do
            cp -v "$result" "$WORK_DIR/"
        done
        echo ""
    else
        echo "No existing slice results found to copy."
    fi

    # Copy existing histograms
    if [ -d "${RESUME_FROM_DIR}/histograms" ]; then
        echo "Copying existing histogram files..."
        cp -r "${RESUME_FROM_DIR}/histograms" "$WORK_DIR/"
        echo ""
    fi
fi

# Generate slice list if it doesn't exist
if [ ! -f "$SLICE_LIST" ]; then
    echo "Generating slice list..."
    # Try the standard slice directory first
    SLICE_DIR="${BASE_DIR}/slice_${SIM_NAME}"
    if [ ! -d "$SLICE_DIR" ]; then
        # Try alternative naming
        SLICE_DIR="${BASE_DIR}/sfunctor_results/slice_${SIM_NAME}"
    fi

    if [ -d "$SLICE_DIR" ]; then
        ls $SLICE_DIR/*.npz > $SLICE_LIST 2>/dev/null || {
            echo "ERROR: No slices found in $SLICE_DIR"
            exit 1
        }
        echo "Found $(wc -l < $SLICE_LIST) slices"
    else
        echo "ERROR: Slice directory not found: $SLICE_DIR"
        echo "Please check the path or set SLICE_LIST environment variable"
        exit 1
    fi
else
    echo "Using existing slice list: $SLICE_LIST"
    echo "Total slices: $(wc -l < $SLICE_LIST)"
fi

# Check which slices we've already processed
PROCESSED_SLICES=()
if [ -d "$WORK_DIR" ]; then
    for result in $WORK_DIR/sf_results_*.npz; do
        if [ -f "$result" ]; then
            # Extract slice name from result filename
            basename=$(basename $result)
            slice_id="${basename#sf_results_}"
            slice_id="${slice_id%.npz}"
            PROCESSED_SLICES+=("$slice_id")
        fi
    done
fi

echo ""
echo "======================================"
echo "Analysis Status"
echo "======================================"
echo "Already processed: ${#PROCESSED_SLICES[@]} slices"
echo "Configuration:"
echo "  SIM_NAME: $SIM_NAME"
echo "  Resolution: 10240×10240 (FULL)"
echo "  N_DISP_TOTAL: $N_DISP_TOTAL (10x less than target)"
echo "  N_ELL_BINS: $N_ELL_BINS"
echo "  N_RANDOM_SUBSAMPLES: $N_RANDOM_SUBSAMPLES (10x less than target)"
echo "  STRIDE: $STRIDE"
echo "  STENCIL_WIDTH: $STENCIL_WIDTH (2-point stencil as requested)"
echo "  NRES: $NRES"
echo "  Nodes: $SLURM_JOB_NUM_NODES (64 nodes for 10240 data)"
echo "  CPUs per node: 56 (using ALL cores with shared memory fix)"
echo ""
echo "Note: Starting with 10x less displacements/subsamples for initial test"
echo "Target for production: 1,000,000 displacements, 100,000 subsamples"
echo ""

# Generate displacements
echo "======================================"
echo "Generating Displacements"
echo "======================================"
python $SFUNCTOR_DIR/scripts/production/generate_displacements.py \
    --n_disp_total $N_DISP_TOTAL \
    --n_ell_bins $N_ELL_BINS \
    --stencil_width $STENCIL_WIDTH \
    --Nres $NRES \
    --seed $SEED

echo ""
echo "======================================"
echo "Processing Slices"
echo "======================================"

# Get list of nodes
NODES=($(scontrol show hostname $SLURM_JOB_NODELIST))
TOTAL_NODES=${#NODES[@]}

# Process slices
SLICE_INDEX=0
mapfile -t ALL_SLICES < $SLICE_LIST

for SLICE_PATH in "${ALL_SLICES[@]}"; do
    SLICE_NAME=$(basename $SLICE_PATH .npz)

    # Skip if already processed
    if [[ " ${PROCESSED_SLICES[@]} " =~ " ${SLICE_NAME} " ]]; then
        echo "Skipping $SLICE_NAME (already processed)"
        continue
    fi

    echo ""
    echo "Processing: $SLICE_NAME"
    echo "------------------------------------"

    # Create output directory for this slice
    SLICE_OUTPUT_DIR="$WORK_DIR/${SLICE_NAME}_histograms"
    mkdir -p $SLICE_OUTPUT_DIR

    # Launch analysis on all nodes
    for NODE_ID in $(seq 0 $((TOTAL_NODES - 1))); do
        NODE=${NODES[$NODE_ID]}

        LOG_FILE="$SLICE_OUTPUT_DIR/node_${NODE_ID}.log"

        # Using the fixed scripts/production/run_node_analysis.py with shared memory implementation
        # Using ALL 56 processes per node - shared memory fix prevents duplication!
        srun --exclusive -N 1 -n 1 -w $NODE \
            --cpus-per-task=56 \
            python $SFUNCTOR_DIR/scripts/production/run_node_analysis.py \
            --slice $SLICE_PATH \
            --displacements $WORK_DIR/displacements.npz \
            --node_id $NODE_ID \
            --total_nodes $TOTAL_NODES \
            --output_dir $SLICE_OUTPUT_DIR \
            --stride $STRIDE \
            --N_random_subsamples $N_RANDOM_SUBSAMPLES \
            --stencil_width $STENCIL_WIDTH \
            --n_processes 56 \
            > $LOG_FILE 2>&1 &
    done

    # Wait for all nodes to complete
    echo "Waiting for all 64 nodes to complete..."
    wait

    # Check if any log files show errors
    ERROR_COUNT=0
    for logfile in ${SLICE_OUTPUT_DIR}/node_*.log; do
        if grep -q "Error\|error\|Traceback\|Failed" "$logfile" 2>/dev/null; then
            echo "ERROR found in $logfile"
            tail -n 20 "$logfile"
            ((ERROR_COUNT++))
        fi
    done

    if [ $ERROR_COUNT -gt 0 ]; then
        echo "WARNING: $ERROR_COUNT nodes reported errors for $SLICE_NAME"
        echo "Check log files in $SLICE_OUTPUT_DIR for details"
        # Continue processing other slices even if this one had errors
    else
        echo "All nodes completed successfully for $SLICE_NAME"

        # Combine results from all nodes
        echo "Combining results from all 64 nodes..."
        python $SFUNCTOR_DIR/scripts/production/combine_histograms.py \
            --pattern "${SLICE_OUTPUT_DIR}/histogram_*.npz" \
            --output $WORK_DIR/sf_results_${SLICE_NAME}.npz \
            --mode node

        # Clean up node files to save space (optional)
        if [ -f "$WORK_DIR/sf_results_${SLICE_NAME}.npz" ]; then
            echo "Cleaning up intermediate files..."
            rm -rf $SLICE_OUTPUT_DIR
        fi
    fi

    echo "Completed: $SLICE_NAME"
done

echo ""
echo "======================================"
echo "Combining All Results"
echo "======================================"

# Combine all slice results
python $SFUNCTOR_DIR/scripts/production/combine_histograms.py \
    --pattern "$WORK_DIR/sf_results_*.npz" \
    --output $WORK_DIR/sf_results_all_slices.npz \
    --mode slice

# Generate plots
echo ""
echo "======================================"
echo "Generating Plots"
echo "======================================"

python $SFUNCTOR_DIR/scripts/production/plot_structure_functions.py \
    $WORK_DIR/sf_results_all_slices.npz

echo ""
echo "======================================"
echo "Analysis Complete"
echo "======================================"
echo "End time: $(date)"
echo "Results saved in: $WORK_DIR"
echo ""
echo "Key outputs:"
echo "  - sf_results_all_slices.npz (combined data)"
echo "  - sf_results_all_slices_mean_structure_functions.png"
echo "  - sf_results_all_slices_2d_histograms_normalized.png"
echo "  - sf_results_all_slices_angular_structure_functions.png"
echo "  - sf_results_all_slices_cross_product_ratios.png"
echo "  - Individual channel plots (11 files)"
echo ""
echo "Configuration used:"
echo "  Beta: 25"
echo "  Resolution: 10240×10240"
echo "  Stencil Width: 2"
echo "  Displacements: 100,000 (initial test, target: 1,000,000)"
echo "  Random Subsamples: 10,000 (initial test, target: 100,000)"
echo "  Ell Bins: 128"
echo "  Nodes: 64"
echo ""
echo "Note: This was an initial test with 10x less displacements/subsamples"
echo "For production, increase to:"
echo "  N_DISP_TOTAL=1_000_000"
echo "  N_RANDOM_SUBSAMPLES=100_000"
