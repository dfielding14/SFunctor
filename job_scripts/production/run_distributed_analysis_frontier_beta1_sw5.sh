#!/bin/bash
#SBATCH -A AST207
#SBATCH -J sf_beta1_sw5
#SBATCH -o logs/sf_distributed_%j.out
#SBATCH -e logs/sf_distributed_%j.err
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 32
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56

# Distributed structure function analysis for beta=1 5120 simulation
# Using fixed shared memory implementation and stencil width 5

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

# Configuration for beta=1 5120 simulation
SIM_NAME="Turb_5120_beta1_dedt025_plm"
BASE_DIR="/lustre/orion/ast207/proj-shared/dfielding/Production_plm"

# Parameters as requested
N_DISP_TOTAL=10_000      # 10,000 displacements as requested
N_ELL_BINS=128           # 128 ell bins as requested
N_RANDOM_SUBSAMPLES=1_000  # 1,000 random subsamples as requested
STRIDE=1                 # Full resolution
STENCIL_WIDTH=5          # Stencil width 5 as requested
NRES=5120                # 5120 resolution
SEED=42

# Resume from previous run (optional)
RESUME_FROM_DIR="${RESUME_FROM_DIR:-}"

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
    SLICE_DIR="${BASE_DIR}/2d_slices_${SIM_NAME}"
    ls $SLICE_DIR/*.npz > $SLICE_LIST 2>/dev/null || {
        echo "ERROR: No slices found in $SLICE_DIR"
        exit 1
    }
    echo "Found $(wc -l < $SLICE_LIST) slices"
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
echo "  N_DISP_TOTAL: $N_DISP_TOTAL"
echo "  N_ELL_BINS: $N_ELL_BINS"
echo "  N_RANDOM_SUBSAMPLES: $N_RANDOM_SUBSAMPLES"
echo "  STRIDE: $STRIDE"
echo "  STENCIL_WIDTH: $STENCIL_WIDTH (5-point stencil as requested)"
echo "  NRES: $NRES"
echo "  Nodes: $SLURM_JOB_NUM_NODES"
echo "  CPUs per node: 56 (using fixed shared memory)"
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
            --n_processes 14 \
            > $LOG_FILE 2>&1 &
    done

    # Wait for all nodes to complete
    echo "Waiting for all nodes to complete..."
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
        echo "Combining results from all nodes..."
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

python $SFUNCTOR_DIR/plotting_scripts/plot_structure_functions.py \
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
echo "  Beta: 1"
echo "  Resolution: 5120"
echo "  Stencil Width: 5"
echo "  Displacements: 10,000"
echo "  Random Subsamples: 1,000"
echo "  Ell Bins: 128"
echo ""
echo "Note: Using fixed shared memory implementation to prevent OOM errors"
