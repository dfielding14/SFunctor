#!/bin/bash
#SBATCH -A ast207
#SBATCH -J sf_test_640_v2
#SBATCH -o logs/sf_%x_%j.out
#SBATCH -e logs/sf_%x_%j.err
#SBATCH --time=0:05:00
#SBATCH -p batch
##SBATCH -q debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56

#============================================================================
# TEST AUTO-CHAINING V2 - SIMPLE TIMER APPROACH
#============================================================================

# Configuration
SIM_TYPE="Turb"
RESOLUTION="640"
BETA="25"
DEDT="025"
SCHEME="plm"
STENCIL_WIDTH=2
N_DISP_TOTAL=12500
N_RANDOM_SUBSAMPLES=12500
N_ELL_BINS=128
STRIDE=1
SEED=42

# Auto-chain settings
TIME_BUFFER_SECONDS=90  # Kill job 90 seconds before wall time
MAX_CHAIN_JOBS=3
IS_CONTINUATION="${IS_CONTINUATION:-0}"
CHAIN_COUNT="${CHAIN_COUNT:-1}"
ORIGINAL_JOB_ID="${ORIGINAL_JOB_ID:-$SLURM_JOB_ID}"
RESUME_FROM="${RESUME_FROM:-}"

# Work directory setup
BASE_DIR="/lustre/orion/ast207/proj-shared/dfielding/Production_plm"
SFUNCTOR_DIR="/autofs/nccs-svm1_home2/dfielding/SFunctor"
SIM_NAME="${SIM_TYPE}_${RESOLUTION}_beta${BETA}_dedt${DEDT}_${SCHEME}"
RUN_NAME="ndisp${N_DISP_TOTAL}_nrand${N_RANDOM_SUBSAMPLES}_nell${N_ELL_BINS}_sw${STENCIL_WIDTH}"

if [ "$IS_CONTINUATION" == "1" ] && [ -n "$RESUME_FROM" ]; then
    WORK_DIR="$RESUME_FROM"
else
    WORK_DIR="${BASE_DIR}/sfunctor_results/results_${SIM_NAME}/${RUN_NAME}_job${ORIGINAL_JOB_ID}_test_v2"
    mkdir -p $WORK_DIR
fi

cd $WORK_DIR

#============================================================================
# SIMPLE AUTO-CHAIN TIMER
#============================================================================

# Function to submit continuation and exit
submit_continuation_and_exit() {
    echo ""
    echo "============================================================================"
    echo " AUTO-CHAIN: Time limit approaching, submitting continuation job"
    echo "============================================================================"

    # Check if we should continue
    if [ "$CHAIN_COUNT" -ge "$MAX_CHAIN_JOBS" ]; then
        echo " Maximum chain count reached ($MAX_CHAIN_JOBS) - not continuing"
        exit 0
    fi

    # Check if all work is done
    TOTAL_SLICES=$(wc -l < ${BASE_DIR}/sfunctor_results/slice_list_${SIM_NAME}.txt 2>/dev/null || echo 0)
    COMPLETED_SLICES=$(ls -1 $WORK_DIR/sf_results_*.npz 2>/dev/null | wc -l)

    if [ "$COMPLETED_SLICES" -ge "$TOTAL_SLICES" ] && [ "$TOTAL_SLICES" -gt 0 ]; then
        echo " All slices completed - no continuation needed"
        exit 0
    fi

    # Submit continuation job
    echo " Submitting continuation job (chain $((CHAIN_COUNT + 1))/$MAX_CHAIN_JOBS)"

    NEXT_JOB=$(sbatch \
        --export=ALL,IS_CONTINUATION=1,CHAIN_COUNT=$((CHAIN_COUNT + 1)),ORIGINAL_JOB_ID=$ORIGINAL_JOB_ID,RESUME_FROM=$WORK_DIR \
        $0 2>&1)

    if [ $? -eq 0 ]; then
        NEXT_JOB_ID=$(echo $NEXT_JOB | awk '{print $4}')
        echo " Successfully submitted continuation job: $NEXT_JOB_ID"
        echo " Work directory: $WORK_DIR"

        # Save info
        echo "Continuation job: $NEXT_JOB_ID (chain $((CHAIN_COUNT + 1)))" >> ${WORK_DIR}/chain_history.txt
    else
        echo " ERROR: Failed to submit continuation: $NEXT_JOB"
    fi

    # Kill all child processes and exit
    echo " Stopping current job for continuation..."
    kill -TERM -$$  # Kill process group
    exit 0
}

# Start the timer in background (only if not already near time limit)
if [ -n "$SLURM_JOB_ID" ]; then
    # Calculate sleep duration (5 min job - 90 sec buffer = 210 seconds)
    SLEEP_DURATION=$((5 * 60 - TIME_BUFFER_SECONDS))

    if [ $SLEEP_DURATION -gt 0 ]; then
        echo " Auto-chain timer set: will trigger in $SLEEP_DURATION seconds"
        (sleep $SLEEP_DURATION && submit_continuation_and_exit) &
        TIMER_PID=$!
        echo " Timer PID: $TIMER_PID"
    fi
fi

#============================================================================
# MAIN PROCESSING
#============================================================================

echo "============================================================================"
echo " STRUCTURE FUNCTION ANALYSIS TEST V2 - 640 RESOLUTION"
if [ "$IS_CONTINUATION" == "1" ]; then
    echo " CONTINUATION JOB - Chain position: $CHAIN_COUNT / $MAX_CHAIN_JOBS"
fi
echo "============================================================================"
echo " Job ID: $SLURM_JOB_ID"
echo " Start Time: $(date)"
echo " Nodes: $SLURM_JOB_NUM_NODES"
echo " Chain: $CHAIN_COUNT / $MAX_CHAIN_JOBS"
echo " Work Dir: $WORK_DIR"
echo "============================================================================"

# Load modules
module reset
module load PrgEnv-gnu

# Activate virtual environment
source /autofs/nccs-svm1_home2/dfielding/SFunctor/venv_frontier/bin/activate

# Generate slice list
SLICE_LIST="${BASE_DIR}/sfunctor_results/slice_list_${SIM_NAME}.txt"
if [ ! -f "$SLICE_LIST" ]; then
    SLICE_DIR="${BASE_DIR}/sfunctor_results/slice_${SIM_NAME}"
    [ ! -d "$SLICE_DIR" ] && SLICE_DIR="${BASE_DIR}/slice_${SIM_NAME}"
    ls $SLICE_DIR/*.npz > $SLICE_LIST 2>/dev/null || exit 1
fi

TOTAL_SLICES=$(wc -l < $SLICE_LIST)
echo " Total slices: $TOTAL_SLICES"

# Generate displacements
if [ ! -f "$WORK_DIR/displacements.npz" ]; then
    echo " Generating displacements..."
    python $SFUNCTOR_DIR/scripts/production/generate_displacements.py \
        --n_disp_total $N_DISP_TOTAL \
        --n_ell_bins $N_ELL_BINS \
        --Nres $RESOLUTION \
        --seed $SEED
fi

# Process slices
echo ""
echo " Processing Slices"
echo "============================================================================"

# Check already processed
PROCESSED_SLICES=()
for result in $WORK_DIR/sf_results_*.npz; do
    [ -f "$result" ] && PROCESSED_SLICES+=("$(basename $result .npz | sed 's/sf_results_//')")
done

echo " Already processed: ${#PROCESSED_SLICES[@]} / $TOTAL_SLICES"

# Get nodes
NODES=($(scontrol show hostname $SLURM_JOB_NODELIST))
TOTAL_NODES=${#NODES[@]}

# Process each slice
mapfile -t ALL_SLICES < $SLICE_LIST
SLICE_COUNT=0

for SLICE_PATH in "${ALL_SLICES[@]}"; do
    SLICE_NAME=$(basename $SLICE_PATH .npz)

    # Skip if already processed
    if [[ " ${PROCESSED_SLICES[@]} " =~ " ${SLICE_NAME} " ]]; then
        continue
    fi

    SLICE_COUNT=$((SLICE_COUNT + 1))
    echo ""
    echo " [$SLICE_COUNT] Processing: $SLICE_NAME"
    echo " Time: $(date +%H:%M:%S)"

    # Create output directory
    SLICE_OUTPUT_DIR="$WORK_DIR/${SLICE_NAME}_histograms"
    mkdir -p $SLICE_OUTPUT_DIR

    # Launch on all nodes
    for NODE_ID in $(seq 0 $((TOTAL_NODES - 1))); do
        NODE=${NODES[$NODE_ID]}

        srun --exclusive -N 1 -n 1 -w $NODE --cpus-per-task=56 \
            python $SFUNCTOR_DIR/scripts/production/run_node_analysis.py \
            --slice $SLICE_PATH \
            --displacements $WORK_DIR/displacements.npz \
            --output_dir $SLICE_OUTPUT_DIR \
            --stride $STRIDE \
            --N_random_subsamples $N_RANDOM_SUBSAMPLES \
            --stencil_width $STENCIL_WIDTH \
            --n_processes 56 \
            --node_id $NODE_ID \
            --total_nodes $TOTAL_NODES \
            > $SLICE_OUTPUT_DIR/node_${NODE_ID}.log 2>&1 &
    done

    wait
    echo " Completed: $SLICE_NAME"
done

# Cancel timer if we finished
if [ -n "$TIMER_PID" ]; then
    kill $TIMER_PID 2>/dev/null && echo " Cancelled auto-chain timer (work complete)"
fi

# Merge histograms
echo ""
echo " Merging histograms..."

for dir in $WORK_DIR/*_histograms; do
    [ -d "$dir" ] || continue
    SLICE_NAME=$(basename "$dir" "_histograms")

    if [ ! -f "$WORK_DIR/sf_results_${SLICE_NAME}.npz" ]; then
        python $SFUNCTOR_DIR/scripts/production/combine_histograms_fast.py \
            --pattern "${dir}/histogram_*.npz" \
            --output "$WORK_DIR/sf_results_${SLICE_NAME}.npz" \
            --mode node \
            --workers 32
        [ -f "$WORK_DIR/sf_results_${SLICE_NAME}.npz" ] && rm -rf "$dir"
    fi
done

# Final status
COMPLETED=$(ls -1 $WORK_DIR/sf_results_*.npz 2>/dev/null | wc -l)
echo ""
echo "============================================================================"
echo " STATUS: Processed $COMPLETED / $TOTAL_SLICES slices"
echo " Chain: $CHAIN_COUNT / $MAX_CHAIN_JOBS"
echo "============================================================================"

if [ $COMPLETED -eq $TOTAL_SLICES ]; then
    echo " ALL SLICES COMPLETE - Success!"
else
    echo " Partial completion - may need continuation"
fi