#!/bin/bash
#SBATCH -A ast207
#SBATCH -J sf_test_640
#SBATCH -o sf_%x_%j.out
#SBATCH -e sf_%x_%j.err
#SBATCH --time=0:05:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56

#============================================================================
# TEST AUTO-CHAINING WITH SMALL 640 RESOLUTION JOB
#============================================================================
# This script tests the auto-chaining functionality with a small, quick job
# Should complete some work in 5 minutes then auto-submit continuation
#============================================================================

#============================================================================
# USER CONFIGURATION SECTION
#============================================================================

# Simulation Selection
SIM_TYPE="Turb"
RESOLUTION="640"
BETA="25"
DEDT="025"
SCHEME="plm"

# Analysis Parameters
STENCIL_WIDTH=2
N_DISP_TOTAL=12500
N_RANDOM_SUBSAMPLES=12500
N_ELL_BINS=128
STRIDE=1
SEED=42

# Custom bin edges (leave empty for defaults)
LOG_DELTA_BIN_EDGES_MIN=""
LOG_DELTA_BIN_EDGES_MAX=""
N_DELTA_BIN_EDGES=""

# HPC Configuration
N_NODES=2
TIME_LIMIT="0:05:00"
QUEUE="debug"

# Auto-continuation Configuration
AUTO_CONTINUE=1
TIME_BUFFER_MINUTES=1  # 1 minute buffer for testing
MAX_CHAIN_JOBS=3        # Limit chains for testing

# Resume Configuration
RESUME_FROM=""

# Chain Job Configuration (DO NOT MODIFY - used internally)
IS_CONTINUATION="${IS_CONTINUATION:-0}"
CHAIN_COUNT="${CHAIN_COUNT:-1}"
ORIGINAL_JOB_ID="${ORIGINAL_JOB_ID:-$SLURM_JOB_ID}"

#============================================================================
# HELPER FUNCTIONS
#============================================================================

function get_time_remaining_seconds() {
    # Get remaining time in seconds from SLURM
    if [ -n "$SLURM_JOB_ID" ]; then
        # Get time limit and elapsed time
        TIME_LIMIT_SEC=$(squeue -h -j $SLURM_JOB_ID -o "%l" | awk -F: '{ 
            if (NF == 3) print ($1 * 3600) + ($2 * 60) + $3; 
            else if (NF == 2) print ($1 * 60) + $2;
            else print $1 
        }')
        
        ELAPSED_SEC=$SECONDS
        REMAINING_SEC=$((TIME_LIMIT_SEC - ELAPSED_SEC))
        echo $REMAINING_SEC
    else
        echo 999999  # Return large number if not in SLURM
    fi
}

function should_checkpoint() {
    # Check if we should checkpoint and submit continuation
    if [ "$AUTO_CONTINUE" != "1" ]; then
        return 1  # Auto-continue disabled
    fi
    
    if [ "$CHAIN_COUNT" -ge "$MAX_CHAIN_JOBS" ]; then
        echo " Maximum chain count ($MAX_CHAIN_JOBS) reached - will not auto-continue"
        return 1
    fi
    
    REMAINING_SEC=$(get_time_remaining_seconds)
    BUFFER_SEC=$((TIME_BUFFER_MINUTES * 60))
    
    if [ $REMAINING_SEC -lt $BUFFER_SEC ]; then
        return 0  # Should checkpoint
    else
        return 1  # Still have time
    fi
}

function submit_continuation_job() {
    # Submit a continuation job
    local WORK_DIR=$1
    
    echo ""
    echo "============================================================================"
    echo " SUBMITTING CONTINUATION JOB"
    echo "============================================================================"
    echo " Time remaining is less than ${TIME_BUFFER_MINUTES} minutes"
    echo " Current chain position: $CHAIN_COUNT / $MAX_CHAIN_JOBS"
    echo " Work directory: $WORK_DIR"
    
    # Submit continuation job
    NEXT_JOB=$(sbatch \
        --dependency=afterany:$SLURM_JOB_ID \
        --export=ALL,IS_CONTINUATION=1,CHAIN_COUNT=$((CHAIN_COUNT + 1)),ORIGINAL_JOB_ID=$ORIGINAL_JOB_ID,RESUME_FROM=$WORK_DIR \
        $0)
    
    if [ $? -eq 0 ]; then
        NEXT_JOB_ID=$(echo $NEXT_JOB | awk '{print $4}')
        echo " Successfully submitted continuation job: $NEXT_JOB_ID"
        echo " Chain position: $((CHAIN_COUNT + 1)) / $MAX_CHAIN_JOBS"
        
        # Save continuation info
        cat > ${WORK_DIR}/continuation_info.txt << EOF
Continuation job submitted: $NEXT_JOB_ID
Chain position: $((CHAIN_COUNT + 1)) / $MAX_CHAIN_JOBS
Submitted at: $(date)
Previous job: $SLURM_JOB_ID
Original job: $ORIGINAL_JOB_ID
Work directory: $WORK_DIR
EOF
        
        return 0
    else
        echo " ERROR: Failed to submit continuation job"
        return 1
    fi
}

function check_and_checkpoint() {
    # Check if we need to checkpoint and do it if necessary
    local WORK_DIR=$1
    
    if should_checkpoint; then
        echo ""
        echo " CHECKPOINT: Approaching time limit ($(get_time_remaining_seconds) seconds remaining)"
        
        # Submit continuation job
        submit_continuation_job "$WORK_DIR"
        
        # Clean exit
        echo " Exiting for continuation..."
        echo "============================================================================"
        exit 0
    fi
}

#============================================================================
# MAIN SCRIPT
#============================================================================

# Construct simulation name
SIM_NAME="${SIM_TYPE}_${RESOLUTION}_beta${BETA}_dedt${DEDT}_${SCHEME}"

# Create descriptive job name
if [ "$IS_CONTINUATION" == "1" ]; then
    JOB_NAME="sf_cont${CHAIN_COUNT}_640"
else
    JOB_NAME="sf_test_640"
fi

echo "============================================================================"
echo " STRUCTURE FUNCTION ANALYSIS TEST - 640 RESOLUTION"
if [ "$IS_CONTINUATION" == "1" ]; then
    echo " CONTINUATION JOB - Chain position: $CHAIN_COUNT / $MAX_CHAIN_JOBS"
fi
echo "============================================================================"
echo " Job Configuration:"
echo "   Job ID:        $SLURM_JOB_ID"
echo "   Job Name:      $JOB_NAME"
echo "   Start Time:    $(date)"
echo "   Nodes:         $SLURM_JOB_NUM_NODES"
echo "   Time Limit:    5 minutes"
echo "   Time Buffer:   1 minute"
if [ "$IS_CONTINUATION" == "1" ]; then
    echo "   Original Job:  $ORIGINAL_JOB_ID"
    echo "   Chain Count:   $CHAIN_COUNT"
    echo "   Resuming from: $RESUME_FROM"
fi
echo "============================================================================"

# Load modules
module reset
module load PrgEnv-gnu

# Activate virtual environment
source /autofs/nccs-svm1_home2/dfielding/SFunctor/venv_frontier/bin/activate

# Set paths
BASE_DIR="/lustre/orion/ast207/proj-shared/dfielding/Production_plm"
SFUNCTOR_DIR="/autofs/nccs-svm1_home2/dfielding/SFunctor"
RUN_NAME="ndisp${N_DISP_TOTAL}_nrand${N_RANDOM_SUBSAMPLES}_nell${N_ELL_BINS}_sw${STENCIL_WIDTH}"

# Determine work directory
if [ "$IS_CONTINUATION" == "1" ] && [ -n "$RESUME_FROM" ]; then
    WORK_DIR="$RESUME_FROM"
    echo " Continuing in existing directory: $WORK_DIR"
else
    WORK_DIR="${BASE_DIR}/sfunctor_results/results_${SIM_NAME}/${RUN_NAME}_job${ORIGINAL_JOB_ID}_test"
    mkdir -p $WORK_DIR
fi

SLICE_LIST="${BASE_DIR}/sfunctor_results/slice_list_${SIM_NAME}.txt"

cd $WORK_DIR

# Save configuration
if [ ! -f config.txt ] || [ "$IS_CONTINUATION" != "1" ]; then
    cat > config.txt << EOF
# Test Configuration - 640 Resolution Auto-chain Test
# Job ID: $ORIGINAL_JOB_ID (Chain: $CHAIN_COUNT)
# Date: $(date)
SIM_NAME=$SIM_NAME
RESOLUTION=$RESOLUTION
BETA=$BETA
STENCIL_WIDTH=$STENCIL_WIDTH
N_DISP_TOTAL=$N_DISP_TOTAL
N_RANDOM_SUBSAMPLES=$N_RANDOM_SUBSAMPLES
N_ELL_BINS=$N_ELL_BINS
STRIDE=$STRIDE
N_NODES=$SLURM_JOB_NUM_NODES
AUTO_CONTINUE=$AUTO_CONTINUE
MAX_CHAIN_JOBS=$MAX_CHAIN_JOBS
TIME_BUFFER_MINUTES=$TIME_BUFFER_MINUTES
EOF
fi

#============================================================================
# GENERATE SLICE LIST
#============================================================================

if [ ! -f "$SLICE_LIST" ]; then
    echo " Generating slice list..."
    SLICE_DIR="${BASE_DIR}/sfunctor_results/slice_${SIM_NAME}"
    
    if [ ! -d "$SLICE_DIR" ]; then
        SLICE_DIR="${BASE_DIR}/slice_${SIM_NAME}"
    fi
    
    if [ -d "$SLICE_DIR" ]; then
        ls $SLICE_DIR/*.npz > $SLICE_LIST 2>/dev/null || {
            echo " ERROR: No slices found in $SLICE_DIR"
            exit 1
        }
    else
        echo " ERROR: Slice directory not found: $SLICE_DIR"
        exit 1
    fi
fi

TOTAL_SLICES=$(wc -l < $SLICE_LIST)
echo " Found $TOTAL_SLICES slices to process"

#============================================================================
# GENERATE DISPLACEMENTS
#============================================================================

if [ ! -f "$WORK_DIR/displacements.npz" ]; then
    echo ""
    echo " Generating Displacements"
    python $SFUNCTOR_DIR/scripts/production/generate_displacements.py \
        --n_disp_total $N_DISP_TOTAL \
        --n_ell_bins $N_ELL_BINS \
        --Nres $RESOLUTION \
        --seed $SEED
    
    if [ ! -f "$WORK_DIR/displacements.npz" ]; then
        echo " ERROR: Failed to generate displacements"
        exit 1
    fi
else
    echo " Using existing displacements.npz"
fi

#============================================================================
# PROCESS SLICES WITH TIME CHECKING
#============================================================================

echo ""
echo " Processing Slices"
echo "============================================================================"

# Check which slices already processed
PROCESSED_SLICES=()
for result in $WORK_DIR/sf_results_*.npz; do
    if [ -f "$result" ]; then
        basename=$(basename $result)
        slice_id="${basename#sf_results_}"
        slice_id="${slice_id%.npz}"
        PROCESSED_SLICES+=("$slice_id")
    fi
done

echo " Already processed: ${#PROCESSED_SLICES[@]} / $TOTAL_SLICES slices"

# Get list of nodes
NODES=($(scontrol show hostname $SLURM_JOB_NODELIST))
TOTAL_NODES=${#NODES[@]}

# Process slices
mapfile -t ALL_SLICES < $SLICE_LIST
SLICE_COUNT=0

# For testing, only process first 2 slices to ensure we trigger continuation
MAX_SLICES_PER_JOB=2
PROCESSED_THIS_JOB=0

for SLICE_PATH in "${ALL_SLICES[@]}"; do
    SLICE_NAME=$(basename $SLICE_PATH .npz)
    
    # Skip if already processed
    if [[ " ${PROCESSED_SLICES[@]} " =~ " ${SLICE_NAME} " ]]; then
        continue
    fi
    
    # Check if we should stop to test continuation
    if [ $PROCESSED_THIS_JOB -ge $MAX_SLICES_PER_JOB ]; then
        echo " Limiting to $MAX_SLICES_PER_JOB slices for test - forcing checkpoint"
        check_and_checkpoint "$WORK_DIR"
    fi
    
    # Check if we're running out of time
    check_and_checkpoint "$WORK_DIR"
    
    SLICE_COUNT=$((SLICE_COUNT + 1))
    PROCESSED_THIS_JOB=$((PROCESSED_THIS_JOB + 1))
    
    echo ""
    echo " [$SLICE_COUNT] Processing: $SLICE_NAME"
    echo " Time remaining: $(($(get_time_remaining_seconds) / 60)) minutes"
    
    # Create output directory for this slice
    SLICE_OUTPUT_DIR="$WORK_DIR/${SLICE_NAME}_histograms"
    mkdir -p $SLICE_OUTPUT_DIR
    
    # Build command
    CMD_BASE="python $SFUNCTOR_DIR/scripts/production/run_node_analysis.py \
        --slice $SLICE_PATH \
        --displacements $WORK_DIR/displacements.npz \
        --output_dir $SLICE_OUTPUT_DIR \
        --stride $STRIDE \
        --N_random_subsamples $N_RANDOM_SUBSAMPLES \
        --stencil_width $STENCIL_WIDTH \
        --n_processes 56"
    
    # Launch analysis on all nodes
    echo " Launching on $TOTAL_NODES nodes..."
    for NODE_ID in $(seq 0 $((TOTAL_NODES - 1))); do
        NODE=${NODES[$NODE_ID]}
        LOG_FILE="$SLICE_OUTPUT_DIR/node_${NODE_ID}.log"
        
        srun --exclusive -N 1 -n 1 -w $NODE \
            --cpus-per-task=56 \
            $CMD_BASE \
            --node_id $NODE_ID \
            --total_nodes $TOTAL_NODES \
            > $LOG_FILE 2>&1 &
    done
    
    # Wait for all nodes to complete
    wait
    
    echo " Slice $SLICE_NAME completed"
done

#============================================================================
# MERGE AND FINALIZE (if all done)
#============================================================================

# Check time before merging
check_and_checkpoint "$WORK_DIR"

echo ""
echo " Merging histograms..."

# Find and merge histogram directories
HIST_DIRS=($(find $WORK_DIR -type d -name "*_histograms" 2>/dev/null | sort))
for HIST_DIR in "${HIST_DIRS[@]}"; do
    SLICE_NAME=$(basename "$HIST_DIR" "_histograms")
    
    if [ ! -f "$WORK_DIR/sf_results_${SLICE_NAME}.npz" ]; then
        echo " Merging: $SLICE_NAME"
        python $SFUNCTOR_DIR/scripts/production/combine_histograms_fast.py \
            --pattern "${HIST_DIR}/histogram_*.npz" \
            --output "$WORK_DIR/sf_results_${SLICE_NAME}.npz" \
            --mode node \
            --workers 32
        
        if [ -f "$WORK_DIR/sf_results_${SLICE_NAME}.npz" ]; then
            rm -rf "$HIST_DIR"
        fi
    fi
done

# Final merge if all slices done
SLICE_RESULTS=($(ls -1 $WORK_DIR/sf_results_${SIM_TYPE}*.npz 2>/dev/null))
NUM_SLICE_RESULTS=${#SLICE_RESULTS[@]}

echo ""
echo "============================================================================"
echo " TEST STATUS"
echo "============================================================================"
echo " Processed: $NUM_SLICE_RESULTS / $TOTAL_SLICES slices"
echo " Time elapsed: $SECONDS seconds"
echo " Chain position: $CHAIN_COUNT / $MAX_CHAIN_JOBS"

if [ $NUM_SLICE_RESULTS -eq $TOTAL_SLICES ]; then
    echo " ALL SLICES COMPLETE - Test successful!"
    
    # Do final merge
    python $SFUNCTOR_DIR/scripts/production/combine_histograms_fast.py \
        --pattern "$WORK_DIR/sf_results_${SIM_TYPE}*.npz" \
        --output "$WORK_DIR/sf_results_all_slices.npz" \
        --mode slice \
        --workers 32
else
    echo " Test in progress - continuation may be needed"
fi

echo "============================================================================"
