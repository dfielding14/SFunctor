#!/bin/bash
#============================================================================
# CONFIGURABLE STRUCTURE FUNCTION ANALYSIS FOR FRONTIER HPC
# WITH AUTOMATIC JOB CHAINING FOR TIME LIMIT HANDLING
#============================================================================
# This script provides a unified, configurable interface for running
# structure function analysis on Frontier with intelligent parallel merging
# and automatic job continuation when approaching time limits
#
# Usage: Edit the configuration section below, then submit with:
#        sbatch run_sf_analysis_frontier_autochain.sh
#
# The script will automatically submit continuation jobs if needed
#
# Author: SF Analysis Pipeline
# Date: 2025
#============================================================================

#============================================================================
# USER CONFIGURATION SECTION - MODIFY THESE PARAMETERS
#============================================================================

# Simulation Selection
SIM_TYPE="Turb"                      # Simulation type (e.g., Turb)
RESOLUTION="10240"                    # Resolution (e.g., 5120, 10240)
BETA="25"                            # Beta value (e.g., 1, 6, 25, 100)
DEDT="025"                           # Energy injection rate
SCHEME="plm"                         # Numerical scheme

# Analysis Parameters
STENCIL_WIDTH=3                      # Stencil width (2, 3, or 5)
N_DISP_TOTAL=100000                  # Total number of displacements
N_RANDOM_SUBSAMPLES=100000          # Number of random subsamples
N_ELL_BINS=128                       # Number of ell bins
STRIDE=1                             # Stride for sampling (1 = full resolution)
SEED=42                              # Random seed for reproducibility

# Custom bin edges (optional - leave empty to use defaults)
LOG_SF_BIN_EDGES_MIN="-5 -5 -5 -5 -5 -5 -2 -2 -2 -2 -5"
LOG_SF_BIN_EDGES_MAX="1 1 1 1 1 1 4 4 4 4 3"

# HPC Configuration
N_NODES=64                           # Number of nodes to request
TIME_LIMIT="4:00:00"                # Wall time limit (HH:MM:SS)
QUEUE="debug"                        # Queue (debug, regular, premium)

# Auto-continuation Configuration
AUTO_CONTINUE=1                      # Enable automatic job continuation (1=yes, 0=no)
TIME_BUFFER_MINUTES=15               # Minutes before time limit to trigger continuation
MAX_CHAIN_JOBS=10                    # Maximum number of chained jobs to prevent runaway

# Resume Configuration (optional - set to previous job directory to resume)
RESUME_FROM=""                       # Leave empty for fresh run, or set to previous job dir

# Chain Job Configuration (DO NOT MODIFY - used internally for continuation)
IS_CONTINUATION="${IS_CONTINUATION:-0}"           # Internal flag for continuation jobs
CHAIN_COUNT="${CHAIN_COUNT:-1}"                   # Current chain position
ORIGINAL_JOB_ID="${ORIGINAL_JOB_ID:-$SLURM_JOB_ID}"  # Track original job

#============================================================================
# SLURM DIRECTIVES - These use the configuration above
#============================================================================

# Construct simulation name for job naming
SIM_NAME="${SIM_TYPE}_${RESOLUTION}_beta${BETA}_dedt${DEDT}_${SCHEME}"

# Create descriptive job name (keep under 64 chars for SLURM)
if [ "$IS_CONTINUATION" == "1" ]; then
    JOB_NAME="sf_cont${CHAIN_COUNT}_b${BETA}_${RESOLUTION}"
else
    JOB_NAME="sf_b${BETA}_${RESOLUTION}_sw${STENCIL_WIDTH}"
fi

#SBATCH -A AST207
#SBATCH -J sf_analysis
#SBATCH -o sf_%x_%j.out
#SBATCH -e sf_%x_%j.err
#SBATCH --time=4:00:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56

# Note: Edit the SBATCH directives above to match your configuration
# The values shown are defaults - adjust N_NODES, TIME_LIMIT, and QUEUE above

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
    
    # Create continuation script
    CONT_SCRIPT="${WORK_DIR}/continue_job.sh"
    
    cat > $CONT_SCRIPT << EOF
#!/bin/bash
# Auto-generated continuation script
export IS_CONTINUATION=1
export CHAIN_COUNT=$((CHAIN_COUNT + 1))
export ORIGINAL_JOB_ID=$ORIGINAL_JOB_ID
export RESUME_FROM="$WORK_DIR"

# Submit with same configuration
exec $0
EOF
    
    chmod +x $CONT_SCRIPT
    
    # Submit continuation job
    cd $(dirname $0)  # Make sure we're in the right directory
    
    NEXT_JOB=$(sbatch \
        --dependency=afterany:$SLURM_JOB_ID \
        --nodes=$N_NODES \
        --time=$TIME_LIMIT \
        --partition=batch \
        --qos=$QUEUE \
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
# SCRIPT BEGINS HERE - DO NOT MODIFY BELOW UNLESS YOU KNOW WHAT YOU'RE DOING
#============================================================================

echo "============================================================================"
echo " STRUCTURE FUNCTION ANALYSIS - FRONTIER HPC"
if [ "$IS_CONTINUATION" == "1" ]; then
    echo " CONTINUATION JOB - Chain position: $CHAIN_COUNT / $MAX_CHAIN_JOBS"
fi
echo "============================================================================"
echo " Job Configuration:"
echo "   Job ID:        $SLURM_JOB_ID"
echo "   Job Name:      $JOB_NAME"
echo "   Start Time:    $(date)"
echo "   Nodes:         $SLURM_JOB_NUM_NODES"
echo "   CPUs/Node:     56"
if [ "$IS_CONTINUATION" == "1" ]; then
    echo "   Original Job:  $ORIGINAL_JOB_ID"
    echo "   Chain Count:   $CHAIN_COUNT"
    echo "   Resuming from: $RESUME_FROM"
fi
echo "============================================================================"
echo " Simulation:"
echo "   Name:          $SIM_NAME"
echo "   Resolution:    ${RESOLUTION}×${RESOLUTION}"
echo "   Beta:          $BETA"
echo "============================================================================"
echo " Analysis Parameters:"
echo "   Stencil Width:     $STENCIL_WIDTH"
echo "   Displacements:     $N_DISP_TOTAL"
echo "   Random Samples:    $N_RANDOM_SUBSAMPLES"
echo "   Ell Bins:          $N_ELL_BINS"
echo "   Stride:            $STRIDE"
if [ -n "$LOG_SF_BIN_EDGES_MIN" ]; then
echo "   Custom Bin Edges:  Yes"
fi
echo "============================================================================"
echo " Auto-Continuation:"
echo "   Enabled:           $([ $AUTO_CONTINUE -eq 1 ] && echo Yes || echo No)"
if [ $AUTO_CONTINUE -eq 1 ]; then
echo "   Time Buffer:       ${TIME_BUFFER_MINUTES} minutes"
echo "   Max Chain Jobs:    $MAX_CHAIN_JOBS"
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
    # Continuation job - use existing directory
    WORK_DIR="$RESUME_FROM"
    echo " Continuing in existing directory: $WORK_DIR"
else
    # New job or first in chain
    WORK_DIR="${BASE_DIR}/sfunctor_results/results_${SIM_NAME}/${RUN_NAME}_job${ORIGINAL_JOB_ID}"
    mkdir -p $WORK_DIR
fi

SLICE_LIST="${BASE_DIR}/sfunctor_results/slice_list_${SIM_NAME}.txt"

cd $WORK_DIR

# Save/update configuration for reproducibility
if [ ! -f config.txt ] || [ "$IS_CONTINUATION" != "1" ]; then
    cat > config.txt << EOF
# Structure Function Analysis Configuration
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
LOG_SF_BIN_EDGES_MIN=$LOG_SF_BIN_EDGES_MIN
LOG_SF_BIN_EDGES_MAX=$LOG_SF_BIN_EDGES_MAX
AUTO_CONTINUE=$AUTO_CONTINUE
MAX_CHAIN_JOBS=$MAX_CHAIN_JOBS
EOF
fi

#============================================================================
# RESUME HANDLING
#============================================================================

if [ -n "$RESUME_FROM" ] && [ -d "$RESUME_FROM" ]; then
    if [ "$RESUME_FROM" != "$WORK_DIR" ]; then
        echo ""
        echo " Resume Mode: Copying existing results from:"
        echo "   $RESUME_FROM"
        echo "----------------------------------------------------------------------------"
        
        # Copy existing slice results
        EXISTING_RESULTS=($(ls ${RESUME_FROM}/sf_results_${SIM_TYPE}*.npz 2>/dev/null || true))
        if [ ${#EXISTING_RESULTS[@]} -gt 0 ]; then
            echo " Found ${#EXISTING_RESULTS[@]} existing slice results"
            for result in "${EXISTING_RESULTS[@]}"; do
                cp "$result" "$WORK_DIR/"
                echo "   Copied: $(basename $result)"
            done
        fi
        
        # Copy displacements
        if [ -f "${RESUME_FROM}/displacements.npz" ]; then
            cp "${RESUME_FROM}/displacements.npz" "$WORK_DIR/"
            echo " Copied existing displacements.npz"
        fi
    fi
fi

#============================================================================
# GENERATE SLICE LIST
#============================================================================

if [ ! -f "$SLICE_LIST" ]; then
    echo ""
    echo " Generating slice list..."
    SLICE_DIR="${BASE_DIR}/sfunctor_results/slice_${SIM_NAME}"
    
    if [ ! -d "$SLICE_DIR" ]; then
        # Try alternative location
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
    echo "----------------------------------------------------------------------------"
    python $SFUNCTOR_DIR/generate_displacements.py \
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

# Process each slice using all nodes
mapfile -t ALL_SLICES < $SLICE_LIST
SLICE_COUNT=0
SLICES_TO_PROCESS=0

# Count slices that need processing
for SLICE_PATH in "${ALL_SLICES[@]}"; do
    SLICE_NAME=$(basename $SLICE_PATH .npz)
    if [[ ! " ${PROCESSED_SLICES[@]} " =~ " ${SLICE_NAME} " ]]; then
        SLICES_TO_PROCESS=$((SLICES_TO_PROCESS + 1))
    fi
done

echo " Slices to process: $SLICES_TO_PROCESS"

for SLICE_PATH in "${ALL_SLICES[@]}"; do
    SLICE_NAME=$(basename $SLICE_PATH .npz)
    
    # Skip if already processed
    if [[ " ${PROCESSED_SLICES[@]} " =~ " ${SLICE_NAME} " ]]; then
        continue
    fi
    
    # Check if we're running out of time
    check_and_checkpoint "$WORK_DIR"
    
    SLICE_COUNT=$((SLICE_COUNT + 1))
    echo ""
    echo " [$SLICE_COUNT/$SLICES_TO_PROCESS] Processing: $SLICE_NAME"
    echo " Time remaining: $(($(get_time_remaining_seconds) / 60)) minutes"
    echo " ---------------------------------------------------------------------------"
    
    # Create output directory for this slice
    SLICE_OUTPUT_DIR="$WORK_DIR/${SLICE_NAME}_histograms"
    mkdir -p $SLICE_OUTPUT_DIR
    
    # Build command with optional bin edges
    CMD_BASE="python $SFUNCTOR_DIR/run_node_analysis.py \
        --slice $SLICE_PATH \
        --displacements $WORK_DIR/displacements.npz \
        --output_dir $SLICE_OUTPUT_DIR \
        --stride $STRIDE \
        --N_random_subsamples $N_RANDOM_SUBSAMPLES \
        --stencil_width $STENCIL_WIDTH \
        --n_processes 56"
    
    if [ -n "$LOG_SF_BIN_EDGES_MIN" ]; then
        CMD_BASE="$CMD_BASE --log_sf_bin_edges_min $LOG_SF_BIN_EDGES_MIN"
        CMD_BASE="$CMD_BASE --log_sf_bin_edges_max $LOG_SF_BIN_EDGES_MAX"
    fi
    
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
    
    # Check for errors
    ERROR_COUNT=0
    SUCCESS_COUNT=0
    for logfile in ${SLICE_OUTPUT_DIR}/node_*.log; do
        if [ -f "$logfile" ]; then
            if grep -q "Results saved to" "$logfile" 2>/dev/null; then
                SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            else
                echo " WARNING: Node $(basename $logfile .log) may have failed"
                ERROR_COUNT=$((ERROR_COUNT + 1))
            fi
        fi
    done
    
    echo " Nodes completed: $SUCCESS_COUNT / $TOTAL_NODES"
    if [ $ERROR_COUNT -gt 0 ]; then
        echo " WARNING: $ERROR_COUNT nodes may have errors"
    fi
done

#============================================================================
# PARALLEL MERGING OF SLICE HISTOGRAMS WITH TIME CHECKING
#============================================================================

echo ""
echo " Merging Slice Histograms"
echo "============================================================================"

# Check time before merging
check_and_checkpoint "$WORK_DIR"

# Find all histogram directories that need merging
HIST_DIRS=($(find $WORK_DIR -type d -name "*_histograms" 2>/dev/null | sort))
NUM_HIST_DIRS=${#HIST_DIRS[@]}

if [ $NUM_HIST_DIRS -gt 0 ]; then
    echo " Found $NUM_HIST_DIRS histogram directories to merge"
    
    # Process merging with time checks
    MERGE_COUNT=0
    for HIST_DIR in "${HIST_DIRS[@]}"; do
        # Check time periodically during merging
        if [ $((MERGE_COUNT % 3)) -eq 0 ]; then
            check_and_checkpoint "$WORK_DIR"
        fi
        
        MERGE_COUNT=$((MERGE_COUNT + 1))
        SLICE_NAME=$(basename "$HIST_DIR" "_histograms")
        
        # Skip if already merged
        if [ -f "$WORK_DIR/sf_results_${SLICE_NAME}.npz" ]; then
            echo " [$MERGE_COUNT/$NUM_HIST_DIRS] Already merged: $SLICE_NAME"
            continue
        fi
        
        echo " [$MERGE_COUNT/$NUM_HIST_DIRS] Merging: $SLICE_NAME"
        python $SFUNCTOR_DIR/combine_histograms_fast.py \
            --pattern "${HIST_DIR}/histogram_*.npz" \
            --output "$WORK_DIR/sf_results_${SLICE_NAME}.npz" \
            --mode node \
            --workers 32
        
        # Clean up to save space
        if [ -f "$WORK_DIR/sf_results_${SLICE_NAME}.npz" ]; then
            rm -rf "$HIST_DIR"
            echo "   Cleaned up histogram directory"
        fi
    done
fi

#============================================================================
# FINAL MERGE OF ALL SLICES WITH TIME CHECKING
#============================================================================

echo ""
echo " Final Merge of All Slices"
echo "============================================================================"

# Check time before final merge
check_and_checkpoint "$WORK_DIR"

SLICE_RESULTS=($(ls -1 $WORK_DIR/sf_results_${SIM_TYPE}*.npz 2>/dev/null))
NUM_SLICE_RESULTS=${#SLICE_RESULTS[@]}

if [ $NUM_SLICE_RESULTS -gt 0 ]; then
    echo " Found $NUM_SLICE_RESULTS slice results to merge"
    
    # Skip if already done
    if [ ! -f "$WORK_DIR/sf_results_all_slices.npz" ]; then
        python $SFUNCTOR_DIR/combine_histograms_fast.py \
            --pattern "$WORK_DIR/sf_results_${SIM_TYPE}*.npz" \
            --output "$WORK_DIR/sf_results_all_slices.npz" \
            --mode slice \
            --workers 32
        
        if [ -f "$WORK_DIR/sf_results_all_slices.npz" ]; then
            echo " Successfully created: sf_results_all_slices.npz"
            
            # Get file size
            SIZE=$(du -h "$WORK_DIR/sf_results_all_slices.npz" | cut -f1)
            echo " File size: $SIZE"
        else
            echo " ERROR: Failed to create final merged result"
            exit 1
        fi
    else
        echo " Final merged result already exists: sf_results_all_slices.npz"
    fi
else
    echo " ERROR: No slice results found to merge"
    exit 1
fi

#============================================================================
# GENERATE PLOTS WITH TIME CHECKING
#============================================================================

echo ""
echo " Generating Plots"
echo "============================================================================"

# Check time before plotting
check_and_checkpoint "$WORK_DIR"

# Check if plots already exist
EXISTING_PLOTS=$(ls -1 $WORK_DIR/*.png 2>/dev/null | wc -l)
if [ $EXISTING_PLOTS -gt 10 ]; then
    echo " Plots already exist ($EXISTING_PLOTS found) - skipping generation"
else
    python $SFUNCTOR_DIR/plot_structure_functions.py \
        $WORK_DIR/sf_results_all_slices.npz

    if [ $? -eq 0 ]; then
        echo " Plots generated successfully"
        
        # Count and list generated plots
        NUM_PLOTS=$(ls -1 $WORK_DIR/*.png 2>/dev/null | wc -l)
        echo " Generated $NUM_PLOTS plots:"
        
        for plot in $WORK_DIR/*_mean_structure_functions.png \
                    $WORK_DIR/*_2d_histograms_normalized.png \
                    $WORK_DIR/*_angular_distribution_2d.png \
                    $WORK_DIR/*_cross_product_ratios.png; do
            if [ -f "$plot" ]; then
                echo "   - $(basename $plot)"
            fi
        done
        
        # Count individual channel plots
        CHANNEL_PLOTS=$(ls -1 $WORK_DIR/*_2d_histogram_d*.png 2>/dev/null | wc -l)
        if [ $CHANNEL_PLOTS -gt 0 ]; then
            echo "   - $CHANNEL_PLOTS individual channel 2D histograms"
        fi
    else
        echo " ERROR: Plot generation failed"
    fi
fi

#============================================================================
# CLEANUP AND SUMMARY
#============================================================================

echo ""
echo "============================================================================"
echo " ANALYSIS COMPLETE"
echo "============================================================================"
echo " End Time:      $(date)"
echo " Duration:      $SECONDS seconds"
echo " Results Dir:   $WORK_DIR"
echo ""
echo " Summary:"
echo "   - Processed Slices:    $NUM_SLICE_RESULTS / $TOTAL_SLICES"
echo "   - Output File:         sf_results_all_slices.npz"
echo "   - Plots Generated:     $NUM_PLOTS"
echo "   - Nodes Used:          $TOTAL_NODES"
if [ "$IS_CONTINUATION" == "1" ]; then
echo "   - Chain Position:      $CHAIN_COUNT"
echo "   - Original Job:        $ORIGINAL_JOB_ID"
fi
echo ""
echo " Configuration:"
echo "   - Simulation:          $SIM_NAME"
echo "   - Resolution:          ${RESOLUTION}×${RESOLUTION}"
echo "   - Stencil Width:       $STENCIL_WIDTH"
echo "   - Displacements:       $N_DISP_TOTAL"
echo "   - Random Samples:      $N_RANDOM_SUBSAMPLES"
echo ""
echo " Configuration saved in: $WORK_DIR/config.txt"

# Check if all work is complete
if [ $NUM_SLICE_RESULTS -eq $TOTAL_SLICES ]; then
    echo ""
    echo " ALL SLICES PROCESSED SUCCESSFULLY"
    echo " No continuation needed - job chain complete"
else
    echo ""
    echo " NOTE: Not all slices processed ($NUM_SLICE_RESULTS / $TOTAL_SLICES)"
    if [ "$AUTO_CONTINUE" == "1" ] && [ "$CHAIN_COUNT" -lt "$MAX_CHAIN_JOBS" ]; then
        echo " A continuation job may be submitted if time limit is approaching"
    fi
fi

echo "============================================================================"