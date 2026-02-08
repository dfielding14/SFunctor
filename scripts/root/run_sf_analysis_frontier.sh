#!/bin/bash
#============================================================================
# CONFIGURABLE STRUCTURE FUNCTION ANALYSIS FOR FRONTIER HPC
#============================================================================
# This script provides a unified, configurable interface for running
# structure function analysis on Frontier with intelligent parallel merging
#
# Usage: Edit the configuration section below, then submit with:
#        sbatch run_sf_analysis_frontier.sh
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

# Optional: override unified Δ bin edges (leave empty to use run_node_analysis defaults).
# If set, these must contain one value per histogram channel (see `sfunctor.core.histograms.Channel`).
LOG_DELTA_BIN_EDGES_MIN=""
LOG_DELTA_BIN_EDGES_MAX=""
N_DELTA_BIN_EDGES=""

# HPC Configuration
N_NODES=64                           # Number of nodes to request
TIME_LIMIT="4:00:00"                # Wall time limit (HH:MM:SS)
QUEUE="debug"                        # Queue (debug, regular, premium)

# Resume Configuration (optional - set to previous job directory to resume)
RESUME_FROM=""                       # Leave empty for fresh run, or set to previous job dir

#============================================================================
# SLURM DIRECTIVES - These use the configuration above
#============================================================================

# Construct simulation name for job naming
SIM_NAME="${SIM_TYPE}_${RESOLUTION}_beta${BETA}_dedt${DEDT}_${SCHEME}"

# Create descriptive job name (keep under 64 chars for SLURM)
JOB_NAME="sf_b${BETA}_${RESOLUTION}_sw${STENCIL_WIDTH}"

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
# SCRIPT BEGINS HERE - DO NOT MODIFY BELOW UNLESS YOU KNOW WHAT YOU'RE DOING
#============================================================================

echo "============================================================================"
echo " STRUCTURE FUNCTION ANALYSIS - FRONTIER HPC"
echo "============================================================================"
echo " Job Configuration:"
echo "   Job ID:        $SLURM_JOB_ID"
echo "   Job Name:      $JOB_NAME"
echo "   Start Time:    $(date)"
echo "   Nodes:         $SLURM_JOB_NUM_NODES"
echo "   CPUs/Node:     56"
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
if [ -n "$LOG_DELTA_BIN_EDGES_MIN" ]; then
echo "   Custom Δ Bin Edges:  Yes"
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
WORK_DIR="${BASE_DIR}/sfunctor_results/results_${SIM_NAME}/${RUN_NAME}_job${SLURM_JOB_ID}"
SLICE_LIST="${BASE_DIR}/sfunctor_results/slice_list_${SIM_NAME}.txt"

# Create working directory
mkdir -p $WORK_DIR
cd $WORK_DIR

# Save configuration for reproducibility
cat > config.txt << EOF
# Structure Function Analysis Configuration
# Job ID: $SLURM_JOB_ID
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
LOG_DELTA_BIN_EDGES_MIN=$LOG_DELTA_BIN_EDGES_MIN
LOG_DELTA_BIN_EDGES_MAX=$LOG_DELTA_BIN_EDGES_MAX
N_DELTA_BIN_EDGES=$N_DELTA_BIN_EDGES
EOF

#============================================================================
# RESUME HANDLING
#============================================================================

if [ -n "$RESUME_FROM" ] && [ -d "$RESUME_FROM" ]; then
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
# PROCESS SLICES
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
    
    SLICE_COUNT=$((SLICE_COUNT + 1))
    echo ""
    echo " [$SLICE_COUNT/$SLICES_TO_PROCESS] Processing: $SLICE_NAME"
    echo " ---------------------------------------------------------------------------"
    
    # Create output directory for this slice
    SLICE_OUTPUT_DIR="$WORK_DIR/${SLICE_NAME}_histograms"
    mkdir -p $SLICE_OUTPUT_DIR
    
    # Build command with optional bin edges
    CMD_BASE="python $SFUNCTOR_DIR/scripts/production/run_node_analysis.py \
        --slice $SLICE_PATH \
        --displacements $WORK_DIR/displacements.npz \
        --output_dir $SLICE_OUTPUT_DIR \
        --stride $STRIDE \
        --N_random_subsamples $N_RANDOM_SUBSAMPLES \
        --stencil_width $STENCIL_WIDTH \
        --n_processes 56"
    
    if [ -n "$LOG_DELTA_BIN_EDGES_MIN" ]; then
        CMD_BASE="$CMD_BASE --log_delta_bin_edges_min $LOG_DELTA_BIN_EDGES_MIN"
        CMD_BASE="$CMD_BASE --log_delta_bin_edges_max $LOG_DELTA_BIN_EDGES_MAX"
    fi
    if [ -n "$N_DELTA_BIN_EDGES" ]; then
        CMD_BASE="$CMD_BASE --N_delta_bin_edges $N_DELTA_BIN_EDGES"
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
# PARALLEL MERGING OF SLICE HISTOGRAMS
#============================================================================

echo ""
echo " Merging Slice Histograms"
echo "============================================================================"

# Find all histogram directories that need merging
HIST_DIRS=($(find $WORK_DIR -type d -name "*_histograms" 2>/dev/null | sort))
NUM_HIST_DIRS=${#HIST_DIRS[@]}

if [ $NUM_HIST_DIRS -gt 0 ]; then
    echo " Found $NUM_HIST_DIRS histogram directories to merge"
    
    # If we have multiple nodes, distribute merge jobs across them
    if [ $TOTAL_NODES -gt 1 ] && [ $NUM_HIST_DIRS -gt 1 ]; then
        echo " Distributing merge jobs across $TOTAL_NODES nodes..."
        
        # Calculate distribution
        DIRS_PER_NODE=$(( (NUM_HIST_DIRS + TOTAL_NODES - 1) / TOTAL_NODES ))
        
        # Launch parallel merge jobs
        for NODE_ID in $(seq 0 $((TOTAL_NODES - 1))); do
            NODE=${NODES[$NODE_ID]}
            
            # Calculate which directories this node should process
            START_IDX=$(( NODE_ID * DIRS_PER_NODE ))
            END_IDX=$(( START_IDX + DIRS_PER_NODE - 1 ))
            
            # Don't exceed array bounds
            if [ $END_IDX -ge $NUM_HIST_DIRS ]; then
                END_IDX=$(( NUM_HIST_DIRS - 1 ))
            fi
            
            # Skip if no work for this node
            if [ $START_IDX -ge $NUM_HIST_DIRS ]; then
                continue
            fi
            
            # Create merge script for this node
            MERGE_SCRIPT="$WORK_DIR/merge_node_${NODE_ID}.sh"
            cat > $MERGE_SCRIPT << 'EOF'
#!/bin/bash
source /autofs/nccs-svm1_home2/dfielding/SFunctor/venv_frontier/bin/activate
EOF
            
            # Add merge commands for assigned directories
            NUM_ASSIGNED=0
            for IDX in $(seq $START_IDX $END_IDX); do
                HIST_DIR="${HIST_DIRS[$IDX]}"
                SLICE_NAME=$(basename "$HIST_DIR" "_histograms")
                NUM_ASSIGNED=$((NUM_ASSIGNED + 1))
                
                cat >> $MERGE_SCRIPT << EOF
echo "Node $NODE_ID [$NUM_ASSIGNED]: Merging $SLICE_NAME"
python $SFUNCTOR_DIR/scripts/production/combine_histograms_fast.py \
    --pattern "${HIST_DIR}/histogram_*.npz" \
    --output "$WORK_DIR/sf_results_${SLICE_NAME}.npz" \
    --mode node \
    --workers 32
if [ \$? -eq 0 ]; then
    echo "Node $NODE_ID: Successfully merged $SLICE_NAME"
    rm -rf "${HIST_DIR}"
else
    echo "Node $NODE_ID: Failed to merge $SLICE_NAME"
fi
EOF
            done
            
            chmod +x $MERGE_SCRIPT
            
            echo " Node $NODE_ID: Processing $NUM_ASSIGNED directories (indices $START_IDX-$END_IDX)"
            
            # Launch merge job on node
            srun --exclusive -N 1 -n 1 -w $NODE \
                --cpus-per-task=56 \
                $MERGE_SCRIPT \
                > $WORK_DIR/merge_node_${NODE_ID}.log 2>&1 &
        done
        
        echo " Waiting for parallel merge jobs to complete..."
        wait
        
        # Check merge results
        MERGED_COUNT=$(ls -1 $WORK_DIR/sf_results_${SIM_TYPE}*.npz 2>/dev/null | wc -l)
        echo " Successfully merged: $MERGED_COUNT / $NUM_HIST_DIRS slices"
        
    else
        # Single node merge (sequential with parallel file loading)
        echo " Merging sequentially using fast parallel combiner..."
        
        MERGE_COUNT=0
        for HIST_DIR in "${HIST_DIRS[@]}"; do
            MERGE_COUNT=$((MERGE_COUNT + 1))
            SLICE_NAME=$(basename "$HIST_DIR" "_histograms")
            
            echo " [$MERGE_COUNT/$NUM_HIST_DIRS] Merging: $SLICE_NAME"
            python $SFUNCTOR_DIR/scripts/production/combine_histograms_fast.py \
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
fi

#============================================================================
# FINAL MERGE OF ALL SLICES
#============================================================================

echo ""
echo " Final Merge of All Slices"
echo "============================================================================"

SLICE_RESULTS=($(ls -1 $WORK_DIR/sf_results_${SIM_TYPE}*.npz 2>/dev/null))
NUM_SLICE_RESULTS=${#SLICE_RESULTS[@]}

if [ $NUM_SLICE_RESULTS -gt 0 ]; then
    echo " Found $NUM_SLICE_RESULTS slice results to merge"
    
    python $SFUNCTOR_DIR/scripts/production/combine_histograms_fast.py \
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
    echo " ERROR: No slice results found to merge"
    exit 1
fi

#============================================================================
# GENERATE PLOTS
#============================================================================

echo ""
echo " Generating Plots"
echo "============================================================================"

python $SFUNCTOR_DIR/plotting_scripts/plot_structure_functions.py \
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

#============================================================================
# CLEANUP AND SUMMARY
#============================================================================

# Optional: Remove individual slice results to save space (keep only combined)
# Uncomment the following lines if you want to save space:
# echo ""
# echo " Cleaning up individual slice results..."
# for result in $WORK_DIR/sf_results_${SIM_TYPE}*.npz; do
#     if [ -f "$result" ] && [ "$result" != "$WORK_DIR/sf_results_all_slices.npz" ]; then
#         rm "$result"
#     fi
# done

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
echo ""
echo " Configuration:"
echo "   - Simulation:          $SIM_NAME"
echo "   - Resolution:          ${RESOLUTION}×${RESOLUTION}"
echo "   - Stencil Width:       $STENCIL_WIDTH"
echo "   - Displacements:       $N_DISP_TOTAL"
echo "   - Random Samples:      $N_RANDOM_SUBSAMPLES"
echo ""
echo " Configuration saved in: $WORK_DIR/config.txt"
echo "============================================================================"
