#!/bin/bash
#SBATCH -A ast207
#SBATCH -J sf_5120_b1_sw3
#SBATCH -o sf_%x_%j.out
#SBATCH -e sf_%x_%j.err
#SBATCH --time=2:00:00
#SBATCH -p batch
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56

# Configuration
SIM_TYPE="Turb"
RESOLUTION="5120"
BETA="1"
DEDT="025"
SCHEME="plm"
STENCIL_WIDTH=3
N_DISP_TOTAL=50000
N_RANDOM_SUBSAMPLES=50000
N_ELL_BINS=128
STRIDE=1
SEED=42

echo "============================================================================"
echo " STRUCTURE FUNCTION ANALYSIS"
echo " Resolution: $RESOLUTION, Beta: $BETA, Stencil Width: $STENCIL_WIDTH"
echo "============================================================================"
echo " Job ID: $SLURM_JOB_ID"
echo " Start Time: $(date)"
echo " Nodes: $SLURM_JOB_NUM_NODES"
echo "============================================================================"

# Load modules
module reset
module load PrgEnv-gnu

# Activate virtual environment
source /autofs/nccs-svm1_home2/dfielding/SFunctor/venv_frontier/bin/activate

# Set paths
BASE_DIR="/lustre/orion/ast207/proj-shared/dfielding/Production_plm"
SFUNCTOR_DIR="/autofs/nccs-svm1_home2/dfielding/SFunctor"
SIM_NAME="${SIM_TYPE}_${RESOLUTION}_beta${BETA}_dedt${DEDT}_${SCHEME}"
RUN_NAME="ndisp${N_DISP_TOTAL}_nrand${N_RANDOM_SUBSAMPLES}_nell${N_ELL_BINS}_sw${STENCIL_WIDTH}"
WORK_DIR="${BASE_DIR}/sfunctor_results/results_${SIM_NAME}/${RUN_NAME}_job${SLURM_JOB_ID}"

mkdir -p $WORK_DIR
cd $WORK_DIR

# Save configuration
cat > config.txt << CONFIG_EOF
SIM_NAME=$SIM_NAME
RESOLUTION=$RESOLUTION
BETA=$BETA
STENCIL_WIDTH=$STENCIL_WIDTH
N_DISP_TOTAL=$N_DISP_TOTAL
N_RANDOM_SUBSAMPLES=$N_RANDOM_SUBSAMPLES
N_ELL_BINS=$N_ELL_BINS
STRIDE=$STRIDE
N_NODES=$SLURM_JOB_NUM_NODES
CONFIG_EOF

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
echo " Generating displacements..."
python $SFUNCTOR_DIR/scripts/production/generate_displacements.py \
    --n_disp_total $N_DISP_TOTAL \
    --n_ell_bins $N_ELL_BINS \
    --Nres $RESOLUTION \
    --seed $SEED

# Get nodes
NODES=($(scontrol show hostname $SLURM_JOB_NODELIST))
TOTAL_NODES=${#NODES[@]}

# Process slices
echo ""
echo " Processing Slices"
echo "============================================================================"

mapfile -t ALL_SLICES < $SLICE_LIST
SLICE_COUNT=0

for SLICE_PATH in "${ALL_SLICES[@]}"; do
    SLICE_NAME=$(basename $SLICE_PATH .npz)
    SLICE_COUNT=$((SLICE_COUNT + 1))
    
    echo " [$SLICE_COUNT/$TOTAL_SLICES] Processing: $SLICE_NAME"
    
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
done

# Merge histograms
echo ""
echo " Merging histograms..."

for dir in $WORK_DIR/*_histograms; do
    [ -d "$dir" ] || continue
    SLICE_NAME=$(basename "$dir" "_histograms")
    
    python $SFUNCTOR_DIR/scripts/production/combine_histograms_fast.py \
        --pattern "${dir}/histogram_*.npz" \
        --output "$WORK_DIR/sf_results_${SLICE_NAME}.npz" \
        --mode node \
        --workers 32
    
    [ -f "$WORK_DIR/sf_results_${SLICE_NAME}.npz" ] && rm -rf "$dir"
done

# Final merge
echo " Final merge..."
python $SFUNCTOR_DIR/scripts/production/combine_histograms_fast.py \
    --pattern "$WORK_DIR/sf_results_${SIM_TYPE}*.npz" \
    --output "$WORK_DIR/sf_results_all_slices.npz" \
    --mode slice \
    --workers 32

# Generate plots
echo " Generating plots..."
python $SFUNCTOR_DIR/plotting_scripts/plot_structure_functions.py $WORK_DIR/sf_results_all_slices.npz

echo ""
echo "============================================================================"
echo " COMPLETE"
echo " Results: $WORK_DIR"
echo " End Time: $(date)"
echo "============================================================================"
