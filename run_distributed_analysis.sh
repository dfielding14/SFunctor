#!/bin/bash
#SBATCH -A AST207
#SBATCH -J SF_DISTRIBUTED
#SBATCH -o sf_distributed_%j.out
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -N 4
#SBATCH --ntasks-per-node=1

# Distributed structure function analysis using multiprocessing
# Each node processes a subset of displacements independently

echo "======================================"
echo "Distributed Structure Function Analysis"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Date: $(date)"
echo "======================================"

# Load required modules
module reset
module load gcc/9.3.0 python/.3.11-anaconda3

# Activate virtual environment
source /ccs/home/dfielding/SFunctor/venv_sfunctor/bin/activate

# Set paths
SFUNCTOR_DIR="/ccs/home/dfielding/SFunctor"
WORK_DIR="/lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results/distributed_run_${SLURM_JOB_ID}"
SLICE_LIST="/lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results/slice_list_Turb_640_beta25_dedt025_plm.txt"

# Configuration
N_DISP_TOTAL=10000
N_ELL_BINS=64
N_RANDOM_SUBSAMPLES=5000
STRIDE=1
STENCIL_WIDTH=2

# Create working directory
mkdir -p $WORK_DIR
cd $WORK_DIR

echo ""
echo "Configuration:"
echo "- Working directory: $WORK_DIR"
echo "- Slice list: $SLICE_LIST"
echo "- Total displacements: $N_DISP_TOTAL"
echo "- Random subsamples: $N_RANDOM_SUBSAMPLES"
echo "- Nodes: $SLURM_JOB_NUM_NODES"
echo ""

# Step 1: Generate displacements (only once)
echo "======================================"
echo "Step 1: Generating displacements"
echo "======================================"
python ${SFUNCTOR_DIR}/generate_displacements.py \
    --n_disp_total $N_DISP_TOTAL \
    --n_ell_bins $N_ELL_BINS \
    --output displacements.npz \
    --seed 42

echo ""

# Step 2: Get list of slices to process
if [ ! -f "$SLICE_LIST" ]; then
    echo "Error: Slice list not found: $SLICE_LIST"
    exit 1
fi

# Read slices into array (limit to first 10 for testing)
mapfile -t SLICES < <(head -10 "$SLICE_LIST")
N_SLICES=${#SLICES[@]}

echo "Processing $N_SLICES slices"
echo ""

# Step 3: Process each slice
for i in "${!SLICES[@]}"; do
    SLICE_PATH="${SLICES[$i]}"
    SLICE_NAME=$(basename "$SLICE_PATH" .npz)
    
    echo "======================================"
    echo "Processing slice $((i+1))/$N_SLICES: $SLICE_NAME"
    echo "======================================"
    
    # Create output directory for this slice
    SLICE_OUTPUT_DIR="${WORK_DIR}/histograms_${SLICE_NAME}"
    mkdir -p $SLICE_OUTPUT_DIR
    
    # Launch parallel jobs (one per node)
    echo "Launching $SLURM_JOB_NUM_NODES parallel jobs..."
    
    # Use srun to launch one task per node
    for node_id in $(seq 0 $((SLURM_JOB_NUM_NODES - 1))); do
        srun --exclusive -N1 -n1 --cpu-bind=none \
            python ${SFUNCTOR_DIR}/run_node_analysis.py \
            --slice "$SLICE_PATH" \
            --displacements displacements.npz \
            --node_id $node_id \
            --total_nodes $SLURM_JOB_NUM_NODES \
            --output_dir $SLICE_OUTPUT_DIR \
            --stride $STRIDE \
            --N_random_subsamples $N_RANDOM_SUBSAMPLES \
            --stencil_width $STENCIL_WIDTH \
            --n_processes 0 \
            > ${SLICE_OUTPUT_DIR}/node_${node_id}.log 2>&1 &
    done
    
    # Wait for all nodes to complete
    wait
    
    echo "All nodes completed for $SLICE_NAME"
    
    # Step 4: Combine histograms
    echo "Combining histograms..."
    python ${SFUNCTOR_DIR}/combine_histograms.py \
        --pattern "${SLICE_OUTPUT_DIR}/histogram_*.npz" \
        --output "${WORK_DIR}/sf_results_${SLICE_NAME}.npz" \
        --verbose
    
    # Optional: Clean up intermediate files
    # rm -rf $SLICE_OUTPUT_DIR
    
    echo ""
done

# Summary
echo "======================================"
echo "Analysis Complete"
echo "======================================"
echo "Results saved in: $WORK_DIR"
echo "Processed $N_SLICES slices"
echo ""
ls -lh ${WORK_DIR}/sf_results_*.npz