#!/bin/bash
#SBATCH -A AST207
#SBATCH -J SF_DISTRIBUTED
#SBATCH -o sf_distributed_%j.out
#SBATCH -t 36:00:00
#SBATCH -p batch
#SBATCH -N 64
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

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

# Configuration
SIM_NAME="Turb_5120_beta25_dedt025_plm"
BASE_DIR="/lustre/orion/ast207/proj-shared/dfielding/Production_plm"

# Configuration
N_DISP_TOTAL=100000
N_ELL_BINS=128
N_RANDOM_SUBSAMPLES=10000
STRIDE=1
STENCIL_WIDTH=2
NRES=5120

# Resume from previous run (optional)
# Set this to the directory of a previous incomplete run to resume from it
RESUME_FROM_DIR="${RESUME_FROM_DIR:-}"

# Set paths
SFUNCTOR_DIR="/ccs/home/dfielding/SFunctor"
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
        echo "No existing results found in resume directory."
        echo ""
    fi
    
    # Copy displacements if they exist
    if [ -f "${RESUME_FROM_DIR}/displacements.npz" ]; then
        echo "Copying existing displacements..."
        cp -v "${RESUME_FROM_DIR}/displacements.npz" "$WORK_DIR/"
        echo ""
    fi
fi

echo ""
echo "Configuration:"
echo "- Working directory: $WORK_DIR"
echo "- Slice list: $SLICE_LIST"
echo "- Total displacements: $N_DISP_TOTAL"
echo "- Random subsamples: $N_RANDOM_SUBSAMPLES"
echo "- Grid resolution: $NRES"
echo "- Stencil width: $STENCIL_WIDTH"
echo "- Nodes: $SLURM_JOB_NUM_NODES"
echo ""

# Step 1: Generate displacements (only if not already present)
echo "======================================"
echo "Step 1: Generating/checking displacements"
echo "======================================"
if [ -f "displacements.npz" ]; then
    echo "Displacements already exist, skipping generation."
else
    echo "Generating new displacements..."
    python ${SFUNCTOR_DIR}/scripts/production/generate_displacements.py \
        --n_disp_total $N_DISP_TOTAL \
        --n_ell_bins $N_ELL_BINS \
        --Nres $NRES \
        --stencil_width $STENCIL_WIDTH \
        --output displacements.npz \
        --seed 42
fi

echo ""

# Step 2: Get list of slices to process
if [ ! -f "$SLICE_LIST" ]; then
    echo "Error: Slice list not found: $SLICE_LIST"
    exit 1
fi

# Read slices into array
mapfile -t SLICES < <(cat "$SLICE_LIST")
N_SLICES=${#SLICES[@]}

# Build list of completed slices
declare -A COMPLETED_SLICES
COMPLETED_COUNT=0
for result_file in ${WORK_DIR}/sf_results_Turb*.npz; do
    if [ -f "$result_file" ]; then
        # Extract slice name from result filename
        basename_result=$(basename "$result_file" .npz)
        slice_name=${basename_result#sf_results_}
        COMPLETED_SLICES["$slice_name"]=1
        ((COMPLETED_COUNT++))
    fi
done

REMAINING_COUNT=$((N_SLICES - COMPLETED_COUNT))
echo "Total slices: $N_SLICES"
echo "Already completed: $COMPLETED_COUNT"
echo "Remaining to process: $REMAINING_COUNT"
echo ""

# Step 3: Process each slice
PROCESSED_IN_THIS_RUN=0
for i in "${!SLICES[@]}"; do
    SLICE_PATH="${SLICES[$i]}"
    SLICE_NAME=$(basename "$SLICE_PATH" .npz)
    
    # Check if this slice was already processed
    if [ "${COMPLETED_SLICES[$SLICE_NAME]}" == "1" ]; then
        echo "======================================"
        echo "Skipping slice $((i+1))/$N_SLICES: $SLICE_NAME (already completed)"
        echo "======================================"
        echo ""
        continue
    fi
    
    echo "======================================"
    echo "Processing slice $((i+1))/$N_SLICES: $SLICE_NAME"
    echo "======================================"
    ((PROCESSED_IN_THIS_RUN++))
    
    # Create output directory for this slice
    SLICE_OUTPUT_DIR="${WORK_DIR}/histograms_${SLICE_NAME}"
    mkdir -p $SLICE_OUTPUT_DIR
    
    # Launch parallel jobs (one per node)
    echo "Launching $SLURM_JOB_NUM_NODES parallel jobs..."
    
    # Use srun to launch one task per node
    for node_id in $(seq 0 $((SLURM_JOB_NUM_NODES - 1))); do
        srun --exclusive -N1 -n1 \
            --cpus-per-task=32 \
            --cpu-bind=cores \
            python ${SFUNCTOR_DIR}/scripts/production/run_node_analysis.py \
            --slice "$SLICE_PATH" \
            --displacements displacements.npz \
            --node_id $node_id \
            --total_nodes $SLURM_JOB_NUM_NODES \
            --output_dir $SLICE_OUTPUT_DIR \
            --stride $STRIDE \
            --N_random_subsamples $N_RANDOM_SUBSAMPLES \
            --stencil_width $STENCIL_WIDTH \
            --n_processes 32 \
            --log_sf_bin_edges_min -5 -5 -5 -5 -5 -5 -2 -2 -2 -2 -6 \
            --log_sf_bin_edges_max 1 1 1 1 1 1 4 4 4 4 4 \
            --N_sf_bin_edges 201 \
            --log_product_bin_edges_min -8 \
            --log_product_bin_edges_max 5 \
            --N_product_bin_edges 201 \
            > ${SLICE_OUTPUT_DIR}/node_${node_id}.log 2>&1 &
    done
    
    # Wait for all nodes to complete
    wait
    
    echo "All nodes completed for $SLICE_NAME"
    
    # Step 4: Combine histograms
    echo "Combining histograms..."
    python ${SFUNCTOR_DIR}/scripts/production/combine_histograms.py \
        --pattern "${SLICE_OUTPUT_DIR}/histogram_*.npz" \
        --output "${WORK_DIR}/sf_results_${SLICE_NAME}.npz" \
        --verbose
    
    # Clean up intermediate node histogram files
    echo "Cleaning up intermediate files..."
    rm -rf $SLICE_OUTPUT_DIR
    
    echo ""
done

# Step 5: Merge all slice results
echo "======================================"
echo "Merging all slice results"
echo "======================================"
python ${SFUNCTOR_DIR}/scripts/production/combine_histograms.py \
    --mode slice \
    --pattern "${WORK_DIR}/sf_results_*.npz" \
    --output "${WORK_DIR}/sf_results_all_slices.npz" \
    --verbose

# Step 6: Create plots
echo "======================================"
echo "Creating plots"
echo "======================================"
python ${SFUNCTOR_DIR}/plotting_scripts/plot_structure_functions.py \
    "${WORK_DIR}/sf_results_all_slices.npz" \
    --output_dir "${WORK_DIR}/plots" \
    --format png \
    --dpi 150

# Summary
echo ""
echo "======================================"
echo "Analysis Complete"
echo "======================================"
echo "Results saved in: $WORK_DIR"
echo "Total slices: $N_SLICES"
echo "Previously completed: $COMPLETED_COUNT"
echo "Processed in this run: $PROCESSED_IN_THIS_RUN"
echo "Total completed: $((COMPLETED_COUNT + PROCESSED_IN_THIS_RUN))"
echo ""
echo "Individual slice results:"
ls -lh ${WORK_DIR}/sf_results_Turb*.npz
echo ""
echo "Combined result:"
ls -lh ${WORK_DIR}/sf_results_all_slices.npz
echo ""
echo "Plots saved in: ${WORK_DIR}/plots"
ls -lh ${WORK_DIR}/plots/*.png