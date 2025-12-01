#!/bin/bash
#SBATCH -A AST207
#SBATCH -J sf_beta25_0640_sw3
#SBATCH -o sf_distributed_%j.out
#SBATCH -e sf_distributed_%j.err
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56

set -euo pipefail

# Distributed structure-function analysis for Turb_640_beta25_dedt025_plm
# stencil width = 3, Ndisp = 12,500, Nrand = 12,500
# Runs on 2 Frontier nodes for 2 hours (debug queue)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

module reset
module load PrgEnv-gnu
source /autofs/nccs-svm1_home2/dfielding/SFunctor/venv_frontier/bin/activate
# Ensure local project modules (sfunctor, plotting_scripts, etc.) are importable
export SFUNCTOR_DIR="/autofs/nccs-svm1_home2/dfielding/SFunctor"
export PYTHONPATH="${SFUNCTOR_DIR}:${PYTHONPATH:-}"

SIM_NAME="Turb_640_beta25_dedt025_plm"
BASE_DIR="/lustre/orion/ast207/proj-shared/dfielding/Production_plm"

N_DISP_TOTAL=12500
N_RANDOM_SUBSAMPLES=12500
N_ELL_BINS=128
STRIDE=1
STENCIL_WIDTH=3
NRES=640
SEED=42

LOG_SF_BIN_EDGES_MIN="-5 -5 -5 -5 -5 -5 -2 -2 -2 -2 -5"
LOG_SF_BIN_EDGES_MAX="1 1 1 1 1 1 4 4 4 4 3"

RUN_NAME="ndisp${N_DISP_TOTAL}_nrand${N_RANDOM_SUBSAMPLES}_nell${N_ELL_BINS}_sw${STENCIL_WIDTH}_job${SLURM_JOB_ID}"
WORK_DIR="${BASE_DIR}/sfunctor_results/results_${SIM_NAME}/${RUN_NAME}"
SLICE_LIST="${BASE_DIR}/sfunctor_results/slice_list_${SIM_NAME}.txt"

log "============================================================"
log "Starting distributed structure-function analysis"
log "Job ID: ${SLURM_JOB_ID}"
log "Nodes requested: ${SLURM_JOB_NUM_NODES}"
log "CPUs per task: ${SLURM_CPUS_PER_TASK}"
log "Working directory: ${WORK_DIR}"
log "Slice list: ${SLICE_LIST}"
log "============================================================"

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

if [ ! -f "$SLICE_LIST" ]; then
    SLICE_DIR="${BASE_DIR}/sfunctor_results/slice_${SIM_NAME}"
    [ ! -d "$SLICE_DIR" ] && SLICE_DIR="${BASE_DIR}/slice_${SIM_NAME}"
    if [ ! -d "$SLICE_DIR" ]; then
        echo "ERROR: slice directory not found: $SLICE_DIR"
        exit 1
    fi
    ls "$SLICE_DIR"/*.npz > "$SLICE_LIST"
fi
TOTAL_SLICES=$(wc -l < "$SLICE_LIST")
log "Total slices found: ${TOTAL_SLICES}"

if [ ! -f "$WORK_DIR/displacements.npz" ]; then
    log "Generating displacements (N_disp=${N_DISP_TOTAL}, N_ell=${N_ELL_BINS}, Nres=${NRES})"
    python "$SFUNCTOR_DIR/scripts/production/generate_displacements.py" \
        --n_disp_total "$N_DISP_TOTAL" \
        --n_ell_bins "$N_ELL_BINS" \
        --stencil_width "$STENCIL_WIDTH" \
        --Nres "$NRES" \
        --seed "$SEED"
    log "Finished generating displacements"
else
    log "Reusing existing displacement file at $WORK_DIR/displacements.npz"
fi

NODES=($(scontrol show hostname $SLURM_JOB_NODELIST))
TOTAL_NODES=${#NODES[@]}
log "Node list (${TOTAL_NODES} nodes): ${NODES[*]}"
mapfile -t ALL_SLICES < "$SLICE_LIST"

for ((SLICE_IDX=0; SLICE_IDX<${#ALL_SLICES[@]}; SLICE_IDX++)); do
    SLICE_PATH="${ALL_SLICES[$SLICE_IDX]}"
    SLICE_NAME=$(basename "$SLICE_PATH" .npz)
    log "-----"
    log "Processing slice ${SLICE_NAME} ($((SLICE_IDX + 1))/${TOTAL_SLICES})"
    SLICE_OUTPUT_DIR="$WORK_DIR/${SLICE_NAME}_histograms"
    mkdir -p "$SLICE_OUTPUT_DIR"

    for NODE_ID in $(seq 0 $((TOTAL_NODES - 1))); do
        NODE=${NODES[$NODE_ID]}
        log "Launching node ${NODE_ID}/${TOTAL_NODES} on ${NODE} for slice ${SLICE_NAME}"
        srun --exclusive -N 1 -n 1 -w "$NODE" --cpus-per-task=56 \
            python "$SFUNCTOR_DIR/scripts/production/run_node_analysis.py" \
                --slice "$SLICE_PATH" \
                --displacements "$WORK_DIR/displacements.npz" \
                --node_id "$NODE_ID" \
                --total_nodes "$TOTAL_NODES" \
                --output_dir "$SLICE_OUTPUT_DIR" \
                --stride "$STRIDE" \
                --N_random_subsamples "$N_RANDOM_SUBSAMPLES" \
                --stencil_width "$STENCIL_WIDTH" \
                --n_processes 56 \
                --log_sf_bin_edges_min $LOG_SF_BIN_EDGES_MIN \
                --log_sf_bin_edges_max $LOG_SF_BIN_EDGES_MAX \
                --N_sf_bin_edges 201 \
                --log_product_bin_edges_min -5 \
                --log_product_bin_edges_max 5 \
                --N_product_bin_edges 201 \
            > "$SLICE_OUTPUT_DIR/node_${NODE_ID}.log" 2>&1 &
    done
    log "Waiting for node tasks to finish for slice ${SLICE_NAME}"
    wait
    log "Node tasks complete for slice ${SLICE_NAME}"

    log "Combining node histograms for slice ${SLICE_NAME}"
    python "$SFUNCTOR_DIR/scripts/production/combine_histograms_fast.py" \
        --pattern "${SLICE_OUTPUT_DIR}/histogram_*.npz" \
        --output "$WORK_DIR/sf_results_${SLICE_NAME}.npz" \
        --mode node \
        --workers 32

    log "Removing temporary histogram directory for slice ${SLICE_NAME}"
    rm -rf "$SLICE_OUTPUT_DIR"
done

log "Combining slice-level histograms into all-slice result"
python "$SFUNCTOR_DIR/scripts/production/combine_histograms_fast.py" \
    --pattern "$WORK_DIR/sf_results_${SIM_NAME}_axis*.npz" \
    --output "$WORK_DIR/sf_results_all_slices.npz" \
    --mode slice \
    --workers 32

log "Generating diagnostic plots"
python "$SFUNCTOR_DIR/plotting_scripts/plot_structure_functions.py" \
    "$WORK_DIR/sf_results_all_slices.npz"

log "All processing complete"
