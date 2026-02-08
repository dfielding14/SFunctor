#!/bin/bash
#SBATCH -A AST207
#SBATCH -J sf_beta25_0640_sw3_andes
#SBATCH -o sf_distributed_%j.out
#SBATCH -e sf_distributed_%j.err
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

set -euo pipefail

# Distributed structure-function analysis for Turb_640_beta25_dedt025_plm on Andes
# stencil width = 3, Ndisp = 12,500, Nrand = 12,500
# Runs on 2 Andes nodes for 2 hours (batch)

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

log_duration() {
    local start_ts=$1
    local end_ts=$2
    local label=$3
    local delta=$((end_ts - start_ts))
    log "${label} completed in ${delta}s"
}

SCRIPT_START=$(date +%s)

# Andes modules and virtual environment
module reset
module load gcc/9.3.0 python/.3.11-anaconda3
source /ccs/home/dfielding/SFunctor/venv_sfunctor/bin/activate
# Ensure local project modules (sfunctor, plotting_scripts, etc.) are importable
export SFUNCTOR_DIR="/ccs/home/dfielding/SFunctor"
export PYTHONPATH="${SFUNCTOR_DIR}:${PYTHONPATH:-}"

CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-32}

SIM_NAME="Turb_640_beta25_dedt025_plm"
BASE_DIR="/lustre/orion/ast207/proj-shared/dfielding/Production_plm"

N_DISP_TOTAL=12500
N_RANDOM_SUBSAMPLES=12500
N_ELL_BINS=128
STRIDE=1
STENCIL_WIDTH=3
NRES=640
SEED=42

# Optional: per-channel Δ bin edges (requires 26 values to override defaults)
LOG_DELTA_BIN_EDGES_MIN=""
LOG_DELTA_BIN_EDGES_MAX=""
N_DELTA_BIN_EDGES=""

EXTRA_BINS_ARGS=()
if [ -n "$LOG_DELTA_BIN_EDGES_MIN" ] && [ -z "$LOG_DELTA_BIN_EDGES_MAX" ]; then
    log "ERROR: LOG_DELTA_BIN_EDGES_MIN set but LOG_DELTA_BIN_EDGES_MAX is empty"
    exit 1
fi
if [ -n "$LOG_DELTA_BIN_EDGES_MIN" ]; then
    EXTRA_BINS_ARGS+=(--log_delta_bin_edges_min $LOG_DELTA_BIN_EDGES_MIN --log_delta_bin_edges_max $LOG_DELTA_BIN_EDGES_MAX)
fi
if [ -n "$N_DELTA_BIN_EDGES" ]; then
    EXTRA_BINS_ARGS+=(--N_delta_bin_edges $N_DELTA_BIN_EDGES)
fi

RUN_NAME="ndisp${N_DISP_TOTAL}_nrand${N_RANDOM_SUBSAMPLES}_nell${N_ELL_BINS}_sw${STENCIL_WIDTH}_job${SLURM_JOB_ID}"
WORK_DIR="${BASE_DIR}/sfunctor_results/results_${SIM_NAME}/${RUN_NAME}"
SLICE_LIST="${BASE_DIR}/sfunctor_results/slice_list_${SIM_NAME}.txt"
SLICE_DIR="${BASE_DIR}/sfunctor_results/slice_${SIM_NAME}"
[ ! -d "$SLICE_DIR" ] && SLICE_DIR="${BASE_DIR}/slice_${SIM_NAME}"

log "============================================================"
log "Starting distributed structure-function analysis (Andes)"
log "Job ID: ${SLURM_JOB_ID}"
log "Nodes requested: ${SLURM_JOB_NUM_NODES}"
log "CPUs per task: ${SLURM_CPUS_PER_TASK}"
log "Working directory: ${WORK_DIR}"
log "Slice list: ${SLICE_LIST}"
log "============================================================"

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

if [ ! -d "$SLICE_DIR" ]; then
    echo "ERROR: slice directory not found: $SLICE_DIR"
    exit 1
fi

REFRESH_SLICE_LIST="${REFRESH_SLICE_LIST:-0}"
REFRESH=0
if [ ! -f "$SLICE_LIST" ]; then
    log "Slice list not found; generating from ${SLICE_DIR}"
    REFRESH=1
else
    DIR_COUNT=$(ls "$SLICE_DIR"/*.npz 2>/dev/null | wc -l)
    LIST_COUNT=$(wc -l < "$SLICE_LIST")
    if [ "$DIR_COUNT" -ne "$LIST_COUNT" ]; then
        log "Slice list count (${LIST_COUNT}) differs from slice dir count (${DIR_COUNT}); regenerating."
        REFRESH=1
    elif [ "$REFRESH_SLICE_LIST" = "1" ]; then
        log "REFRESH_SLICE_LIST=1; regenerating slice list from ${SLICE_DIR}"
        REFRESH=1
    else
        log "Using existing slice list: ${SLICE_LIST}"
    fi
fi

if [ "$REFRESH" -eq 1 ]; then
    DIR_COUNT=$(ls "$SLICE_DIR"/*.npz 2>/dev/null | wc -l)
    if [ "$DIR_COUNT" -eq 0 ]; then
        echo "ERROR: no slices found in $SLICE_DIR"
        exit 1
    fi
    ls "$SLICE_DIR"/*.npz > "$SLICE_LIST"
fi

TOTAL_SLICES=$(wc -l < "$SLICE_LIST")
log "Total slices found: ${TOTAL_SLICES}"

if [ ! -f "$WORK_DIR/displacements.npz" ]; then
    DISPL_START=$(date +%s)
    log "Generating displacements (N_disp=${N_DISP_TOTAL}, N_ell=${N_ELL_BINS}, Nres=${NRES})"
    python "$SFUNCTOR_DIR/scripts/production/generate_displacements.py" \
        --n_disp_total "$N_DISP_TOTAL" \
        --n_ell_bins "$N_ELL_BINS" \
        --stencil_width "$STENCIL_WIDTH" \
        --Nres "$NRES" \
        --seed "$SEED"
    DISPL_END=$(date +%s)
    log_duration "$DISPL_START" "$DISPL_END" "Displacement generation"
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
    SLICE_START=$(date +%s)
    log "-----"
    log "Processing slice ${SLICE_NAME} ($((SLICE_IDX + 1))/${TOTAL_SLICES})"
    SLICE_OUTPUT_DIR="$WORK_DIR/${SLICE_NAME}_histograms"
    mkdir -p "$SLICE_OUTPUT_DIR"

    for NODE_ID in $(seq 0 $((TOTAL_NODES - 1))); do
        NODE=${NODES[$NODE_ID]}
        log "Launching node ${NODE_ID}/${TOTAL_NODES} on ${NODE} for slice ${SLICE_NAME}"
        srun --exclusive -N 1 -n 1 -w "$NODE" --cpus-per-task="${CPUS_PER_TASK}" \
            python "$SFUNCTOR_DIR/scripts/production/run_node_analysis.py" \
                --slice "$SLICE_PATH" \
                --displacements "$WORK_DIR/displacements.npz" \
                --node_id "$NODE_ID" \
                --total_nodes "$TOTAL_NODES" \
                --output_dir "$SLICE_OUTPUT_DIR" \
	                --stride "$STRIDE" \
	                --N_random_subsamples "$N_RANDOM_SUBSAMPLES" \
	                --stencil_width "$STENCIL_WIDTH" \
	                --n_processes "${CPUS_PER_TASK}" \
                    "${EXTRA_BINS_ARGS[@]}" \
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
        --workers "${CPUS_PER_TASK}"

    log "Removing temporary histogram directory for slice ${SLICE_NAME}"
    rm -rf "$SLICE_OUTPUT_DIR"

    SLICE_END=$(date +%s)
    log_duration "$SLICE_START" "$SLICE_END" "Slice ${SLICE_NAME}"
done

log "Combining slice-level histograms into all-slice result"
python "$SFUNCTOR_DIR/scripts/production/combine_histograms_fast.py" \
    --pattern "$WORK_DIR/sf_results_${SIM_NAME}_axis*.npz" \
    --output "$WORK_DIR/sf_results_all_slices.npz" \
    --mode slice \
    --workers "${CPUS_PER_TASK}"

log "Generating diagnostic plots"
python "$SFUNCTOR_DIR/plotting_scripts/plot_structure_functions.py" \
    "$WORK_DIR/sf_results_all_slices.npz"

SCRIPT_END=$(date +%s)
log_duration "$SCRIPT_START" "$SCRIPT_END" "Total runtime"
log "All processing complete"
