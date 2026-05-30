#!/bin/bash
#SBATCH -A AST207
#SBATCH -J sf_generic
#SBATCH -o logs/sf_distributed_%j.out
#SBATCH -e logs/sf_distributed_%j.err
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH --qos debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56

set -euo pipefail

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

: "${SIM_NAME:?SIM_NAME is required}"
: "${STENCIL_WIDTH:?STENCIL_WIDTH is required}"
: "${N_DISP_TOTAL:?N_DISP_TOTAL is required}"
: "${N_RANDOM_SUBSAMPLES:?N_RANDOM_SUBSAMPLES is required}"
N_ELL_BINS="${N_ELL_BINS:-96}"
STRIDE="${STRIDE:-1}"

# Detect Nres from SIM_NAME (expects pattern Turb_<N>_)
if [[ "$SIM_NAME" =~ Turb_([0-9]+)_ ]]; then
    NRES="${BASH_REMATCH[1]}"
else
    echo "ERROR: could not parse resolution from SIM_NAME=${SIM_NAME}"
    exit 1
fi

# Derive node count if not provided and not overridden by sbatch -N
if [[ -n "${NODES:-}" ]]; then
    TOTAL_NODES="${NODES}"
elif [[ -n "${SLURM_JOB_NUM_NODES:-}" ]]; then
    TOTAL_NODES="${SLURM_JOB_NUM_NODES}"
else
    case "$NRES" in
        10240) TOTAL_NODES=128 ;;
        5120)  TOTAL_NODES=64 ;;
        2560)  TOTAL_NODES=32 ;;
        1280)  TOTAL_NODES=16 ;;
        640)   TOTAL_NODES=2 ;;
        *)     TOTAL_NODES=2 ;;
    esac
fi

module reset
module load PrgEnv-gnu
source /autofs/nccs-svm1_home2/dfielding/SFunctor/venv_frontier/bin/activate
export SFUNCTOR_DIR="/autofs/nccs-svm1_home2/dfielding/SFunctor"
export PYTHONPATH="${SFUNCTOR_DIR}:${PYTHONPATH:-}"

BASE_DIR="/lustre/orion/ast207/proj-shared/dfielding/Production_plm"

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

RUN_NAME="ndisp${N_DISP_TOTAL}_nrand${N_RANDOM_SUBSAMPLES}_nell${N_ELL_BINS}_sw${STENCIL_WIDTH}_job${SLURM_JOB_ID:-manual}"
WORK_DIR="${BASE_DIR}/sfunctor_results/results_${SIM_NAME}/${RUN_NAME}"
SLICE_LIST="${BASE_DIR}/sfunctor_results/slice_list_${SIM_NAME}.txt"

log "============================================================"
log "Starting distributed analysis"
log "SIM: ${SIM_NAME}  Nres=${NRES}  SW=${STENCIL_WIDTH}"
log "Ndisp=${N_DISP_TOTAL}  Nrand=${N_RANDOM_SUBSAMPLES}  Nell=${N_ELL_BINS}"
log "Nodes: ${TOTAL_NODES} (override with sbatch -N or NODES=)"
log "Working dir: ${WORK_DIR}"
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
        --seed 42
    log "Finished generating displacements"
else
    log "Reusing existing displacement file at $WORK_DIR/displacements.npz"
fi

NODES=($(scontrol show hostname $SLURM_JOB_NODELIST))
TOTAL_NODES_ACTUAL=${#NODES[@]}
if [[ $TOTAL_NODES_ACTUAL -ne $TOTAL_NODES ]]; then
    log "Warning: requested ${TOTAL_NODES} nodes but scheduler provided ${TOTAL_NODES_ACTUAL}"
    TOTAL_NODES=$TOTAL_NODES_ACTUAL
fi
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
