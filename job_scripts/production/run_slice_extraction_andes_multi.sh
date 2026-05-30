#!/bin/bash
#SBATCH -A AST207
#SBATCH -J slice_extract_andes_multi
#SBATCH -o slice_extract_%x_%j.out
#SBATCH -e slice_extract_%x_%j.err
#SBATCH -t 48:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

# Multi-node slice extraction for Andes/frontier: each task handles a subset of slices independently.
# Only the module/venv paths change based on host detection.
# Usage:
#   sbatch -N <nodes> --job-name="slice_${SIM_NAME}" \
#     --export=ALL,SIM_NAME=...,FILE_NUMBER=-1 job_scripts/production/run_slice_extraction_andes_multi.sh
# Optional env: ELONG_Z_FACTOR (defaults to 1, set to 3 for beta=1 elongated z)

set -euo pipefail

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

if [[ -z "${SIM_NAME:-}" ]]; then
    echo "ERROR: SIM_NAME is required (e.g., Turb_640_beta25_dedt025_plm)"
    exit 1
fi

FILE_NUMBER=${FILE_NUMBER:-"-1"}      # -1 means use final file
ELONG_Z_FACTOR=${ELONG_Z_FACTOR:-""}
if [[ -z "$ELONG_Z_FACTOR" ]]; then
    # Only treat truly elongated beta=1 runs as 3x; avoid matching beta100/10
    if [[ "$SIM_NAME" == *"beta1_"* ]]; then
        ELONG_Z_FACTOR=3
    else
        ELONG_Z_FACTOR=1
    fi
fi

module reset
module load gcc/9.3.0 python/.3.11-anaconda3
source /ccs/home/dfielding/SFunctor/venv_sfunctor/bin/activate

SFUNCTOR_DIR="/ccs/home/dfielding/SFunctor"
BASE_DIR="/lustre/orion/ast207/proj-shared/dfielding/Production_plm"
export PYTHONPATH="${SFUNCTOR_DIR}:${PYTHONPATH:-}"

# Positions (fraction of box length) for axes 1/2; axis 3 scales by ELONG_Z_FACTOR if elongated.
# Added higher-resolution near-origin slices: 1/128, 1/64, 1/32, 1/16.
POS_LIST_STR="-0.375 -0.25 -0.125 0.0 0.0078125 0.015625 0.03125 0.0625 0.125 0.25 0.375 0.5"

log "Starting slice extraction for ${SIM_NAME} (Andes multi)"
log "File number: ${FILE_NUMBER}"
log "Elongation factor (axis 3): ${ELONG_Z_FACTOR}"

cd "$BASE_DIR"

SLICE_DIR="${BASE_DIR}/sfunctor_results/slice_${SIM_NAME}"
initial_count=$(find "${SLICE_DIR}" -maxdepth 1 -name "${SIM_NAME}_axis*.npz" 2>/dev/null | wc -l || true)
if [ -d "$SLICE_DIR" ]; then
    log "Existing slices in ${SLICE_DIR}:"
    ls "${SLICE_DIR}/${SIM_NAME}_axis"*".npz" 2>/dev/null || true
else
    log "No existing slice directory at ${SLICE_DIR}; creating it"
    mkdir -p "${SLICE_DIR}"
fi

log "Planned slice positions:"
log "  Axes 1/2: ${POS_LIST_STR}"
log "  Axis 3 (scaled):"
for p in ${POS_LIST_STR}; do
    scaled=$(python - <<PY
pos = float("${p}")
factor = float("${ELONG_Z_FACTOR}")
print(pos * factor)
PY
)
    log "    ${scaled}"
done

# Build list of slice combos and pre-filter those already cached to balance work
WORK_FILE="${SLURM_SUBMIT_DIR:-$(pwd)}/slice_work_${SLURM_JOB_ID:-$$}.lst"
rm -f "${WORK_FILE}"
touch "${WORK_FILE}"

pos_to_str() {
    python - <<PY
v = float("$1")
s = f"{v:.8f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")
print(s or "0")
PY
}

TOTAL_COMBOS=0
SKIP_COMBOS=0
PENDING_COMBOS=0

for pos in ${POS_LIST_STR}; do
    for axis in 1 2; do
        TOTAL_COMBOS=$((TOTAL_COMBOS + 1))
        pos_str=$(pos_to_str "${pos}")
        target="${SLICE_DIR}/${SIM_NAME}_axis${axis}_slice${pos_str}_file"
        if ls ${target}*.npz 1>/dev/null 2>&1; then
            SKIP_COMBOS=$((SKIP_COMBOS + 1))
            continue
        fi
        echo "${axis} ${pos}" >> "${WORK_FILE}"
        PENDING_COMBOS=$((PENDING_COMBOS + 1))
    done
done
for pos in ${POS_LIST_STR}; do
    scaled=$(python - <<PY
pos = float("${pos}")
factor = float("${ELONG_Z_FACTOR}")
print(pos * factor)
PY
)
    TOTAL_COMBOS=$((TOTAL_COMBOS + 1))
    pos_str=$(pos_to_str "${scaled}")
    target="${SLICE_DIR}/${SIM_NAME}_axis3_slice${pos_str}_file"
    if ls ${target}*.npz 1>/dev/null 2>&1; then
        SKIP_COMBOS=$((SKIP_COMBOS + 1))
        continue
    fi
    echo "3 ${scaled}" >> "${WORK_FILE}"
    PENDING_COMBOS=$((PENDING_COMBOS + 1))
done

log "Work distribution: total combos=${TOTAL_COMBOS}, cached/skipped=${SKIP_COMBOS}, pending=${PENDING_COMBOS}"
if [[ ${PENDING_COMBOS} -eq 0 ]]; then
    log "No pending slices to extract; exiting."
    exit 0
fi

NTASKS=${SLURM_NTASKS:-${SLURM_JOB_NUM_NODES:-1}}

# Launch one task per node; each task handles a subset of pending slices
srun --ntasks=${NTASKS} --ntasks-per-node=1 bash -lc '
set -euo pipefail
log() { echo "[$(date '"+'"'%Y-%m-%d %H:%M:%S'"'"') ${SLURM_PROCID}] $*"; }

SIM_NAME="'"${SIM_NAME}"'"
FILE_NUMBER="'"${FILE_NUMBER}"'"
ELONG_Z_FACTOR="'"${ELONG_Z_FACTOR}"'"
SFUNCTOR_DIR="'"${SFUNCTOR_DIR}"'"
BASE_DIR="'"${BASE_DIR}"'"
SLICE_DIR="'"${SLICE_DIR}"'"
POS_LIST_STR="'"${POS_LIST_STR}"'"
PYTHONPATH="'"${PYTHONPATH}"'"
WORK_FILE="'"${WORK_FILE}"'"
export PYTHONPATH

cd "'"${BASE_DIR}"'"

module reset
module load gcc/9.3.0 python/.3.11-anaconda3
source /ccs/home/dfielding/SFunctor/venv_sfunctor/bin/activate

mapfile -t COMBOS < "$WORK_FILE"

log "Assigned slice tasks (total ${#COMBOS[@]} combos, task ${SLURM_PROCID}/${SLURM_NTASKS}):"
for idx in "${!COMBOS[@]}"; do
    if (( idx % SLURM_NTASKS != SLURM_PROCID )); then
        continue
    fi
    log "  ${COMBOS[idx]}"
done
log "Task ${SLURM_PROCID} starting slice work"

run_slice() {
    local axis="$1"
    local pos="$2"
    local pos_str
    pos_str=$(python - <<PY
v = float("${pos}")
s = f"{v:.8f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")
print(s or "0")
PY
)
    local target="${SLICE_DIR}/${SIM_NAME}_axis${axis}_slice${pos_str}_file"
    if ls ${target}*.npz 1>/dev/null 2>&1; then
        log "Skipping axis=${axis} pos=${pos} (cache exists: ${target}*.npz)"
        return
    fi
    log "Extracting axis=${axis} pos=${pos}"
    python "${SFUNCTOR_DIR}/scripts/production/extractor.py" \
        --sim_name "${SIM_NAME}" \
        --axis "${axis}" \
        --position "${pos}" \
        --file_number "${FILE_NUMBER}"
}

for idx in "${!COMBOS[@]}"; do
    if (( idx % SLURM_NTASKS != SLURM_PROCID )); then
        continue
    fi
    axis=${COMBOS[idx]% *}
    pos=${COMBOS[idx]#* }
    run_slice "$axis" "$pos"
done

log "Task ${SLURM_PROCID} slice work complete"
'

log "All slice tasks dispatched"
final_count=$(find "${SLICE_DIR}" -maxdepth 1 -name "${SIM_NAME}_axis*.npz" 2>/dev/null | wc -l || true)
new_created=$(( final_count - initial_count ))
log "Slice files before: ${initial_count}, after: ${final_count}, created this run: ${new_created}"
