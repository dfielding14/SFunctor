#!/bin/bash
#SBATCH -A AST207
#SBATCH -J slice_extract
#SBATCH -o slice_extract_%x_%j.out
#SBATCH -e slice_extract_%x_%j.err
#SBATCH -t 2:00:00
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

# Generic slice extraction for periodic boxes.
# Usage:
#   sbatch --wait --job-name="slice_${SIM_NAME}" --export=ALL,SIM_NAME=...,FILE_NUMBER=-1 job_scripts/production/run_slice_extraction_frontier.sh
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
    if [[ "$SIM_NAME" == *"beta1"* ]]; then
        ELONG_Z_FACTOR=3
    else
        ELONG_Z_FACTOR=1
    fi
fi

module reset
module load PrgEnv-gnu
source /autofs/nccs-svm1_home2/dfielding/SFunctor/venv_frontier/bin/activate

SFUNCTOR_DIR="/autofs/nccs-svm1_home2/dfielding/SFunctor"
BASE_DIR="/lustre/orion/ast207/proj-shared/dfielding/Production_plm"
export PYTHONPATH="${SFUNCTOR_DIR}:${PYTHONPATH:-}"

# Positions (fraction of box length) for axes 1/2; axis 3 scales by ELONG_Z_FACTOR if elongated.
POS_LIST=(-0.375 -0.25 -0.125 0.0 0.125 0.25 0.375 0.5)

log "Starting slice extraction for ${SIM_NAME}"
log "File number: ${FILE_NUMBER}"
log "Elongation factor (axis 3): ${ELONG_Z_FACTOR}"

cd "$BASE_DIR"

SLICE_DIR="${BASE_DIR}/sfunctor_results/slice_${SIM_NAME}"
if [ -d "$SLICE_DIR" ]; then
    log "Existing slices in ${SLICE_DIR}:"
    ls "${SLICE_DIR}/${SIM_NAME}_axis"*".npz" 2>/dev/null || true
else
    log "No existing slice directory at ${SLICE_DIR}"
fi

log "Planned slice positions:"
log "  Axes 1/2: ${POS_LIST[*]}"
log "  Axis 3 (scaled):"
for p in "${POS_LIST[@]}"; do
    scaled=$(python - <<PY
pos = float("${p}")
factor = float("${ELONG_Z_FACTOR}")
print(pos * factor)
PY
)
    log "    ${scaled}"
done

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
    # Check for existing cached slice (any file index)
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

# Axes 1 and 2 use POS_LIST directly
for axis in 1 2; do
    for pos in "${POS_LIST[@]}"; do
        run_slice "${axis}" "${pos}"
    done
done

# Axis 3 optionally elongated
for pos in "${POS_LIST[@]}"; do
    scaled_pos=$(python - <<PY
pos = float("${pos}")
factor = float("${ELONG_Z_FACTOR}")
print(pos * factor)
PY
)
    run_slice 3 "${scaled_pos}"
done

log "Slice extraction complete for ${SIM_NAME}"
