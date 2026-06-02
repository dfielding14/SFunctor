#!/usr/bin/env bash
#SBATCH -A AST207
#SBATCH -J phase4_completion_supplement
#SBATCH -o logs/phase4_completion_supplement_%x_%j.out
#SBATCH -e logs/phase4_completion_supplement_%x_%j.err
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

set -euo pipefail

SFUNCTOR_DIR="${SFUNCTOR_DIR:-/ccs/home/dfielding/SFunctor}"
: "${PHASE1_ROOT:?PHASE1_ROOT must be the trusted Phase 1 catalog release}"
: "${EXTRACTION_ROOT:?EXTRACTION_ROOT must be the immutable Phase 4 extraction release}"
: "${BATCH_A_ROOT:?BATCH_A_ROOT must be the immutable Phase 4 Batch A release}"
: "${ALL21_3POINT_EXTENSION_ROOT:?ALL21_3POINT_EXTENSION_ROOT must be the immutable all-21 labeled 3-point release}"
: "${ALL21_BATCH_B_ROOT:?ALL21_BATCH_B_ROOT must be the immutable all-21 Batch B release}"
: "${LEDGER_SUMMARY:?LEDGER_SUMMARY must be the settled zero-pending compute-budget summary}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be an explicit unique report directory}"
: "${RUN_DIR:?RUN_DIR must be an explicit unique allocation directory}"
CLI="${SFUNCTOR_DIR}/scripts/phase4/generate_phase4_completion_supplement.py"
PYTHON_BIN="${SFUNCTOR_DIR}/venv_sfunctor/bin/python"

cd "${SFUNCTOR_DIR}"
if [[ -e "${RUN_DIR}" ]]; then
  echo "ERROR: report allocation directory already exists: ${RUN_DIR}" >&2
  exit 3
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "ERROR: report output already exists: ${OUTPUT_DIR}" >&2
  exit 3
fi
mkdir -p logs "${RUN_DIR}/logs" "$(dirname -- "${OUTPUT_DIR}")"

archive_slurm_resources() {
  local destination

  if [[ ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || ! command -v sacct >/dev/null 2>&1; then
    return
  fi
  mkdir -p "${RUN_DIR}/resources"
  destination="${RUN_DIR}/resources/sacct_${SLURM_JOB_ID}.psv"
  sacct -j "${SLURM_JOB_ID}" \
    --format=JobID,ElapsedRaw,NNodes,AllocCPUS,MaxRSS,MaxRSSNode,MaxRSSTask,AveRSS,MaxVMSize,State,ExitCode \
    -n -P > "${destination}.tmp" || return
  mv "${destination}.tmp" "${destination}"
}

finalize() {
  local status=$?

  trap - EXIT
  archive_slurm_resources || true
  exit "${status}"
}
trap finalize EXIT

module reset
module load gcc/9.3.0 python/.3.11-anaconda3
source "${SFUNCTOR_DIR}/venv_sfunctor/bin/activate"
export PYTHONPATH="${SFUNCTOR_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

/usr/bin/time -v srun -N 1 -n 1 --cpu-bind=cores "$PYTHON_BIN" "${CLI}" \
  --phase1-root "${PHASE1_ROOT}" \
  --extraction-root "${EXTRACTION_ROOT}" \
  --batch-a-root "${BATCH_A_ROOT}" \
  --all21-3point-extension-root "${ALL21_3POINT_EXTENSION_ROOT}" \
  --all21-batch-b-root "${ALL21_BATCH_B_ROOT}" \
  --ledger-summary "${LEDGER_SUMMARY}" \
  --output-dir "${OUTPUT_DIR}" \
  > "${RUN_DIR}/logs/report.log" 2>&1
