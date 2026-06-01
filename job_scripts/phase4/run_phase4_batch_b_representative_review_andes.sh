#!/usr/bin/env bash
#SBATCH -A AST207
#SBATCH -J phase4_batch_b_review
#SBATCH -o logs/phase4_batch_b_review_%x_%j.out
#SBATCH -e logs/phase4_batch_b_review_%x_%j.err
#SBATCH -t 00:20:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

set -euo pipefail

SFUNCTOR_DIR="${SFUNCTOR_DIR:-/ccs/home/dfielding/SFunctor}"
: "${RELEASE_ROOT:?RELEASE_ROOT must be the retained Phase 4 Batch B representative release}"
: "${LEDGER_SUMMARY:?LEDGER_SUMMARY must be an archived zero-pending compute-budget summary}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be an explicit unique report directory}"
: "${RUN_DIR:?RUN_DIR must be an explicit unique allocation directory}"
CLI="${SFUNCTOR_DIR}/scripts/phase4/generate_phase4_batch_b_representative_review.py"
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
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

/usr/bin/time -v srun -N 1 -n 1 --cpu-bind=cores "$PYTHON_BIN" "${CLI}" \
  --release-root "${RELEASE_ROOT}" \
  --ledger-summary "${LEDGER_SUMMARY}" \
  --output-dir "${OUTPUT_DIR}" \
  > "${RUN_DIR}/logs/report.log" 2>&1
