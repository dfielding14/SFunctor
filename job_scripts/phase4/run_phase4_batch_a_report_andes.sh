#!/usr/bin/env bash
#SBATCH -A AST207
#SBATCH -J phase4_batch_a_report
#SBATCH -o logs/phase4_batch_a_report_%x_%j.out
#SBATCH -e logs/phase4_batch_a_report_%x_%j.err
#SBATCH -t 02:30:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

set -euo pipefail

SFUNCTOR_DIR="${SFUNCTOR_DIR:-/ccs/home/dfielding/SFunctor}"
PHASE1_ROOT="${PHASE1_ROOT:-/lustre/orion/ast207/proj-shared/dfielding/Production_plm/prompt1_catalog/t6_final_primary_20260530}"
LEDGER_SUMMARY="${LEDGER_SUMMARY:-}"
: "${EXTRACTION_ROOT:?EXTRACTION_ROOT must be the retained Phase 4 extraction directory}"
: "${RELEASE_ROOT:?RELEASE_ROOT must be the retained Phase 4 Batch A release directory}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be an explicit unique report directory}"
: "${RUN_DIR:?RUN_DIR must be an explicit unique allocation directory}"
: "${LEDGER_SUMMARY:?LEDGER_SUMMARY must be an archived zero-pending compute-budget summary}"
ACTION="${ACTION:-report}"
CLI="${SFUNCTOR_DIR}/scripts/phase4/generate_phase4_batch_a_status_figures.py"
PROFILE_CLI="${SFUNCTOR_DIR}/scripts/phase4/profile_phase4_batch_a_report_io.py"
PREFLIGHT_CLI="${SFUNCTOR_DIR}/scripts/phase4/preflight_phase4_batch_a_report.py"
PYTHON_BIN="${SFUNCTOR_DIR}/venv_sfunctor/bin/python"

cd "${SFUNCTOR_DIR}"
if [[ -e "${RUN_DIR}" ]]; then
  echo "ERROR: report allocation directory already exists: ${RUN_DIR}" >&2
  exit 3
fi
mkdir -p logs "${RUN_DIR}/logs" "$(dirname -- "${OUTPUT_DIR}")"
if [[ "${ACTION}" == "report" && -e "${OUTPUT_DIR}" ]]; then
  echo "ERROR: report output already exists: ${OUTPUT_DIR}" >&2
  exit 3
fi

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

case "${ACTION}" in
  preflight)
    /usr/bin/time -v srun -N 1 -n 1 --cpu-bind=cores "$PYTHON_BIN" "${PREFLIGHT_CLI}" \
      --phase1-root "${PHASE1_ROOT}" \
      --extraction-root "${EXTRACTION_ROOT}" \
      --release-root "${RELEASE_ROOT}" \
      --ledger-summary "${LEDGER_SUMMARY}" \
      --output-json "${RUN_DIR}/phase4_batch_a_report_structural_preflight.json" \
      > "${RUN_DIR}/logs/preflight.log" 2>&1
    ;;
  profile)
    : "${PROFILE_CUBE_ID:?PROFILE_CUBE_ID must be one retained Phase 4 cube for ACTION=profile}"
    /usr/bin/time -v srun -N 1 -n 1 --cpu-bind=cores "$PYTHON_BIN" "${PROFILE_CLI}" \
      --extraction-root "${EXTRACTION_ROOT}" \
      --cube-id "${PROFILE_CUBE_ID}" \
      --ledger-summary "${LEDGER_SUMMARY}" \
      --output-json "${RUN_DIR}/phase4_batch_a_report_io_profile.json" \
      > "${RUN_DIR}/logs/profile.log" 2>&1
    ;;
  report)
    /usr/bin/time -v srun -N 1 -n 1 --cpu-bind=cores "$PYTHON_BIN" "${CLI}" \
      --phase1-root "${PHASE1_ROOT}" \
      --extraction-root "${EXTRACTION_ROOT}" \
      --release-root "${RELEASE_ROOT}" \
      --ledger-summary "${LEDGER_SUMMARY}" \
      --output-dir "${OUTPUT_DIR}" \
      > "${RUN_DIR}/logs/report.log" 2>&1
    ;;
  *)
    echo "unsupported ACTION=${ACTION}; use preflight, profile, or report" >&2
    exit 2
    ;;
esac
