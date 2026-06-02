#!/usr/bin/env bash
#SBATCH -A AST207
#SBATCH -J phase5_cross_scale_report
#SBATCH -o logs/phase5_cross_scale_report_%x_%j.out
#SBATCH -e logs/phase5_cross_scale_report_%x_%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

set -euo pipefail

SFUNCTOR_DIR="${SFUNCTOR_DIR:-/ccs/home/dfielding/SFunctor}"
: "${CAMPAIGN_CONFIG:?CAMPAIGN_CONFIG must be the approved Phase 5 report config}"
: "${PHASE5_DECISION_RECORD:?PHASE5_DECISION_RECORD must bind the retained Phase 5 report release matrix}"
: "${PHASE1_ROOT:?PHASE1_ROOT must be the retained Phase 1 catalog root}"
: "${PHASE5_RELEASES:?PHASE5_RELEASES must contain whitespace-separated label=path retained releases}"
: "${LEDGER_SUMMARY:?LEDGER_SUMMARY must be the settled zero-pending compute-budget summary}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be an explicit unique report directory}"
: "${RUN_DIR:?RUN_DIR must be an explicit unique allocation directory}"
CLI="${SFUNCTOR_DIR}/scripts/phase5/generate_phase5_status_figures.py"
PYTHON_BIN="${SFUNCTOR_DIR}/venv_sfunctor/bin/python"

cd "${SFUNCTOR_DIR}"
if [[ -e "${RUN_DIR}" ]]; then
  echo "ERROR: report allocation directory already exists: ${RUN_DIR}" >&2
  exit 3
fi
if [[ -e "${OUTPUT_DIR}" ]] && {
  [[ ! -d "${OUTPUT_DIR}" ]] ||
    [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]
}; then
  echo "ERROR: report output is not an empty directory: ${OUTPUT_DIR}" >&2
  exit 3
fi
mkdir -p logs "$(dirname -- "${OUTPUT_DIR}")"
if ! mkdir "${RUN_DIR}"; then
  echo "ERROR: report allocation directory appeared concurrently: ${RUN_DIR}" >&2
  exit 3
fi
mkdir "${RUN_DIR}/logs"

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

read -r -a release_specs <<< "${PHASE5_RELEASES}"
release_args=()
for release_spec in "${release_specs[@]}"; do
  release_args+=(--release "${release_spec}")
done
optional_args=()
if [[ -n "${PHASE4_BATCH_A_ROOT:-}" ]]; then
  optional_args+=(--phase4-batch-a-root "${PHASE4_BATCH_A_ROOT}")
fi

/usr/bin/time -v srun -N 1 -n 1 --cpu-bind=cores "$PYTHON_BIN" "${CLI}" \
  --campaign-config "${CAMPAIGN_CONFIG}" \
  --decision-record "${PHASE5_DECISION_RECORD}" \
  --phase1-root "${PHASE1_ROOT}" \
  "${release_args[@]}" \
  --ledger-summary "${LEDGER_SUMMARY}" \
  --output-dir "${OUTPUT_DIR}" \
  "${optional_args[@]}" \
  > "${RUN_DIR}/logs/report.log" 2>&1
