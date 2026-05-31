#!/usr/bin/env bash
#SBATCH -A AST207
#SBATCH -J phase4_batch_a
#SBATCH -o logs/phase4_batch_a_%x_%j.out
#SBATCH -e logs/phase4_batch_a_%x_%j.err
#SBATCH -t 02:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

set -euo pipefail

SFUNCTOR_DIR="${SFUNCTOR_DIR:-/ccs/home/dfielding/SFunctor}"
: "${PHASE2_ROOT:?PHASE2_ROOT must be an explicit Phase 4 extraction directory}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be an explicit unique Phase 4 Batch A directory}"
: "${RUN_DIR:?RUN_DIR must be an explicit unique allocation directory}"
: "${ACTION:?ACTION must be plan, work, reduce, verify, or summarize}"
CLI="${SFUNCTOR_DIR}/scripts/phase4/run_phase4_batch_a_sampler.py"
PYTHON_BIN="${SFUNCTOR_DIR}/venv_sfunctor/bin/python"
PHASE4_WORKERS="${PHASE4_WORKERS:-1}"

cd "${SFUNCTOR_DIR}"
mkdir -p logs "${RUN_DIR}/logs" "$(dirname -- "${OUTPUT_ROOT}")"
LOCK_DIR="$(dirname -- "${OUTPUT_ROOT}")/.$(basename -- "${OUTPUT_ROOT}").phase4_batch_a_action_lock"
LOCK_OWNER="${LOCK_DIR}/owner.txt"
LOCK_RECOVERY_CLAIM="${LOCK_DIR}/.recovery_claim"
LOCK_TOKEN="${SLURM_JOB_ID:-unknown}.$$.$RANDOM.$RANDOM"

release_lock() {
  local owner_token

  if [[ ! -e "${LOCK_OWNER}" ]]; then
    rmdir "${LOCK_DIR}" 2>/dev/null || true
    return
  fi
  if ! ln "${LOCK_OWNER}" "${LOCK_RECOVERY_CLAIM}" 2>/dev/null; then
    return
  fi
  owner_token="$(sed -n 's/^token=//p' "${LOCK_RECOVERY_CLAIM}")"
  if [[ "${owner_token}" != "${LOCK_TOKEN}" ]] || ! [[ "${LOCK_OWNER}" -ef "${LOCK_RECOVERY_CLAIM}" ]]; then
    rm -f "${LOCK_RECOVERY_CLAIM}"
    return
  fi
  rm -f "${LOCK_OWNER}" "${LOCK_RECOVERY_CLAIM}"
  rmdir "${LOCK_DIR}" 2>/dev/null || true
}

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
  release_lock
  exit "${status}"
}

recover_stale_lock() {
  local owner_job_id queue_rows queued_job_id unexpected_entry

  if ! ln "${LOCK_OWNER}" "${LOCK_RECOVERY_CLAIM}" 2>/dev/null; then
    return 1
  fi
  if ! [[ "${LOCK_OWNER}" -ef "${LOCK_RECOVERY_CLAIM}" ]]; then
    rm -f "${LOCK_RECOVERY_CLAIM}"
    return 1
  fi
  owner_job_id="$(sed -n 's/^job_id=//p' "${LOCK_RECOVERY_CLAIM}")"
  if [[ ! "${owner_job_id}" =~ ^[0-9]+$ ]]; then
    rm -f "${LOCK_RECOVERY_CLAIM}"
    return 1
  fi
  if ! queue_rows="$(squeue -h -o '%A' 2>/dev/null)"; then
    rm -f "${LOCK_RECOVERY_CLAIM}"
    return 1
  fi
  while IFS= read -r queued_job_id; do
    if [[ -z "${queued_job_id}" ]]; then
      continue
    fi
    if [[ ! "${queued_job_id}" =~ ^[0-9]+$ ]] || [[ "${queued_job_id}" == "${owner_job_id}" ]]; then
      rm -f "${LOCK_RECOVERY_CLAIM}"
      return 1
    fi
  done <<< "${queue_rows}"
  if ! unexpected_entry="$(find "${LOCK_DIR}" -mindepth 1 -maxdepth 1 \
      ! -name owner.txt ! -name .recovery_claim -print -quit)"; then
    rm -f "${LOCK_RECOVERY_CLAIM}"
    return 1
  fi
  if [[ -n "${unexpected_entry}" ]] || ! [[ "${LOCK_OWNER}" -ef "${LOCK_RECOVERY_CLAIM}" ]]; then
    rm -f "${LOCK_RECOVERY_CLAIM}"
    return 1
  fi
  rm -f "${LOCK_OWNER}" "${LOCK_RECOVERY_CLAIM}"
  if ! rmdir "${LOCK_DIR}" 2>/dev/null; then
    return 1
  fi
  echo "Recovered stale Phase 4 Batch A action lock for inactive Slurm job ${owner_job_id}: ${LOCK_DIR}" >&2
}

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  if ! recover_stale_lock || ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    echo "ERROR: another Phase 4 Batch A action holds ${LOCK_DIR}" >&2
    exit 3
  fi
fi
trap finalize EXIT
printf 'job_id=%s\naction=%s\nrun_dir=%s\ncreated_utc=%s\ntoken=%s\n' \
  "${SLURM_JOB_ID:-unknown}" "${ACTION}" "${RUN_DIR}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${LOCK_TOKEN}" \
  > "${LOCK_OWNER}"

module reset
module load gcc/9.3.0 python/.3.11-anaconda3
source "${SFUNCTOR_DIR}/venv_sfunctor/bin/activate"
export PYTHONPATH="${SFUNCTOR_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

run_python() {
  /usr/bin/time -v srun -N 1 -n 1 --cpu-bind=cores "$PYTHON_BIN" "${CLI}" "$@"
}

case "${ACTION}" in
  work)
    /usr/bin/time -v srun --ntasks="${SLURM_JOB_NUM_NODES:-1}" --ntasks-per-node=1 \
      --cpus-per-task="${SLURM_CPUS_PER_TASK:-32}" --cpu-bind=cores \
      "$PYTHON_BIN" "${CLI}" work --phase2-root "${PHASE2_ROOT}" --output-root "${OUTPUT_ROOT}" \
      --workers "${PHASE4_WORKERS}" > "${RUN_DIR}/logs/work.log" 2>&1
    ;;
  plan|reduce|verify|summarize)
    run_python "${ACTION}" --phase2-root "${PHASE2_ROOT}" --output-root "${OUTPUT_ROOT}" \
      --workers "${PHASE4_WORKERS}" > "${RUN_DIR}/logs/${ACTION}.log" 2>&1
    ;;
  *)
    echo "unsupported ACTION=${ACTION}; use plan, work, reduce, verify, or summarize" >&2
    exit 2
    ;;
esac
