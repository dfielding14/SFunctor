#!/usr/bin/env bash
#SBATCH -A AST207
#SBATCH -J phase5_extract
#SBATCH -o logs/phase5_extract_%x_%j.out
#SBATCH -e logs/phase5_extract_%x_%j.err
#SBATCH -t 02:30:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

set -euo pipefail

SFUNCTOR_DIR="${SFUNCTOR_DIR:-/ccs/home/dfielding/SFunctor}"
TRUSTED_RUN="${TRUSTED_RUN:-/lustre/orion/ast207/proj-shared/dfielding/Production_plm/prompt1_catalog/t6_final_primary_20260530}"
DATA_ROOT="${DATA_ROOT:-/lustre/orion/ast207/proj-shared/dfielding/Production_plm/data/data_Turb_10240_beta25_dedt025_plm}"
: "${CAMPAIGN_CONFIG:?CAMPAIGN_CONFIG must be one frozen hash-checked Phase 5 config}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be an explicit unique Phase 5 scale/subset plan directory}"
: "${RUN_DIR:?RUN_DIR must be an explicit unique allocation directory}"
: "${ACTION:?ACTION must be plan, extract, verify, or inspect}"
: "${SCALE:?SCALE must be one configured Phase 5 scale: 640, 320, 160, or 80}"
: "${SUBSET:?SUBSET must be all, smoke, or matched_smoke}"
CLI="${SFUNCTOR_DIR}/scripts/phase5/run_phase5_extraction.py"
PYTHON_BIN="${SFUNCTOR_DIR}/venv_sfunctor/bin/python"
PHASE5_CLEAN_PARTIAL="${PHASE5_CLEAN_PARTIAL:-0}"
PHASE5_CLEAN_INCOMPLETE="${PHASE5_CLEAN_INCOMPLETE:-0}"
PHASE5_CLEAN_STALE_LOCK="${PHASE5_CLEAN_STALE_LOCK:-0}"

if [[ -e "${RUN_DIR}" ]]; then
  echo "ERROR: Phase 5 extraction allocation directory already exists: ${RUN_DIR}" >&2
  exit 3
fi

cd "${SFUNCTOR_DIR}"
mkdir -p logs "$(dirname -- "${OUTPUT_ROOT}")"
LOCK_DIR="$(dirname -- "${OUTPUT_ROOT}")/.$(basename -- "${OUTPUT_ROOT}").phase5_extract_action_lock"
LOCK_OWNER="${LOCK_DIR}/owner.txt"
LOCK_RECOVERY_CLAIM="${LOCK_DIR}/.recovery_claim"
LOCK_TOKEN="${SLURM_JOB_ID:-unknown}.$$.$RANDOM.$RANDOM"
RUN_DIR_CREATED=0

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

  if [[ "${RUN_DIR_CREATED}" != 1 ]]; then
    return
  fi
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
  echo "Recovered stale Phase 5 extraction action lock for inactive Slurm job ${owner_job_id}: ${LOCK_DIR}" >&2
}

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  if ! recover_stale_lock || ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    echo "ERROR: another Phase 5 extraction action holds ${LOCK_DIR}" >&2
    exit 3
  fi
fi
trap finalize EXIT
if ! mkdir "${RUN_DIR}"; then
  echo "ERROR: Phase 5 extraction allocation directory appeared concurrently: ${RUN_DIR}" >&2
  exit 3
fi
RUN_DIR_CREATED=1
mkdir "${RUN_DIR}/logs"
printf 'job_id=%s\naction=%s\nscale=%s\nsubset=%s\nrun_dir=%s\ncreated_utc=%s\ntoken=%s\n' \
  "${SLURM_JOB_ID:-unknown}" "${ACTION}" "${SCALE}" "${SUBSET}" "${RUN_DIR}" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${LOCK_TOKEN}" > "${LOCK_OWNER}"

module reset
module load gcc/9.3.0 python/.3.11-anaconda3
source "${SFUNCTOR_DIR}/venv_sfunctor/bin/activate"
export PYTHONPATH="${SFUNCTOR_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

COMMON=(
  --trusted-run "${TRUSTED_RUN}"
  --data-root "${DATA_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --campaign-config "${CAMPAIGN_CONFIG}"
  --scale "${SCALE}"
  --subset "${SUBSET}"
)
EXTRACT_FLAGS=()
[[ "${PHASE5_CLEAN_PARTIAL}" == "1" ]] && EXTRACT_FLAGS+=(--clean-partial)
[[ "${PHASE5_CLEAN_INCOMPLETE}" == "1" ]] && EXTRACT_FLAGS+=(--clean-incomplete)
[[ "${PHASE5_CLEAN_STALE_LOCK}" == "1" ]] && EXTRACT_FLAGS+=(--clean-stale-lock)

run_python() {
  /usr/bin/time -v srun -N 1 -n 1 --cpu-bind=cores "$PYTHON_BIN" "${CLI}" "$@"
}

case "${ACTION}" in
  plan)
    run_python plan "${COMMON[@]}" > "${RUN_DIR}/logs/plan.log" 2>&1
    ;;
  extract)
    run_python extract "${COMMON[@]}" "${EXTRACT_FLAGS[@]}" > "${RUN_DIR}/logs/extract.log" 2>&1
    ;;
  verify|inspect)
    run_python "${ACTION}" "${COMMON[@]}" > "${RUN_DIR}/logs/${ACTION}.log" 2>&1
    ;;
  *)
    echo "unsupported ACTION=${ACTION}; use plan, extract, verify, or inspect" >&2
    exit 2
    ;;
esac
