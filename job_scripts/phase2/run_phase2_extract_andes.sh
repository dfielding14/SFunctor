#!/usr/bin/env bash
#SBATCH -A AST207
#SBATCH -J phase2_extract
#SBATCH -o logs/phase2_%x_%j.out
#SBATCH -e logs/phase2_%x_%j.err
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

set -euo pipefail

SFUNCTOR_DIR="${SFUNCTOR_DIR:-/ccs/home/dfielding/SFunctor}"
TRUSTED_RUN="${TRUSTED_RUN:-/lustre/orion/ast207/proj-shared/dfielding/Production_plm/prompt1_catalog/t6_final_primary_20260530}"
DATA_ROOT="${DATA_ROOT:-/lustre/orion/ast207/proj-shared/dfielding/Production_plm/data/data_Turb_10240_beta25_dedt025_plm}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/lustre/orion/ast207/proj-shared/dfielding/Production_plm/phase2_extract/benchmark_20260530}"
: "${RUN_DIR:?RUN_DIR must be an explicit unique allocation directory}"
: "${ACTION:?ACTION must be preflight, smoke, benchmark_remaining, or benchmark_final_all}"
CLI="${SFUNCTOR_DIR}/scripts/phase2/run_phase2_extraction.py"
SMOKE_ID=L640_sub00370
REMAINING_IDS=(L640_sub03942 L640_sub00579 L640_sub00738)
FINAL_IDS=(L640_sub00370 L640_sub03942 L640_sub00579 L640_sub00738)

cd "${SFUNCTOR_DIR}"
mkdir -p logs "${OUTPUT_ROOT}/logs" "${RUN_DIR}/logs"
LOCK_DIR="${OUTPUT_ROOT}/.phase2_action_lock"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "ERROR: another Phase 2 action holds ${LOCK_DIR}" >&2
  exit 3
fi
trap 'rmdir "${LOCK_DIR}"' EXIT

module reset
module load gcc/9.3.0 python/.3.11-anaconda3
source "${SFUNCTOR_DIR}/venv_sfunctor/bin/activate"
export PYTHONPATH="${SFUNCTOR_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

COMMON=(--trusted-run "${TRUSTED_RUN}" --data-root "${DATA_ROOT}" --output-root "${OUTPUT_ROOT}")

run_python() {
  srun --cpu-bind=cores python "${CLI}" "$@"
}

run_subset() {
  local cube_ids=("$@")
  local cube_id
  for cube_id in "${cube_ids[@]}"; do
    /usr/bin/time -v srun --cpu-bind=cores python "${CLI}" preflight "${COMMON[@]}" --cube-id "${cube_id}" \
      > "${RUN_DIR}/logs/preflight_${cube_id}.log" 2>&1
    /usr/bin/time -v srun --cpu-bind=cores python "${CLI}" extract "${COMMON[@]}" --cube-id "${cube_id}" \
      > "${RUN_DIR}/logs/extract_${cube_id}.log" 2>&1
    /usr/bin/time -v srun --cpu-bind=cores python "${CLI}" verify "${COMMON[@]}" --cube-id "${cube_id}" \
      > "${RUN_DIR}/logs/restart_${cube_id}.log" 2>&1
    /usr/bin/time -v srun --cpu-bind=cores python "${CLI}" extract "${COMMON[@]}" --cube-id "${cube_id}" \
      > "${RUN_DIR}/logs/reuse_${cube_id}.log" 2>&1
    run_python inspect --output-root "${OUTPUT_ROOT}" --cube-id "${cube_id}" \
      > "${RUN_DIR}/logs/inspect_${cube_id}.log" 2>&1
  done
}

case "${ACTION}" in
  preflight)
    run_python preflight "${COMMON[@]}" --all-pilot
    run_python probe "${COMMON[@]}"
    ;;
  smoke)
    run_python preflight "${COMMON[@]}" --all-pilot \
      > "${RUN_DIR}/logs/preflight_all_pilot.log" 2>&1
    run_python probe "${COMMON[@]}" \
      > "${RUN_DIR}/logs/preflight_probes.log" 2>&1
    run_python stream-validate "${COMMON[@]}" \
      > "${RUN_DIR}/logs/stream_validation.log" 2>&1
    run_subset "${SMOKE_ID}"
    ;;
  benchmark_remaining)
    run_subset "${REMAINING_IDS[@]}"
    run_python summarize "${COMMON[@]}" \
      > "${RUN_DIR}/logs/summary.log" 2>&1
    ;;
  benchmark_final_all)
    run_python preflight "${COMMON[@]}" --all-pilot \
      > "${RUN_DIR}/logs/preflight_all_pilot.log" 2>&1
    run_python probe "${COMMON[@]}" \
      > "${RUN_DIR}/logs/preflight_probes.log" 2>&1
    run_python stream-validate "${COMMON[@]}" \
      > "${RUN_DIR}/logs/stream_validation.log" 2>&1
    run_subset "${FINAL_IDS[@]}"
    run_python summarize "${COMMON[@]}" \
      > "${RUN_DIR}/logs/summary.log" 2>&1
    ;;
  *)
    echo "unsupported ACTION=${ACTION}; use preflight, smoke, benchmark_remaining, or benchmark_final_all" >&2
    exit 2
    ;;
esac
