#!/usr/bin/env bash
#SBATCH -A AST207
#SBATCH -J phase3a_sampler
#SBATCH -o logs/phase3a_%x_%j.out
#SBATCH -e logs/phase3a_%x_%j.err
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

set -euo pipefail

SFUNCTOR_DIR="${SFUNCTOR_DIR:-/ccs/home/dfielding/SFunctor}"
PHASE2_ROOT="${PHASE2_ROOT:-/lustre/orion/ast207/proj-shared/dfielding/Production_plm/phase2_extract/benchmark_verified_release_primary_20260530}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be an explicit unique Phase 3a directory}"
: "${RUN_DIR:?RUN_DIR must be an explicit unique allocation directory}"
: "${ACTION:?ACTION must be plan, work, reduce, verify, controls, multinode_control, convergence, or summarize}"
CLI="${SFUNCTOR_DIR}/scripts/phase3a/run_phase3a_sampler.py"
PYTHON_BIN="${SFUNCTOR_DIR}/venv_sfunctor/bin/python"
PHASE3A_WORKERS="${PHASE3A_WORKERS:-1}"

cd "${SFUNCTOR_DIR}"
mkdir -p logs "${OUTPUT_ROOT}/logs" "${RUN_DIR}/logs"
LOCK_DIR="${OUTPUT_ROOT}/.phase3a_action_lock"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "ERROR: another Phase 3a action holds ${LOCK_DIR}" >&2
  exit 3
fi
printf 'job_id=%s\naction=%s\nrun_dir=%s\ncreated_utc=%s\n' \
  "${SLURM_JOB_ID:-unknown}" "${ACTION}" "${RUN_DIR}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "${LOCK_DIR}/owner.txt"
trap 'rm -f "${LOCK_DIR}/owner.txt"; rmdir "${LOCK_DIR}"' EXIT

module reset
module load gcc/9.3.0 python/.3.11-anaconda3
source "${SFUNCTOR_DIR}/venv_sfunctor/bin/activate"
export PYTHONPATH="${SFUNCTOR_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

run_python() {
  /usr/bin/time -v srun -N 1 -n 1 --cpu-bind=cores "$PYTHON_BIN" "$@"
}

case "${ACTION}" in
  work)
    /usr/bin/time -v srun --ntasks="${SLURM_JOB_NUM_NODES:-1}" --ntasks-per-node=1 \
      --cpus-per-task="${SLURM_CPUS_PER_TASK:-32}" --cpu-bind=cores \
      "$PYTHON_BIN" "${CLI}" work --phase2-root "${PHASE2_ROOT}" --output-root "${OUTPUT_ROOT}" \
      --workers "${PHASE3A_WORKERS}" \
      > "${RUN_DIR}/logs/work.log" 2>&1
    ;;
  multinode_control)
    /usr/bin/time -v srun --ntasks="${SLURM_JOB_NUM_NODES:-1}" --ntasks-per-node=1 \
      --cpus-per-task="${SLURM_CPUS_PER_TASK:-32}" --cpu-bind=cores \
      "$PYTHON_BIN" "${CLI}" multinode-work --phase2-root "${PHASE2_ROOT}" --output-root "${OUTPUT_ROOT}" \
      --workers "${PHASE3A_WORKERS}" \
      > "${RUN_DIR}/logs/multinode_work.log" 2>&1
    run_python "${CLI}" multinode-reduce --phase2-root "${PHASE2_ROOT}" --output-root "${OUTPUT_ROOT}" \
      --workers "${PHASE3A_WORKERS}" --task-count "${SLURM_JOB_NUM_NODES:-1}" \
      > "${RUN_DIR}/logs/multinode_reduce.log" 2>&1
    ;;
  plan|reduce|verify|controls|convergence|summarize)
    run_python "${CLI}" "${ACTION}" --phase2-root "${PHASE2_ROOT}" --output-root "${OUTPUT_ROOT}" \
      --workers "${PHASE3A_WORKERS}" \
      > "${RUN_DIR}/logs/${ACTION}.log" 2>&1
    ;;
  *)
    echo "unsupported ACTION=${ACTION}; use plan, work, reduce, verify, controls, multinode_control, convergence, or summarize" >&2
    exit 2
    ;;
esac
