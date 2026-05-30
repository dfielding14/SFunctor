#!/bin/bash
#SBATCH -A AST207
#SBATCH -J prompt1_cbin
#SBATCH -o logs/prompt1_%x_%j.out
#SBATCH -e logs/prompt1_%x_%j.err
#SBATCH -t 04:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
set -euo pipefail

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

: "${ACTION:?ACTION must be validate, build, or analyze}"
: "${RUN_DIR:?RUN_DIR must be an explicit unique workflow directory}"
SFUNCTOR_DIR="/ccs/home/dfielding/SFunctor"

module reset
module load gcc/9.3.0 python/.3.11-anaconda3
source "${SFUNCTOR_DIR}/venv_sfunctor/bin/activate"
export PYTHONPATH="${SFUNCTOR_DIR}:${PYTHONPATH:-}"

mkdir -p "${RUN_DIR}/logs"
LOCK_DIR="${RUN_DIR}/.prompt1_action_lock"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "ERROR: another Prompt 1 action holds ${LOCK_DIR}" >&2
  exit 3
fi
trap 'rmdir "${LOCK_DIR}"' EXIT
log "Prompt 1 cbin workflow"
log "Action: ${ACTION}"
log "Run directory: ${RUN_DIR}"
log "Job: ${SLURM_JOB_ID:-manual}"
log "Node list: ${SLURM_JOB_NODELIST:-manual}"
log "CPUs: ${SLURM_CPUS_PER_TASK:-manual}"

case "${ACTION}" in
  validate)
    command=(
      python "${SFUNCTOR_DIR}/scripts/prompt1/validate_reconstruction.py"
      --output-dir "${RUN_DIR}/validation"
    )
    ;;
  build)
    command=(
      python "${SFUNCTOR_DIR}/scripts/prompt1/run_prompt1_catalog.py"
      build
      --run-dir "${RUN_DIR}"
    )
    ;;
  analyze)
    command=(
      python "${SFUNCTOR_DIR}/scripts/prompt1/run_prompt1_catalog.py"
      analyze
      --run-dir "${RUN_DIR}"
    )
    ;;
  *)
    echo "ERROR: unknown ACTION=${ACTION}" >&2
    exit 2
    ;;
esac

log "Launching: ${command[*]}"
/usr/bin/time -v srun --cpu-bind=cores "${command[@]}" \
  > "${RUN_DIR}/logs/${ACTION}_${SLURM_JOB_ID:-manual}.log" 2>&1
log "Action ${ACTION} complete"
