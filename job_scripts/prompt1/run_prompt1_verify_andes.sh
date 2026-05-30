#!/bin/bash
#SBATCH -A AST207
#SBATCH -J prompt1_verify
#SBATCH -o logs/prompt1_%x_%j.out
#SBATCH -e logs/prompt1_%x_%j.err
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
set -euo pipefail

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

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
log "Prompt 1 independent catalog verification"
log "Run directory: ${RUN_DIR}"
log "Job: ${SLURM_JOB_ID:-manual}"
log "Node list: ${SLURM_JOB_NODELIST:-manual}"
log "CPUs: ${SLURM_CPUS_PER_TASK:-manual}"

/usr/bin/time -v srun --cpu-bind=cores \
  python "${SFUNCTOR_DIR}/scripts/prompt1/verify_prompt1_outputs.py" \
  --run-dir "${RUN_DIR}" \
  > "${RUN_DIR}/logs/verify_${SLURM_JOB_ID:-manual}.log" 2>&1
log "Independent catalog verification complete"
