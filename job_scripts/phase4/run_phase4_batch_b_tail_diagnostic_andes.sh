#!/usr/bin/env bash
#SBATCH -A AST207
#SBATCH -J phase4_batch_b_tail_diagnostic
#SBATCH -o logs/phase4_batch_b_tail_diagnostic_%x_%j.out
#SBATCH -e logs/phase4_batch_b_tail_diagnostic_%x_%j.err
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

set -euo pipefail

SFUNCTOR_DIR="${SFUNCTOR_DIR:-/ccs/home/dfielding/SFunctor}"
: "${PHASE2_ROOT:?PHASE2_ROOT must be the retained Phase 4 extraction directory}"
: "${REPRESENTATIVE_RELEASE_ROOT:?REPRESENTATIVE_RELEASE_ROOT must be the retained representative Batch B release}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be an explicit unique diagnostic directory}"
: "${RUN_DIR:?RUN_DIR must be an explicit unique allocation directory}"
: "${ACTION:?ACTION must be work, summarize, or verify}"
CLI="${SFUNCTOR_DIR}/scripts/phase4/run_phase4_batch_b_tail_diagnostic.py"
PYTHON_BIN="${SFUNCTOR_DIR}/venv_sfunctor/bin/python"

cd "${SFUNCTOR_DIR}"
mkdir -p logs "${RUN_DIR}/logs" "$(dirname -- "${OUTPUT_ROOT}")"

archive_slurm_resources() {
  local destination
  if [[ ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || ! command -v sacct >/dev/null 2>&1; then return; fi
  mkdir -p "${RUN_DIR}/resources"
  destination="${RUN_DIR}/resources/sacct_${SLURM_JOB_ID}.psv"
  sacct -j "${SLURM_JOB_ID}" \
    --format=JobID,ElapsedRaw,NNodes,AllocCPUS,MaxRSS,MaxRSSNode,MaxRSSTask,AveRSS,MaxVMSize,State,ExitCode \
    -n -P > "${destination}.tmp" || return
  mv "${destination}.tmp" "${destination}"
}
trap archive_slurm_resources EXIT

module reset
module load gcc/9.3.0 python/.3.11-anaconda3
source "${SFUNCTOR_DIR}/venv_sfunctor/bin/activate"
export PYTHONPATH="${SFUNCTOR_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

case "${ACTION}" in
  work)
    /usr/bin/time -v srun --ntasks="${SLURM_JOB_NUM_NODES:-1}" --ntasks-per-node=1 \
      --cpus-per-task="${SLURM_CPUS_PER_TASK:-32}" --cpu-bind=cores \
      "$PYTHON_BIN" "$CLI" work --phase2-root "$PHASE2_ROOT" \
      --representative-release-root "$REPRESENTATIVE_RELEASE_ROOT" --output-root "$OUTPUT_ROOT" \
      > "${RUN_DIR}/logs/work.log" 2>&1
    ;;
  summarize|verify)
    /usr/bin/time -v srun -N 1 -n 1 --cpu-bind=cores \
      "$PYTHON_BIN" "$CLI" "$ACTION" --phase2-root "$PHASE2_ROOT" \
      --representative-release-root "$REPRESENTATIVE_RELEASE_ROOT" --output-root "$OUTPUT_ROOT" \
      > "${RUN_DIR}/logs/${ACTION}.log" 2>&1
    ;;
  *)
    echo "unsupported ACTION=${ACTION}; use work, summarize, or verify" >&2
    exit 2
    ;;
esac
