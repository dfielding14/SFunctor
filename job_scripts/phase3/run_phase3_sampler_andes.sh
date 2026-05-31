#!/usr/bin/env bash
#SBATCH -A AST207
#SBATCH -J phase3_sampler
#SBATCH -o logs/phase3_%x_%j.out
#SBATCH -e logs/phase3_%x_%j.err
#SBATCH -t 00:15:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

set -euo pipefail

SFUNCTOR_DIR="${SFUNCTOR_DIR:-/ccs/home/dfielding/SFunctor}"
PHASE2_ROOT="${PHASE2_ROOT:-/lustre/orion/ast207/proj-shared/dfielding/Production_plm/phase2_extract/benchmark_verified_release_primary_20260530}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/lustre/orion/ast207/proj-shared/dfielding/Production_plm/phase3_sampler/smoke_20260530}"
: "${RUN_DIR:?RUN_DIR must be an explicit unique allocation directory}"
: "${ACTION:?ACTION must be synthetic, profile, profile_controls, profile_final, robustness_final, or smoke_final}"
CLI="${SFUNCTOR_DIR}/scripts/phase3/run_phase3_sampler.py"
CUBE_IDS=(L640_sub00370 L640_sub03942 L640_sub00579 L640_sub00738)

cd "${SFUNCTOR_DIR}"
mkdir -p logs "${OUTPUT_ROOT}/logs" "${RUN_DIR}/logs"
LOCK_DIR="${OUTPUT_ROOT}/.phase3_action_lock"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "ERROR: another Phase 3 action holds ${LOCK_DIR}" >&2
  exit 3
fi
trap 'rmdir "${LOCK_DIR}"' EXIT

module reset
module load gcc/9.3.0 python/.3.11-anaconda3
source "${SFUNCTOR_DIR}/venv_sfunctor/bin/activate"
export PYTHONPATH="${SFUNCTOR_DIR}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

run_python() {
  /usr/bin/time -v srun --cpu-bind=cores python "$@"
}

case "${ACTION}" in
  synthetic)
    run_python "${CLI}" synthetic --output-root "${OUTPUT_ROOT}" \
      > "${RUN_DIR}/logs/synthetic_validation.log" 2>&1
    ;;
  profile)
    run_python "${CLI}" smoke \
      --phase2-root "${PHASE2_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --cube-id L640_sub00370 \
      --ell-max 64 \
      --directions-per-radius 16 \
      --sample-count 4096 \
      --robustness-sample-count 1024 \
      --pair-batch-size 1024 \
      > "${RUN_DIR}/logs/profile_L640_sub00370.log" 2>&1
    run_python "${CLI}" verify \
      --phase2-root "${PHASE2_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --cube-id L640_sub00370 \
      > "${RUN_DIR}/logs/profile_restart_L640_sub00370.log" 2>&1
    ;;
  profile_controls)
    run_python "${CLI}" profile-controls \
      --phase2-root "${PHASE2_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      > "${RUN_DIR}/logs/profile_controls.log" 2>&1
    ;;
  profile_final)
    run_python "${CLI}" smoke \
      --phase2-root "${PHASE2_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --cube-id L640_sub00370 \
      > "${RUN_DIR}/logs/profile_final_L640_sub00370.log" 2>&1
    run_python "${CLI}" verify \
      --phase2-root "${PHASE2_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --cube-id L640_sub00370 \
      > "${RUN_DIR}/logs/profile_final_restart_L640_sub00370.log" 2>&1
    ;;
  robustness_final)
    run_python "${CLI}" robustness \
      --phase2-root "${PHASE2_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --cube-id L640_sub00370 \
      > "${RUN_DIR}/logs/robustness_L640_sub00370.log" 2>&1
    ;;
  smoke_final)
    run_python "${CLI}" synthetic --output-root "${OUTPUT_ROOT}" \
      > "${RUN_DIR}/logs/synthetic_validation.log" 2>&1
    for cube_id in "${CUBE_IDS[@]}"; do
      run_python "${CLI}" smoke \
        --phase2-root "${PHASE2_ROOT}" \
        --output-root "${OUTPUT_ROOT}" \
        --cube-id "${cube_id}" \
        > "${RUN_DIR}/logs/smoke_${cube_id}.log" 2>&1
      run_python "${CLI}" verify \
        --phase2-root "${PHASE2_ROOT}" \
        --output-root "${OUTPUT_ROOT}" \
        --cube-id "${cube_id}" \
        > "${RUN_DIR}/logs/restart_${cube_id}.log" 2>&1
    done
    run_python "${CLI}" robustness \
      --phase2-root "${PHASE2_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --cube-id L640_sub00370 \
      > "${RUN_DIR}/logs/robustness_L640_sub00370.log" 2>&1
    run_python "${CLI}" summarize \
      --phase2-root "${PHASE2_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      > "${RUN_DIR}/logs/summary.log" 2>&1
    ;;
  *)
    echo "unsupported ACTION=${ACTION}; use synthetic, profile, profile_controls, profile_final, robustness_final, or smoke_final" >&2
    exit 2
    ;;
esac
