#!/bin/bash
# Submit a chain of 2-hour jobs for the 10240 simulation
# Usage: ./submit_10240_chain.sh [NUM_JOBS]
# Default: 12 jobs (24 hours total)

set -euo pipefail

NUM_JOBS="${1:-12}"
TIME_LIMIT="2:00:00"
NODES=128

# Job parameters
export SIM_NAME="Turb_10240_beta25_dedt025_plm"
export STENCIL_WIDTH=3
export N_DISP_TOTAL=256000
export N_RANDOM_SUBSAMPLES=128000
export N_ELL_BINS=96

# Fixed run name so all jobs work in the same directory
RUN_NAME="ndisp${N_DISP_TOTAL}_nrand${N_RANDOM_SUBSAMPLES}_nell${N_ELL_BINS}_sw${STENCIL_WIDTH}"
export RUN_NAME_OVERRIDE="${RUN_NAME}"
export RESULTS_BASE="/ccs/home/dfielding/SFunctor/Results"
WORK_DIR="${RESULTS_BASE}/results_${SIM_NAME}/${RUN_NAME}"
export RESUME_DIR="${WORK_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERIC_SCRIPT="${SCRIPT_DIR}/run_distributed_analysis_andes_generic.sh"
RESTART_SCRIPT="${SCRIPT_DIR}/run_distributed_analysis_andes_restart.sh"

echo "=============================================="
echo "Submitting chain of ${NUM_JOBS} jobs"
echo "SIM_NAME: ${SIM_NAME}"
echo "Parameters: ndisp=${N_DISP_TOTAL}, nrand=${N_RANDOM_SUBSAMPLES}, nell=${N_ELL_BINS}, sw=${STENCIL_WIDTH}"
echo "Time per job: ${TIME_LIMIT}"
echo "Nodes: ${NODES}"
echo "Work directory: ${WORK_DIR}"
echo "=============================================="

# Submit first job (uses generic or restart - restart handles both cases)
FIRST_JOB=$(sbatch \
    --parsable \
    -t "${TIME_LIMIT}" \
    -N "${NODES}" \
    --export=ALL \
    "${RESTART_SCRIPT}")

echo "Submitted job 1/${NUM_JOBS}: ${FIRST_JOB}"

PREV_JOB="${FIRST_JOB}"

# Submit subsequent restart jobs with dependencies
for ((i=2; i<=NUM_JOBS; i++)); do
    NEXT_JOB=$(sbatch \
        --parsable \
        -t "${TIME_LIMIT}" \
        -N "${NODES}" \
        --dependency=afterany:${PREV_JOB} \
        --export=ALL \
        "${RESTART_SCRIPT}")

    echo "Submitted job ${i}/${NUM_JOBS}: ${NEXT_JOB} (depends on ${PREV_JOB})"
    PREV_JOB="${NEXT_JOB}"
done

echo ""
echo "All jobs submitted. Check queue with: squeue -u \$USER"
echo "Work directory: ${WORK_DIR}"
echo ""
echo "To cancel all jobs in the chain:"
echo "  scancel ${FIRST_JOB} ${PREV_JOB}  # (and intermediate job IDs)"
