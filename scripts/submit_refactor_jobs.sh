#!/bin/bash
# Submit all rerun jobs after histogram refactor.
# Usage: bash scripts/submit_refactor_jobs.sh

set -euo pipefail

QOS="${QOS:-}"  # optionally set QOS=... when invoking the script

# Helper to submit a single job
submit_job() {
    local nodes="$1"
    local sim="$2"
    local sw="$3"
    local ndisp="$4"
    local nrand="$5"
    local nell="${6:-96}"
    local job_name="sf_${sim}_sw${sw}"
    local log_prefix="logs/${sim}_sw${sw}"
    mkdir -p logs
    local sbatch_opts=(-N "$nodes" -t 36:00:00)
    if [[ -n "$QOS" ]]; then
        sbatch_opts+=(--qos="$QOS")
    fi
    sbatch "${sbatch_opts[@]}" \
        -J "$job_name" \
        -o "${log_prefix}_%j.out" \
        -e "${log_prefix}_%j.err" \
        --export=ALL,SIM_NAME="$sim",STENCIL_WIDTH="$sw",N_DISP_TOTAL="$ndisp",N_RANDOM_SUBSAMPLES="$nrand",N_ELL_BINS="$nell" \
        job_scripts/production/run_distributed_analysis_andes_generic.sh
}

# submit_job 64 "Turb_10240_beta25_dedt025_plm" "2" 128000 32000
submit_job 64 "Turb_5120_beta25_dedt025_plm" "3" 64000 16000
# sf_Turb_5120_beta25_dedt025_plm_sw3

# # 10240 (cap at 64 nodes as requested)
# for sw in 2 3 5; do
#     submit_job 64 "Turb_10240_beta25_dedt025_plm" "$sw" 128000 32000
# done

# echo "Submitting 5120 beta survey (sw=2/3/5)..."
# for beta in 1 6 25 100; do
#     for sw in 2 3 5; do
#         submit_job 64 "Turb_5120_beta${beta}_dedt025_plm" "$sw" 64000 16000
#     done
# done

# echo "Submitting beta=25 resolution study (sw=2/3/5)..."
# # 2560
# for sw in 2 3 5; do
#     submit_job 32 "Turb_2560_beta25_dedt025_plm" "$sw" 32000 8000
# done
# # 1280
# for sw in 2 3 5; do
#     submit_job 16 "Turb_1280_beta25_dedt025_plm" "$sw" 16000 4000
# done
# # 640
# for sw in 2 3 5; do
#     submit_job 8 "Turb_640_beta25_dedt025_plm" "$sw" 8000 2000
# done
# # # 5120 --- this is already done as part of the beta survey
# # for sw in 2 3 5; do
# #     submit_job 64 "Turb_5120_beta25_dedt025_plm" "$sw" 64000 16000
# # done


# submit_job 64 "Turb_10240_beta25_dedt025_plm" "3" 256000 64000

# echo "All jobs submitted."
