#!/bin/bash
#============================================================================
# SUBMIT ALL STRUCTURE FUNCTION ANALYSIS JOBS FROM Jobs_to_do.md
#============================================================================
# This script submits all pending analyses with appropriate configurations
# Each job uses auto-chaining to handle time limits automatically
#============================================================================

echo "============================================================================"
echo " STRUCTURE FUNCTION ANALYSIS - BATCH JOB SUBMISSION"
echo " Date: $(date)"
echo "============================================================================"

# Track submitted jobs
SUBMITTED_JOBS=()
FAILED_SUBMISSIONS=()

# Base script to use
BASE_SCRIPT="/autofs/nccs-svm1_home2/dfielding/SFunctor/run_sf_analysis_frontier_autochain.sh"

# Function to create and submit a job
submit_job() {
    local RESOLUTION=$1
    local BETA=$2
    local STENCIL_WIDTH=$3
    local N_DISP=$4
    local N_RAND=$5
    local N_NODES=$6
    local TIME_HOURS=$7
    local DESCRIPTION=$8
    
    # Create unique job script
    JOB_SCRIPT="/autofs/nccs-svm1_home2/dfielding/SFunctor/job_scripts/production/job_${RESOLUTION}_beta${BETA}_sw${STENCIL_WIDTH}.sh"
    
    # Copy base script
    cp $BASE_SCRIPT $JOB_SCRIPT
    
    # Create sed script for modifications
    cat > /tmp/job_config_${RESOLUTION}_${BETA}_${STENCIL_WIDTH}.sed << EOF
# Simulation configuration
s/^RESOLUTION="[0-9]*"/RESOLUTION="$RESOLUTION"/
s/^BETA="[0-9]*"/BETA="$BETA"/
s/^STENCIL_WIDTH=[0-9]/STENCIL_WIDTH=$STENCIL_WIDTH/
s/^N_DISP_TOTAL=[0-9_]*/N_DISP_TOTAL=$N_DISP/
s/^N_RANDOM_SUBSAMPLES=[0-9_]*/N_RANDOM_SUBSAMPLES=$N_RAND/

# HPC Configuration
s/^N_NODES=[0-9]*/N_NODES=$N_NODES/
s/^TIME_LIMIT="[0-9:]*"/TIME_LIMIT="${TIME_HOURS}:00:00"/
s/^QUEUE="[a-z]*"/QUEUE="batch"/

# Auto-continuation - 1 minute buffer for all jobs as requested
s/^TIME_BUFFER_MINUTES=[0-9]*/TIME_BUFFER_MINUTES=1/
s/^MAX_CHAIN_JOBS=[0-9]*/MAX_CHAIN_JOBS=10/

# Clear custom bin edges
s/^LOG_SF_BIN_EDGES_MIN=.*/LOG_SF_BIN_EDGES_MIN=""/
s/^LOG_SF_BIN_EDGES_MAX=.*/LOG_SF_BIN_EDGES_MAX=""/

# Update SBATCH directives
s/#SBATCH --time=[0-9:].*/#SBATCH --time=${TIME_HOURS}:00:00/
s/#SBATCH -q [a-z]*/#SBATCH -q batch/
s/#SBATCH --nodes=[0-9]*/#SBATCH --nodes=$N_NODES/
s/#SBATCH -J [a-zA-Z0-9_]*/#SBATCH -J sf_${RESOLUTION}_b${BETA}_sw${STENCIL_WIDTH}/
EOF
    
    # Apply configuration
    sed -f /tmp/job_config_${RESOLUTION}_${BETA}_${STENCIL_WIDTH}.sed -i $JOB_SCRIPT
    
    # Make executable
    chmod +x $JOB_SCRIPT
    
    # Submit job
    echo ""
    echo " Submitting: $DESCRIPTION"
    echo "   Resolution: $RESOLUTION, Beta: $BETA, SW: $STENCIL_WIDTH"
    echo "   Nodes: $N_NODES, Time: ${TIME_HOURS}h"
    echo "   Displacements: $N_DISP, Samples: $N_RAND"
    
    JOB_OUTPUT=$(sbatch $JOB_SCRIPT 2>&1)
    
    if [ $? -eq 0 ]; then
        JOB_ID=$(echo $JOB_OUTPUT | awk '{print $4}')
        echo "   ✓ Submitted: Job ID $JOB_ID"
        SUBMITTED_JOBS+=("$JOB_ID:$DESCRIPTION")
    else
        echo "   ✗ Failed to submit"
        echo "   Error: $JOB_OUTPUT"
        FAILED_SUBMISSIONS+=("$DESCRIPTION")
    fi
    
    # Clean up temp file
    rm -f /tmp/job_config_${RESOLUTION}_${BETA}_${STENCIL_WIDTH}.sed
}

# Create job scripts directory
mkdir -p /autofs/nccs-svm1_home2/dfielding/SFunctor/job_scripts

#============================================================================
# BETA SURVEY AT 5120
#============================================================================

echo ""
echo "============================================================================"
echo " BETA SURVEY AT 5120 RESOLUTION"
echo "============================================================================"

# Stencil width = 2
submit_job 5120 1 2 50000 50000 64 2 "Beta survey: 5120 β=1 SW=2"
submit_job 5120 6 2 50000 50000 64 2 "Beta survey: 5120 β=6 SW=2"
submit_job 5120 25 2 50000 50000 64 2 "Beta survey: 5120 β=25 SW=2"
submit_job 5120 100 2 50000 50000 64 2 "Beta survey: 5120 β=100 SW=2"

# Stencil width = 3
submit_job 5120 1 3 50000 50000 64 2 "Beta survey: 5120 β=1 SW=3"
submit_job 5120 6 3 50000 50000 64 2 "Beta survey: 5120 β=6 SW=3"
submit_job 5120 25 3 50000 50000 64 2 "Beta survey: 5120 β=25 SW=3"
submit_job 5120 100 3 50000 50000 64 2 "Beta survey: 5120 β=100 SW=3"

# Stencil width = 5
submit_job 5120 1 5 50000 50000 64 2 "Beta survey: 5120 β=1 SW=5"
submit_job 5120 6 5 50000 50000 64 2 "Beta survey: 5120 β=6 SW=5"
submit_job 5120 25 5 50000 50000 64 2 "Beta survey: 5120 β=25 SW=5"
submit_job 5120 100 5 50000 50000 64 2 "Beta survey: 5120 β=100 SW=5"

#============================================================================
# RESOLUTION STUDY AT BETA=25
#============================================================================

echo ""
echo "============================================================================"
echo " RESOLUTION STUDY AT BETA=25"
echo "============================================================================"

# Stencil width = 2
submit_job 640 25 2 12500 12500 2 2 "Resolution study: 640 β=25 SW=2"
submit_job 2560 25 2 25000 25000 32 2 "Resolution study: 2560 β=25 SW=2"
submit_job 5120 25 2 50000 50000 64 2 "Resolution study: 5120 β=25 SW=2 (duplicate)"
submit_job 10240 25 2 100000 100000 128 2 "Resolution study: 10240 β=25 SW=2"

# Stencil width = 3
submit_job 640 25 3 12500 12500 2 2 "Resolution study: 640 β=25 SW=3"
submit_job 2560 25 3 25000 25000 32 2 "Resolution study: 2560 β=25 SW=3"
submit_job 5120 25 3 50000 50000 64 2 "Resolution study: 5120 β=25 SW=3 (duplicate)"
# Note: 10240 β=25 SW=3 is marked as done [x] in Jobs_to_do.md

# Stencil width = 5
submit_job 640 25 5 12500 12500 2 2 "Resolution study: 640 β=25 SW=5"
submit_job 2560 25 5 25000 25000 32 2 "Resolution study: 2560 β=25 SW=5"
submit_job 5120 25 5 50000 50000 64 2 "Resolution study: 5120 β=25 SW=5 (duplicate)"
submit_job 10240 25 5 100000 100000 128 2 "Resolution study: 10240 β=25 SW=5"

#============================================================================
# SUMMARY
#============================================================================

echo ""
echo "============================================================================"
echo " SUBMISSION SUMMARY"
echo "============================================================================"
echo " Total jobs submitted: ${#SUBMITTED_JOBS[@]}"
echo " Failed submissions: ${#FAILED_SUBMISSIONS[@]}"
echo ""

if [ ${#SUBMITTED_JOBS[@]} -gt 0 ]; then
    echo " Successfully Submitted Jobs:"
    for job in "${SUBMITTED_JOBS[@]}"; do
        echo "   $job"
    done
fi

if [ ${#FAILED_SUBMISSIONS[@]} -gt 0 ]; then
    echo ""
    echo " Failed Submissions:"
    for job in "${FAILED_SUBMISSIONS[@]}"; do
        echo "   $job"
    done
fi

echo ""
echo " Monitor jobs with: squeue -u $USER"
echo " Check job status: squeue -j <job_id>"
echo ""
echo " All jobs configured with:"
echo "   - Auto-chaining enabled (max 10 continuations)"
echo "   - 1 minute time buffer before continuation"
echo "   - 2 hour time limit per job segment"
echo ""
echo "============================================================================"

# Save submission log
LOG_FILE="/autofs/nccs-svm1_home2/dfielding/SFunctor/job_scripts/utilities/submission_log_$(date +%Y%m%d_%H%M%S).txt"
{
    echo "Submission Log - $(date)"
    echo "========================"
    echo ""
    echo "Submitted Jobs:"
    for job in "${SUBMITTED_JOBS[@]}"; do
        echo "  $job"
    done
    echo ""
    echo "Failed:"
    for job in "${FAILED_SUBMISSIONS[@]}"; do
        echo "  $job"
    done
} > $LOG_FILE

echo " Log saved to: $LOG_FILE"
echo "============================================================================"
