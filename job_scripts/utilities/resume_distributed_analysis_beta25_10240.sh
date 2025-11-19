#!/bin/bash
# Resume wrapper script for beta25 10240 distributed structure function analysis
# This script resumes the incomplete run from job 3659610 or any specified job
#
# Usage: ./resume_distributed_analysis_beta25_10240.sh [previous_run_directory]
#
# Example:
#   ./resume_distributed_analysis_beta25_10240.sh /lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results/results_Turb_10240_beta25_dedt025_plm/ndisp100_000_nrand10_000_nell128_sw2_job3659610
#
# If no argument is provided, it will use the most recent job 3659610 directory

if [ $# -eq 1 ]; then
    # Use the provided directory
    export RESUME_FROM_DIR="$1"
    echo "Resuming from specified directory: $RESUME_FROM_DIR"
elif [ $# -eq 0 ]; then
    # Default to job 3659610 directory which timed out
    export RESUME_FROM_DIR="/lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results/results_Turb_10240_beta25_dedt025_plm/ndisp100_000_nrand10_000_nell128_sw2_job3659610"

    if [ -d "$RESUME_FROM_DIR" ]; then
        echo "Resuming from job 3659610 directory: $RESUME_FROM_DIR"
    else
        # Try to find the most recent run directory
        BASE_RESULTS_DIR="/lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results/results_Turb_10240_beta25_dedt025_plm"

        if [ -d "$BASE_RESULTS_DIR" ]; then
            # Find the most recent run directory (by modification time)
            LATEST_DIR=$(find "$BASE_RESULTS_DIR" -maxdepth 1 -type d -name "ndisp*" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2)

            if [ -n "$LATEST_DIR" ]; then
                export RESUME_FROM_DIR="$LATEST_DIR"
                echo "Auto-detected most recent run directory: $RESUME_FROM_DIR"
            else
                echo "No previous run directories found in $BASE_RESULTS_DIR"
                echo "Starting fresh run..."
            fi
        else
            echo "Base results directory not found: $BASE_RESULTS_DIR"
            echo "Starting fresh run..."
        fi
    fi
else
    echo "Usage: $0 [previous_run_directory]"
    echo "  If no directory is specified, will resume from job 3659610 or auto-detect the most recent run"
    exit 1
fi

# Check if resume directory exists and has results
if [ -n "$RESUME_FROM_DIR" ] && [ -d "$RESUME_FROM_DIR" ]; then
    echo ""
    echo "Checking resume directory contents..."

    # Count completed slice results
    NUM_RESULTS=$(ls -1 ${RESUME_FROM_DIR}/sf_results_Turb*.npz 2>/dev/null | wc -l)

    if [ $NUM_RESULTS -gt 0 ]; then
        echo "Found $NUM_RESULTS completed slice results to resume from"
        echo ""
        echo "Completed slices:"
        ls -1 ${RESUME_FROM_DIR}/sf_results_Turb*.npz 2>/dev/null | while read file; do
            basename "$file" .npz | sed 's/sf_results_//'
        done | head -10

        if [ $NUM_RESULTS -gt 10 ]; then
            echo "... and $((NUM_RESULTS - 10)) more"
        fi
    else
        echo "No completed results found in resume directory"
        echo ""
        echo "Note: Job 3659610 completed these 4 slices before timeout:"
        echo "  - Turb_10240_beta25_dedt025_plm_axis1_slice0p125_file0011"
        echo "  - Turb_10240_beta25_dedt025_plm_axis1_slice0p375_file0011"
        echo "  - Turb_10240_beta25_dedt025_plm_axis1_slicem0p125_file0011"
        echo "  - Turb_10240_beta25_dedt025_plm_axis1_slicem0p375_file0011"
        echo ""
        echo "The results should be in the directory but may not be accessible from this node."
    fi

    # Check for existing displacements file
    if [ -f "${RESUME_FROM_DIR}/displacements.npz" ]; then
        echo ""
        echo "Found existing displacements.npz - will reuse"
    fi

    echo ""
    echo "Analysis parameters from previous run:"
    echo "  - 100,000 displacements (10x less than target)"
    echo "  - 10,000 random subsamples (10x less than target)"
    echo "  - 128 ell bins"
    echo "  - Stencil width 2"
    echo "  - 64 nodes"
    echo ""
    echo "Starting resumed run..."
    echo "=================================="
else
    echo ""
    echo "Starting fresh run (no resume directory)..."
    echo "=================================="
fi

# Submit the job with resume capability
echo "Submitting job to continue processing remaining slices..."
sbatch /autofs/nccs-svm1_home2/dfielding/SFunctor/run_distributed_analysis_frontier_beta25_10240_sw2.sh