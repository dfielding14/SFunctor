#!/bin/bash
# Resume wrapper script for distributed structure function analysis on Frontier
# This script demonstrates how to resume a previous incomplete run
#
# Usage: ./resume_distributed_analysis_frontier.sh [previous_run_directory]
# 
# Example:
#   ./resume_distributed_analysis_frontier.sh /lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results/results_Turb_5120_beta6_dedt025_plm/ndisp10_000_nrand1_000_nell128_sw2_job3650097
#
# If no argument is provided, it will try to find the most recent run directory

if [ $# -eq 1 ]; then
    # Use the provided directory
    export RESUME_FROM_DIR="$1"
    echo "Resuming from specified directory: $RESUME_FROM_DIR"
elif [ $# -eq 0 ]; then
    # Try to find the most recent run directory
    # Note: Updated for the beta6 simulation name that's currently being used
    BASE_RESULTS_DIR="/lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results/results_Turb_5120_beta6_dedt025_plm"
    
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
else
    echo "Usage: $0 [previous_run_directory]"
    echo "  If no directory is specified, will try to auto-detect the most recent run"
    exit 1
fi

# Check if resume directory exists and has results
if [ -n "$RESUME_FROM_DIR" ] && [ -d "$RESUME_FROM_DIR" ]; then
    echo ""
    echo "Checking resume directory contents..."
    NUM_RESULTS=$(ls -1 ${RESUME_FROM_DIR}/sf_results_Turb*.npz 2>/dev/null | wc -l)
    
    if [ $NUM_RESULTS -gt 0 ]; then
        echo "Found $NUM_RESULTS completed slice results to resume from"
        echo ""
        echo "Completed slices:"
        ls -1 ${RESUME_FROM_DIR}/sf_results_Turb*.npz | head -10
        if [ $NUM_RESULTS -gt 10 ]; then
            echo "... and $((NUM_RESULTS - 10)) more"
        fi
    else
        echo "No completed results found in resume directory"
    fi
    
    # Check for displacements file
    DISP_FILES=$(ls -1 ${RESUME_FROM_DIR}/displacements*.npz 2>/dev/null | wc -l)
    if [ $DISP_FILES -gt 0 ]; then
        echo ""
        echo "Found displacement file(s):"
        ls -1 ${RESUME_FROM_DIR}/displacements*.npz
    fi
    
    echo ""
    echo "Starting resumed run..."
    echo "=================================="
else
    echo ""
    echo "Starting fresh run (no resume directory)..."
    echo "=================================="
fi

# Submit the job with resume capability to Frontier
sbatch /autofs/nccs-svm1_home2/dfielding/SFunctor/run_distributed_analysis_frontier.sh