#!/bin/bash
#============================================================================
# COLLECT ALL RESULTS FROM JOBS IN Jobs_to_do.md
#============================================================================
# This script organizes all completed analysis results into a single directory
# with subdirectories named by resolution_beta_stencilwidth
#============================================================================

echo "============================================================================"
echo " COLLECTING ALL STRUCTURE FUNCTION RESULTS"
echo " Date: $(date)"
echo "============================================================================"

# Create main results directory
RESULTS_DIR="/autofs/nccs-svm1_home2/dfielding/SFunctor/ALL_RESULTS_$(date +%Y%m%d)"
mkdir -p $RESULTS_DIR

echo " Results will be collected in: $RESULTS_DIR"
echo ""

# Base directory for all job results
BASE_DIR="/lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results"

# Function to copy results if they exist
copy_results() {
    local JOB_ID=$1
    local RES=$2
    local BETA=$3
    local SW=$4
    local DESC=$5
    
    # Construct source directory path
    SOURCE_DIR="${BASE_DIR}/results_Turb_${RES}_beta${BETA}_dedt025_plm"
    
    # Find the specific job directory
    if [ "$RES" == "10240" ]; then
        NDISP="100000"
        NRAND="100000"
    elif [ "$RES" == "5120" ]; then
        NDISP="50000"
        NRAND="50000"
    elif [ "$RES" == "2560" ]; then
        NDISP="25000"
        NRAND="25000"
    elif [ "$RES" == "640" ]; then
        NDISP="12500"
        NRAND="12500"
    fi
    
    JOB_DIR="${SOURCE_DIR}/ndisp${NDISP}_nrand${NRAND}_nell128_sw${SW}_job${JOB_ID}"
    
    # Target directory
    TARGET_DIR="${RESULTS_DIR}/${RES}_b${BETA}_sw${SW}"
    
    echo "Processing: $DESC (Job $JOB_ID)"
    
    if [ -d "$JOB_DIR" ]; then
        echo "  Source: $JOB_DIR"
        
        # Check if final merged result exists
        if [ -f "${JOB_DIR}/sf_results_all_slices.npz" ]; then
            mkdir -p "$TARGET_DIR"
            
            # Copy merged results
            cp "${JOB_DIR}/sf_results_all_slices.npz" "$TARGET_DIR/"
            echo "  ✓ Copied merged results"
            
            # Copy all plots
            PLOT_COUNT=$(ls -1 ${JOB_DIR}/*.png 2>/dev/null | wc -l)
            if [ $PLOT_COUNT -gt 0 ]; then
                cp ${JOB_DIR}/*.png "$TARGET_DIR/"
                echo "  ✓ Copied $PLOT_COUNT plots"
            else
                echo "  ⚠ No plots found"
            fi
            
            # Copy config if exists
            if [ -f "${JOB_DIR}/config.txt" ]; then
                cp "${JOB_DIR}/config.txt" "$TARGET_DIR/"
            fi
            
            # Create summary
            echo "Job $JOB_ID - $DESC" > "$TARGET_DIR/job_info.txt"
            echo "Source: $JOB_DIR" >> "$TARGET_DIR/job_info.txt"
            echo "Copied: $(date)" >> "$TARGET_DIR/job_info.txt"
            
        else
            echo "  ⚠ No merged results found (sf_results_all_slices.npz missing)"
            echo "  ⚠ Job may not have completed merging"
        fi
    else
        echo "  ✗ Directory not found: $JOB_DIR"
    fi
    echo ""
}

#============================================================================
# BETA SURVEY AT 5120
#============================================================================

echo "BETA SURVEY AT 5120 RESOLUTION"
echo "============================================================================"

# Based on job submission order from submit_all_jobs_simple.sh
copy_results 3665039 5120 1 2 "5120 β=1 SW=2"
copy_results 3665040 5120 6 2 "5120 β=6 SW=2"
copy_results 3665041 5120 25 2 "5120 β=25 SW=2"
copy_results 3665042 5120 100 2 "5120 β=100 SW=2"

copy_results 3665043 5120 1 3 "5120 β=1 SW=3"
copy_results 3665044 5120 6 3 "5120 β=6 SW=3"
copy_results 3665045 5120 25 3 "5120 β=25 SW=3"
copy_results 3665046 5120 100 3 "5120 β=100 SW=3"

copy_results 3665047 5120 1 5 "5120 β=1 SW=5"
copy_results 3665048 5120 6 5 "5120 β=6 SW=5"
copy_results 3665049 5120 25 5 "5120 β=25 SW=5"
copy_results 3665050 5120 100 5 "5120 β=100 SW=5"

#============================================================================
# RESOLUTION STUDY AT BETA=25
#============================================================================

echo "RESOLUTION STUDY AT BETA=25"
echo "============================================================================"

# 640 resolution
copy_results 3665051 640 25 2 "640 β=25 SW=2"
copy_results 3665054 640 25 3 "640 β=25 SW=3"
copy_results 3665056 640 25 5 "640 β=25 SW=5"

# 2560 resolution
copy_results 3665052 2560 25 2 "2560 β=25 SW=2"
copy_results 3665055 2560 25 3 "2560 β=25 SW=3"
copy_results 3665057 2560 25 5 "2560 β=25 SW=5"

# 10240 resolution
copy_results 3665053 10240 25 2 "10240 β=25 SW=2"
# Note: 10240 β=25 SW=3 was marked as done in Jobs_to_do.md
copy_results 3665058 10240 25 5 "10240 β=25 SW=5"

# Also check for the previously completed 10240 β=25 SW=3 (job 3663362)
echo "Previously completed job:"
copy_results 3663362 10240 25 3 "10240 β=25 SW=3 (previous)"

#============================================================================
# CHECK FOR FINISH_10240_MERGE.SH RESULTS
#============================================================================

echo "Checking for updated 10240 results from finish_10240_merge.sh..."
echo "============================================================================"

# The finish script should have completed these
if [ -f "${BASE_DIR}/results_Turb_10240_beta25_dedt025_plm/ndisp100000_nrand100000_nell128_sw2_job3665053/sf_results_all_slices.npz" ]; then
    echo "✓ 10240 SW=2 merge completed"
else
    echo "⚠ 10240 SW=2 still needs merging - check finish_10240_merge.log"
fi

if [ -f "${BASE_DIR}/results_Turb_10240_beta25_dedt025_plm/ndisp100000_nrand100000_nell128_sw5_job3665058/sf_results_all_slices.npz" ]; then
    echo "✓ 10240 SW=5 merge completed"  
else
    echo "⚠ 10240 SW=5 still needs merging - check finish_10240_merge.log"
fi

#============================================================================
# SUMMARY
#============================================================================

echo ""
echo "============================================================================"
echo " COLLECTION SUMMARY"
echo "============================================================================"

# Count collected results
COLLECTED=$(ls -d ${RESULTS_DIR}/*_b*_sw* 2>/dev/null | wc -l)
echo " Collected results: $COLLECTED directories"

echo ""
echo " Expected from Jobs_to_do.md:"
echo "   - Beta survey at 5120: 12 jobs (β=1,6,25,100 × SW=2,3,5)"
echo "   - Resolution study at β=25: 8 jobs (640,2560,10240 × SW=2,3,5)"
echo "   - Previously done: 10240 β=25 SW=3"
echo "   Total expected: 21 result sets"

echo ""
echo " Results organized in: $RESULTS_DIR"

# List what was collected
echo ""
echo " Collected directories:"
ls -la ${RESULTS_DIR}/ | grep "^d" | grep -v "^total"

echo ""
echo " Missing or incomplete:"
# Check for missing directories
for res in 640 2560 5120 10240; do
    for beta in 1 6 25 100; do
        for sw in 2 3 5; do
            # Skip invalid combinations
            if [ "$res" != "5120" ] && [ "$beta" != "25" ]; then
                continue
            fi
            
            TARGET="${RESULTS_DIR}/${res}_b${beta}_sw${sw}"
            if [ ! -d "$TARGET" ]; then
                echo "   ✗ ${res}_b${beta}_sw${sw}"
            elif [ ! -f "$TARGET/sf_results_all_slices.npz" ]; then
                echo "   ⚠ ${res}_b${beta}_sw${sw} (missing merged file)"
            fi
        done
    done
done

echo ""
echo "============================================================================"
echo " Collection complete: $(date)"
echo "============================================================================"