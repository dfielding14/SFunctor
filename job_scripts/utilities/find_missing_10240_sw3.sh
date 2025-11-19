#!/bin/bash
#============================================================================
# FIND AND COPY THE MISSING 10240 BETA25 SW3 RESULTS
#============================================================================

echo "Searching for 10240 beta25 sw3 results..."

BASE_DIR="/lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results"
RESULTS_DIR="/autofs/nccs-svm1_home2/dfielding/SFunctor/ALL_RESULTS_20250813"

# The job was marked as completed in Jobs_to_do.md
# Let's search for any sw3 job directories
echo "Checking for sw3 directories in 10240 beta25..."

# Try the job from our conversation history
JOB_DIR="${BASE_DIR}/results_Turb_10240_beta25_dedt025_plm/ndisp100_000_nrand100_000_nell128_sw3_job3663362"

if [ -d "$JOB_DIR" ]; then
    echo "Found: $JOB_DIR"
    TARGET_DIR="${RESULTS_DIR}/10240_b25_sw3"
    
    if [ -f "${JOB_DIR}/sf_results_all_slices.npz" ]; then
        mkdir -p "$TARGET_DIR"
        cp "${JOB_DIR}/sf_results_all_slices.npz" "$TARGET_DIR/"
        echo "✓ Copied merged results"
        
        # Copy plots
        PLOT_COUNT=$(ls -1 ${JOB_DIR}/*.png 2>/dev/null | wc -l)
        if [ $PLOT_COUNT -gt 0 ]; then
            cp ${JOB_DIR}/*.png "$TARGET_DIR/"
            echo "✓ Copied $PLOT_COUNT plots"
        fi
        
        # Create summary
        echo "Job 3663362 - 10240 β=25 SW=3" > "$TARGET_DIR/job_info.txt"
        echo "Source: $JOB_DIR" >> "$TARGET_DIR/job_info.txt"
        echo "Copied: $(date)" >> "$TARGET_DIR/job_info.txt"
    else
        echo "⚠ No merged results file found"
    fi
else
    echo "Directory not found with underscores, trying without..."
    
    # Try without underscores in the numbers
    JOB_DIR="${BASE_DIR}/results_Turb_10240_beta25_dedt025_plm/ndisp100000_nrand100000_nell128_sw3_job3663362"
    
    if [ -d "$JOB_DIR" ]; then
        echo "Found: $JOB_DIR"
        TARGET_DIR="${RESULTS_DIR}/10240_b25_sw3"
        
        if [ -f "${JOB_DIR}/sf_results_all_slices.npz" ]; then
            mkdir -p "$TARGET_DIR"
            cp "${JOB_DIR}/sf_results_all_slices.npz" "$TARGET_DIR/"
            echo "✓ Copied merged results"
            
            # Copy plots
            PLOT_COUNT=$(ls -1 ${JOB_DIR}/*.png 2>/dev/null | wc -l)
            if [ $PLOT_COUNT -gt 0 ]; then
                cp ${JOB_DIR}/*.png "$TARGET_DIR/"
                echo "✓ Copied $PLOT_COUNT plots"
            fi
            
            echo "Job 3663362 - 10240 β=25 SW=3" > "$TARGET_DIR/job_info.txt"
            echo "Source: $JOB_DIR" >> "$TARGET_DIR/job_info.txt"
            echo "Copied: $(date)" >> "$TARGET_DIR/job_info.txt"
        fi
    else
        echo "Still not found. Searching for any sw3 directories..."
        
        # Look for any sw3 directories
        for dir in ${BASE_DIR}/results_Turb_10240_beta25_dedt025_plm/*sw3*; do
            if [ -d "$dir" ]; then
                echo "Found potential match: $dir"
                
                if [ -f "${dir}/sf_results_all_slices.npz" ]; then
                    echo "  Has merged results file!"
                    
                    TARGET_DIR="${RESULTS_DIR}/10240_b25_sw3"
                    mkdir -p "$TARGET_DIR"
                    cp "${dir}/sf_results_all_slices.npz" "$TARGET_DIR/"
                    echo "  ✓ Copied merged results"
                    
                    # Copy plots
                    PLOT_COUNT=$(ls -1 ${dir}/*.png 2>/dev/null | wc -l)
                    if [ $PLOT_COUNT -gt 0 ]; then
                        cp ${dir}/*.png "$TARGET_DIR/"
                        echo "  ✓ Copied $PLOT_COUNT plots"
                    fi
                    
                    echo "10240 β=25 SW=3 (found)" > "$TARGET_DIR/job_info.txt"
                    echo "Source: $dir" >> "$TARGET_DIR/job_info.txt"
                    echo "Copied: $(date)" >> "$TARGET_DIR/job_info.txt"
                    
                    break
                fi
            fi
        done
    fi
fi

echo ""
echo "Final check of ALL_RESULTS directory:"
ls -la ${RESULTS_DIR}/ | grep 10240
echo ""
echo "Total directories in ALL_RESULTS:"
ls -d ${RESULTS_DIR}/*_b*_sw* 2>/dev/null | wc -l