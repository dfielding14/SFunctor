#!/bin/bash
# Complete processing of remaining histogram directories for job 3663362

echo "Completing job 3663362 processing..."
echo "Processing remaining histogram directories..."

# Activate virtual environment
source /autofs/nccs-svm1_home2/dfielding/SFunctor/venv_frontier/bin/activate

# Set paths
JOB_DIR="/lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results/results_Turb_10240_beta25_dedt025_plm/ndisp100_000_nrand100_000_nell128_sw3_job3663362"
SFUNCTOR_DIR="/autofs/nccs-svm1_home2/dfielding/SFunctor"

cd $JOB_DIR

# Process each histogram directory that doesn't have a corresponding sf_results file
TOTAL=0
PROCESSED=0
SKIPPED=0

for dir in *_histograms; do
    if [ -d "$dir" ]; then
        TOTAL=$((TOTAL + 1))
        slice_name="${dir%_histograms}"
        
        if [ ! -f "sf_results_${slice_name}.npz" ]; then
            PROCESSED=$((PROCESSED + 1))
            echo ""
            echo "[$PROCESSED] Processing: $slice_name"
            echo "------------------------------------"
            
            python ${SFUNCTOR_DIR}/combine_histograms_fast.py \
                --pattern "${JOB_DIR}/${dir}/histogram_*.npz" \
                --output "${JOB_DIR}/sf_results_${slice_name}.npz" \
                --mode node \
                --workers 32
            
            if [ $? -eq 0 ]; then
                echo "Successfully created: sf_results_${slice_name}.npz"
            else
                echo "ERROR: Failed to process $slice_name"
            fi
        else
            SKIPPED=$((SKIPPED + 1))
            echo "[$SKIPPED skipped] Already exists: sf_results_${slice_name}.npz"
        fi
    fi
done

echo ""
echo "======================================"
echo "Summary:"
echo "  Total histogram directories: $TOTAL"
echo "  Newly processed: $PROCESSED"
echo "  Already existed: $SKIPPED"
echo ""

# Now combine all slice results
echo "======================================"
echo "Combining all slice results..."
python ${SFUNCTOR_DIR}/combine_histograms_fast.py \
    --pattern "${JOB_DIR}/sf_results_Turb*.npz" \
    --output "${JOB_DIR}/sf_results_all_slices_complete.npz" \
    --mode slice \
    --workers 32

if [ $? -eq 0 ]; then
    echo ""
    echo "Successfully created complete combined results: sf_results_all_slices_complete.npz"
    
    # Remove old incomplete version and rename
    if [ -f "${JOB_DIR}/sf_results_all_slices.npz" ]; then
        echo "Replacing old incomplete sf_results_all_slices.npz..."
        mv "${JOB_DIR}/sf_results_all_slices_complete.npz" "${JOB_DIR}/sf_results_all_slices.npz"
    else
        mv "${JOB_DIR}/sf_results_all_slices_complete.npz" "${JOB_DIR}/sf_results_all_slices.npz"
    fi
    
    echo ""
    echo "======================================"
    echo "Regenerating plots with complete data..."
    python ${SFUNCTOR_DIR}/plot_structure_functions.py \
        ${JOB_DIR}/sf_results_all_slices.npz
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "======================================"
        echo "Processing Complete!"
        echo "======================================"
        echo ""
        echo "All 12 slices have been processed and combined."
        echo "Results available in: $JOB_DIR"
        echo ""
        
        # List all slice results
        echo "Slice results created:"
        ls -1 ${JOB_DIR}/sf_results_Turb*.npz | grep -v all_slices | while read file; do
            basename "$file"
        done
        echo ""
        echo "Combined result: sf_results_all_slices.npz"
        echo "Plots regenerated with complete data."
    else
        echo "ERROR: Plot generation failed"
    fi
else
    echo "ERROR: Failed to combine slice results"
fi

echo ""
echo "Script finished at: $(date)"