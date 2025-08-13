#!/bin/bash
# Script to combine and plot results from job 3663362

echo "Processing job 3663362 results..."

# Activate virtual environment
source /autofs/nccs-svm1_home2/dfielding/SFunctor/venv_frontier/bin/activate

# Set paths
JOB_DIR="/lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results/results_Turb_10240_beta25_dedt025_plm/ndisp100_000_nrand100_000_nell128_sw3_job3663362"
SFUNCTOR_DIR="/autofs/nccs-svm1_home2/dfielding/SFunctor"

# Check if job directory exists
if [ ! -d "$JOB_DIR" ]; then
    echo "ERROR: Job directory not found: $JOB_DIR"
    exit 1
fi

echo ""
echo "Checking for completed slice results in job 3663362..."
SLICE_RESULTS=$(ls -1 ${JOB_DIR}/sf_results_*.npz 2>/dev/null)

if [ -z "$SLICE_RESULTS" ]; then
    echo "No completed slice results found (sf_results_*.npz files)"
    echo ""
    echo "Checking for histogram directories that need combining..."
    
    # Find all histogram directories
    HIST_DIRS=$(find ${JOB_DIR} -type d -name "*_histograms" 2>/dev/null)
    
    if [ -z "$HIST_DIRS" ]; then
        echo "No histogram directories found either"
        exit 1
    fi
    
    echo "Found histogram directories to process:"
    echo "$HIST_DIRS" | while read dir; do
        basename "$dir"
    done
    
    # Process each histogram directory
    for HIST_DIR in $HIST_DIRS; do
        SLICE_NAME=$(basename "$HIST_DIR" "_histograms")
        echo ""
        echo "Processing: $SLICE_NAME"
        echo "------------------------------------"
        
        # Check if histogram files exist
        HIST_COUNT=$(ls -1 ${HIST_DIR}/histogram_*.npz 2>/dev/null | wc -l)
        
        if [ $HIST_COUNT -gt 0 ]; then
            echo "Found $HIST_COUNT histogram files to combine"
            
            # Combine node histograms for this slice
            python ${SFUNCTOR_DIR}/combine_histograms.py \
                --pattern "${HIST_DIR}/histogram_*.npz" \
                --output ${JOB_DIR}/sf_results_${SLICE_NAME}.npz \
                --mode node
            
            if [ $? -eq 0 ]; then
                echo "Successfully created: sf_results_${SLICE_NAME}.npz"
            else
                echo "ERROR: Failed to combine histograms for $SLICE_NAME"
            fi
        else
            echo "No histogram files found in $HIST_DIR"
        fi
    done
    
    echo ""
    echo "Re-checking for completed slice results..."
    SLICE_RESULTS=$(ls -1 ${JOB_DIR}/sf_results_*.npz 2>/dev/null)
fi

if [ -n "$SLICE_RESULTS" ]; then
    NUM_SLICES=$(echo "$SLICE_RESULTS" | wc -l)
    echo "Found $NUM_SLICES completed slice results:"
    echo "$SLICE_RESULTS" | while read file; do
        basename "$file"
    done | head -10
    
    if [ $NUM_SLICES -gt 10 ]; then
        echo "... and $((NUM_SLICES - 10)) more"
    fi
    
    echo ""
    echo "Combining all slice results..."
    python ${SFUNCTOR_DIR}/combine_histograms.py \
        --pattern "${JOB_DIR}/sf_results_*.npz" \
        --output ${JOB_DIR}/sf_results_all_slices.npz \
        --mode slice
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "Successfully created combined results: sf_results_all_slices.npz"
        
        echo ""
        echo "Generating plots..."
        python ${SFUNCTOR_DIR}/plot_structure_functions.py \
            ${JOB_DIR}/sf_results_all_slices.npz
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "Plots generated successfully!"
            echo ""
            echo "Results available in: $JOB_DIR"
            echo "Main outputs:"
            echo "  - sf_results_all_slices.npz (combined data)"
            echo "  - sf_results_all_slices_mean_structure_functions.png"
            echo "  - sf_results_all_slices_2d_histograms_normalized.png"
            echo "  - sf_results_all_slices_angular_distribution_2d.png"
            echo "  - sf_results_all_slices_cross_product_ratios.png"
            echo "  - Individual channel 2D histograms (11 files)"
        else
            echo "ERROR: Plot generation failed"
        fi
    else
        echo "ERROR: Failed to combine slice results"
    fi
else
    echo "No slice results found to combine"
fi