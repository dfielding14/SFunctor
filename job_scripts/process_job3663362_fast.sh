#!/bin/bash
# Fast parallel script to combine and plot results from job 3663362

echo "Processing job 3663362 results with fast parallel combiner..."

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
    HIST_DIRS=$(find ${JOB_DIR} -type d -name "*_histograms" 2>/dev/null | sort)
    
    if [ -z "$HIST_DIRS" ]; then
        echo "No histogram directories found either"
        exit 1
    fi
    
    NUM_DIRS=$(echo "$HIST_DIRS" | wc -l)
    echo "Found $NUM_DIRS histogram directories to process"
    echo ""
    
    # Process each histogram directory using the fast combiner
    COUNTER=0
    for HIST_DIR in $HIST_DIRS; do
        COUNTER=$((COUNTER + 1))
        SLICE_NAME=$(basename "$HIST_DIR" "_histograms")
        echo "[$COUNTER/$NUM_DIRS] Processing: $SLICE_NAME"
        echo "------------------------------------"
        
        # Check if histogram files exist
        HIST_COUNT=$(ls -1 ${HIST_DIR}/histogram_*.npz 2>/dev/null | wc -l)
        
        if [ $HIST_COUNT -gt 0 ]; then
            echo "Found $HIST_COUNT histogram files to combine"
            
            # Use the fast parallel combiner for node histograms
            python ${SFUNCTOR_DIR}/combine_histograms_fast.py \
                --pattern "${HIST_DIR}/histogram_*.npz" \
                --output ${JOB_DIR}/sf_results_${SLICE_NAME}.npz \
                --mode node \
                --workers 32
            
            if [ $? -eq 0 ]; then
                echo "Successfully created: sf_results_${SLICE_NAME}.npz"
                echo ""
            else
                echo "ERROR: Failed to combine histograms for $SLICE_NAME"
                echo ""
            fi
        else
            echo "No histogram files found in $HIST_DIR"
            echo ""
        fi
    done
    
    echo "======================================"
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
    echo "======================================"
    echo "Combining all slice results using fast parallel combiner..."
    python ${SFUNCTOR_DIR}/combine_histograms_fast.py \
        --pattern "${JOB_DIR}/sf_results_*.npz" \
        --output ${JOB_DIR}/sf_results_all_slices.npz \
        --mode slice \
        --workers 32
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "Successfully created combined results: sf_results_all_slices.npz"
        
        echo ""
        echo "======================================"
        echo "Generating plots..."
        python ${SFUNCTOR_DIR}/plot_structure_functions.py \
            ${JOB_DIR}/sf_results_all_slices.npz
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "======================================"
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
            echo ""
            echo "Processing complete!"
        else
            echo "ERROR: Plot generation failed"
        fi
    else
        echo "ERROR: Failed to combine slice results"
    fi
else
    echo "No slice results found to combine"
fi

echo ""
echo "Script finished at: $(date)"