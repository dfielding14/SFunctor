#!/bin/bash
#============================================================================
# FINISH MERGING AND PLOTTING FOR 10240 JOBS
#============================================================================
# Jobs 3665053 (SW2) and 3665058 (SW5) completed all processing
# but need final merge and plotting
#============================================================================

echo "============================================================================"
echo " FINISHING 10240 MERGE AND PLOTTING"
echo " Date: $(date)"
echo "============================================================================"

# Load modules
module reset
module load PrgEnv-gnu

# Activate virtual environment
source /autofs/nccs-svm1_home2/dfielding/SFunctor/venv_frontier/bin/activate

SFUNCTOR_DIR="/autofs/nccs-svm1_home2/dfielding/SFunctor"
BASE_DIR="/lustre/orion/ast207/proj-shared/dfielding/Production_plm/sfunctor_results"

#============================================================================
# JOB 1: 10240 SW2 (Job 3665053)
#============================================================================

echo ""
echo "Processing 10240 SW2 (Job 3665053)..."
echo "----------------------------------------------------------------------------"

WORK_DIR="${BASE_DIR}/results_Turb_10240_beta25_dedt025_plm/ndisp100000_nrand100000_nell128_sw2_job3665053"
cd $WORK_DIR

# Check for any remaining histogram directories to merge
echo "Checking for unmerged histogram directories..."
for dir in ${WORK_DIR}/*_histograms; do
    if [ -d "$dir" ]; then
        SLICE_NAME=$(basename "$dir" "_histograms")
        OUTPUT_FILE="${WORK_DIR}/sf_results_${SLICE_NAME}.npz"
        
        if [ ! -f "$OUTPUT_FILE" ]; then
            echo "  Merging: $SLICE_NAME"
            python $SFUNCTOR_DIR/combine_histograms_fast.py \
                --pattern "${dir}/histogram_*.npz" \
                --output "$OUTPUT_FILE" \
                --mode node \
                --workers 32
            
            if [ -f "$OUTPUT_FILE" ]; then
                echo "  ✓ Created: $(basename $OUTPUT_FILE)"
                rm -rf "$dir"
                echo "  ✓ Cleaned up histogram directory"
            fi
        fi
    fi
done

# Count slice results
NUM_SLICES=$(ls -1 ${WORK_DIR}/sf_results_Turb*.npz 2>/dev/null | wc -l)
echo "Found $NUM_SLICES slice results"

# Final merge of all slices
if [ $NUM_SLICES -gt 0 ] && [ ! -f "${WORK_DIR}/sf_results_all_slices.npz" ]; then
    echo "Performing final merge of all slices..."
    python $SFUNCTOR_DIR/combine_histograms_fast.py \
        --pattern "${WORK_DIR}/sf_results_Turb*.npz" \
        --output "${WORK_DIR}/sf_results_all_slices.npz" \
        --mode slice \
        --workers 32
    
    if [ -f "${WORK_DIR}/sf_results_all_slices.npz" ]; then
        echo "✓ Created final merged file: sf_results_all_slices.npz"
        SIZE=$(du -h "${WORK_DIR}/sf_results_all_slices.npz" | cut -f1)
        echo "  File size: $SIZE"
    fi
fi

# Generate plots
if [ -f "${WORK_DIR}/sf_results_all_slices.npz" ]; then
    echo "Generating plots..."
    python $SFUNCTOR_DIR/plot_structure_functions.py ${WORK_DIR}/sf_results_all_slices.npz
    
    NUM_PLOTS=$(ls -1 ${WORK_DIR}/*.png 2>/dev/null | wc -l)
    echo "✓ Generated $NUM_PLOTS plots"
fi

#============================================================================
# JOB 2: 10240 SW5 (Job 3665058)
#============================================================================

echo ""
echo "Processing 10240 SW5 (Job 3665058)..."
echo "----------------------------------------------------------------------------"

WORK_DIR="${BASE_DIR}/results_Turb_10240_beta25_dedt025_plm/ndisp100000_nrand100000_nell128_sw5_job3665058"
cd $WORK_DIR

# Check for any remaining histogram directories to merge
echo "Checking for unmerged histogram directories..."
for dir in ${WORK_DIR}/*_histograms; do
    if [ -d "$dir" ]; then
        SLICE_NAME=$(basename "$dir" "_histograms")
        OUTPUT_FILE="${WORK_DIR}/sf_results_${SLICE_NAME}.npz"
        
        if [ ! -f "$OUTPUT_FILE" ]; then
            echo "  Merging: $SLICE_NAME"
            python $SFUNCTOR_DIR/combine_histograms_fast.py \
                --pattern "${dir}/histogram_*.npz" \
                --output "$OUTPUT_FILE" \
                --mode node \
                --workers 32
            
            if [ -f "$OUTPUT_FILE" ]; then
                echo "  ✓ Created: $(basename $OUTPUT_FILE)"
                rm -rf "$dir"
                echo "  ✓ Cleaned up histogram directory"
            fi
        fi
    fi
done

# Count slice results
NUM_SLICES=$(ls -1 ${WORK_DIR}/sf_results_Turb*.npz 2>/dev/null | wc -l)
echo "Found $NUM_SLICES slice results"

# Final merge of all slices
if [ $NUM_SLICES -gt 0 ] && [ ! -f "${WORK_DIR}/sf_results_all_slices.npz" ]; then
    echo "Performing final merge of all slices..."
    python $SFUNCTOR_DIR/combine_histograms_fast.py \
        --pattern "${WORK_DIR}/sf_results_Turb*.npz" \
        --output "${WORK_DIR}/sf_results_all_slices.npz" \
        --mode slice \
        --workers 32
    
    if [ -f "${WORK_DIR}/sf_results_all_slices.npz" ]; then
        echo "✓ Created final merged file: sf_results_all_slices.npz"
        SIZE=$(du -h "${WORK_DIR}/sf_results_all_slices.npz" | cut -f1)
        echo "  File size: $SIZE"
    fi
fi

# Generate plots
if [ -f "${WORK_DIR}/sf_results_all_slices.npz" ]; then
    echo "Generating plots..."
    python $SFUNCTOR_DIR/plot_structure_functions.py ${WORK_DIR}/sf_results_all_slices.npz
    
    NUM_PLOTS=$(ls -1 ${WORK_DIR}/*.png 2>/dev/null | wc -l)
    echo "✓ Generated $NUM_PLOTS plots"
fi

#============================================================================
# SUMMARY
#============================================================================

echo ""
echo "============================================================================"
echo " COMPLETION SUMMARY"
echo "============================================================================"
echo " 10240 SW2 (Job 3665053):"
echo "   Directory: ${BASE_DIR}/results_Turb_10240_beta25_dedt025_plm/ndisp100000_nrand100000_nell128_sw2_job3665053"
if [ -f "${BASE_DIR}/results_Turb_10240_beta25_dedt025_plm/ndisp100000_nrand100000_nell128_sw2_job3665053/sf_results_all_slices.npz" ]; then
    echo "   Status: ✓ Complete"
else
    echo "   Status: ⚠ Check results"
fi

echo ""
echo " 10240 SW5 (Job 3665058):"
echo "   Directory: ${BASE_DIR}/results_Turb_10240_beta25_dedt025_plm/ndisp100000_nrand100000_nell128_sw5_job3665058"
if [ -f "${BASE_DIR}/results_Turb_10240_beta25_dedt025_plm/ndisp100000_nrand100000_nell128_sw5_job3665058/sf_results_all_slices.npz" ]; then
    echo "   Status: ✓ Complete"
else
    echo "   Status: ⚠ Check results"
fi

echo ""
echo " End Time: $(date)"
echo "============================================================================"