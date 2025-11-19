#!/bin/bash
# Script to check progress of beta25 10240 SF analysis

echo "========================================"
echo "Beta25 10240 SF Analysis Progress Check"
echo "========================================"
echo ""

# Define paths
BASE_DIR="/lustre/orion/ast207/proj-shared/dfielding/Production_plm"
RESULTS_BASE="${BASE_DIR}/sfunctor_results/results_Turb_10240_beta25_dedt025_plm"
SLICE_DIR="${BASE_DIR}/sfunctor_results/slice_Turb_10240_beta25_dedt025_plm"

# Check for recent job directories
echo "Recent job directories:"
if [ -d "$RESULTS_BASE" ]; then
    ls -lt "$RESULTS_BASE" 2>/dev/null | head -5 | while read line; do
        echo "  $line"
    done
else
    echo "  Results directory not found: $RESULTS_BASE"
fi

echo ""
echo "----------------------------------------"
echo "Job 3659610 Status (known from logs):"
echo "----------------------------------------"
echo "Directory: ${RESULTS_BASE}/ndisp100_000_nrand10_000_nell128_sw2_job3659610"
echo ""
echo "Completed slices (4 total before timeout):"
echo "  1. Turb_10240_beta25_dedt025_plm_axis1_slice0p125_file0011"
echo "  2. Turb_10240_beta25_dedt025_plm_axis1_slice0p375_file0011"
echo "  3. Turb_10240_beta25_dedt025_plm_axis1_slicem0p125_file0011"
echo "  4. Turb_10240_beta25_dedt025_plm_axis1_slicem0p375_file0011"
echo ""
echo "Status: Job timed out after 2 hours"
echo "Configuration:"
echo "  - 100,000 displacements (test run, 10x less than target)"
echo "  - 10,000 random subsamples (test run, 10x less than target)"
echo "  - 128 ell bins"
echo "  - Stencil width: 2"
echo "  - Nodes: 64"
echo ""

# Count total slices available
if [ -f "${BASE_DIR}/sfunctor_results/slice_list_Turb_10240_beta25_dedt025_plm.txt" ]; then
    TOTAL_SLICES=$(wc -l < "${BASE_DIR}/sfunctor_results/slice_list_Turb_10240_beta25_dedt025_plm.txt")
    echo "Total slices available: $TOTAL_SLICES"
    echo "Progress: 4 / $TOTAL_SLICES slices completed"
    echo "Remaining: $((TOTAL_SLICES - 4)) slices"
    
    # Estimate time needed
    # Assuming ~30 minutes per slice based on the 4 slices in 2 hours
    MINUTES_PER_SLICE=30
    REMAINING_MINUTES=$((MINUTES_PER_SLICE * (TOTAL_SLICES - 4)))
    REMAINING_HOURS=$((REMAINING_MINUTES / 60))
    echo ""
    echo "Estimated time to complete (at ~30 min/slice): ${REMAINING_HOURS} hours"
else
    echo "Slice list not found, checking slice directory..."
    if [ -d "$SLICE_DIR" ]; then
        TOTAL_SLICES=$(ls -1 "$SLICE_DIR"/*.npz 2>/dev/null | wc -l)
        if [ $TOTAL_SLICES -gt 0 ]; then
            echo "Total slices found: $TOTAL_SLICES"
            echo "Progress: 4 / $TOTAL_SLICES slices completed"
            echo "Remaining: $((TOTAL_SLICES - 4)) slices"
        else
            echo "No slice files found in $SLICE_DIR"
        fi
    else
        echo "Slice directory not found: $SLICE_DIR"
    fi
fi

echo ""
echo "========================================"
echo "Next Steps:"
echo "========================================"
echo "1. To resume from job 3659610:"
echo "   ./resume_distributed_analysis_beta25_10240.sh"
echo ""
echo "2. The script will:"
echo "   - Copy the 4 completed results"
echo "   - Reuse the existing displacements.npz"
echo "   - Continue processing remaining slices"
echo "   - Request another 2 hours (may need multiple runs)"
echo ""
echo "3. For production run with full parameters:"
echo "   Edit run_distributed_analysis_frontier_beta25_10240_sw2.sh:"
echo "   - Change N_DISP_TOTAL to 1_000_000"
echo "   - Change N_RANDOM_SUBSAMPLES to 100_000"
echo "   - Consider increasing time limit if needed"
echo ""