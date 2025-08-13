#!/bin/bash
#============================================================================
# TEST AUTO-CHAINING WITH SMALL 640 RESOLUTION JOB
#============================================================================
# This script tests the auto-chaining functionality with a small, quick job
# Should complete some work in 5 minutes then auto-submit continuation
#============================================================================

# Copy the autochain script and modify for test
cp /autofs/nccs-svm1_home2/dfielding/SFunctor/run_sf_analysis_frontier_autochain.sh \
   /autofs/nccs-svm1_home2/dfielding/SFunctor/run_test_640_autochain.sh

# Modify the configuration in the copied script
cat > /tmp/test_config.sed << 'EOF'
# Simulation configuration
s/^RESOLUTION="10240"/RESOLUTION="640"/
s/^BETA="25"/BETA="25"/
s/^STENCIL_WIDTH=3/STENCIL_WIDTH=2/
s/^N_DISP_TOTAL=100000/N_DISP_TOTAL=12500/
s/^N_RANDOM_SUBSAMPLES=100000/N_RANDOM_SUBSAMPLES=12500/
s/^N_ELL_BINS=128/N_ELL_BINS=128/

# HPC Configuration - 2 nodes for 640, 5 minute test
s/^N_NODES=64/N_NODES=2/
s/^TIME_LIMIT="4:00:00"/TIME_LIMIT="0:05:00"/
s/^QUEUE="debug"/QUEUE="debug"/

# Auto-continuation - 1 minute buffer for testing
s/^TIME_BUFFER_MINUTES=15/TIME_BUFFER_MINUTES=1/
s/^MAX_CHAIN_JOBS=10/MAX_CHAIN_JOBS=3/  # Limit chains for testing

# Clear custom bin edges for test
s/^LOG_SF_BIN_EDGES_MIN=.*/LOG_SF_BIN_EDGES_MIN=""/
s/^LOG_SF_BIN_EDGES_MAX=.*/LOG_SF_BIN_EDGES_MAX=""/

# Update SBATCH directives
s/#SBATCH --time=4:00:00/#SBATCH --time=0:05:00/
s/#SBATCH -q debug/#SBATCH -q debug/
s/#SBATCH --nodes=64/#SBATCH --nodes=2/
EOF

# Apply the configuration changes
sed -f /tmp/test_config.sed -i /autofs/nccs-svm1_home2/dfielding/SFunctor/run_test_640_autochain.sh

echo "Test script created: run_test_640_autochain.sh"
echo ""
echo "Configuration for test:"
echo "  Resolution: 640"
echo "  Beta: 25"
echo "  Stencil Width: 2"
echo "  Nodes: 2"
echo "  Time Limit: 5 minutes"
echo "  Time Buffer: 1 minute (will trigger continuation after 4 minutes)"
echo "  Max Chain Jobs: 3"
echo "  Displacements: 12,500"
echo "  Random Samples: 12,500"
echo ""
echo "To submit the test job:"
echo "  sbatch run_test_640_autochain.sh"
echo ""
echo "The job should:"
echo "1. Start processing slices"
echo "2. After ~4 minutes, detect approaching time limit"
echo "3. Submit a continuation job automatically"
echo "4. Exit cleanly to let continuation take over"
echo ""
echo "Monitor with:"
echo "  squeue -u $USER"
echo "  tail -f sf_test_640_*.out"