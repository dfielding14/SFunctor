#!/bin/bash
#============================================================================
# SUBMIT ALL JOBS FROM Jobs_to_do.md - SIMPLE VERSION (NO AUTO-CHAINING)
#============================================================================

echo "============================================================================"
echo " STRUCTURE FUNCTION ANALYSIS - BATCH JOB SUBMISSION"
echo " Date: $(date)"
echo "============================================================================"

# Track submissions
SUBMITTED_JOBS=()
FAILED_SUBMISSIONS=()

# Function to create and submit a job
submit_job() {
    local RESOLUTION=$1
    local BETA=$2
    local STENCIL_WIDTH=$3
    local N_DISP=$4
    local N_RAND=$5
    local N_NODES=$6
    local DESCRIPTION=$7
    
    # Create job script
    JOB_SCRIPT="/autofs/nccs-svm1_home2/dfielding/SFunctor/job_scripts/production/job_${RESOLUTION}_beta${BETA}_sw${STENCIL_WIDTH}.sh"
    mkdir -p /autofs/nccs-svm1_home2/dfielding/SFunctor/job_scripts
    
    cat > $JOB_SCRIPT << EOF
#!/bin/bash
#SBATCH -A ast207
#SBATCH -J sf_${RESOLUTION}_b${BETA}_sw${STENCIL_WIDTH}
#SBATCH -o sf_%x_%j.out
#SBATCH -e sf_%x_%j.err
#SBATCH --time=2:00:00
#SBATCH -p batch
#SBATCH --nodes=$N_NODES
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56

# Configuration
SIM_TYPE="Turb"
RESOLUTION="$RESOLUTION"
BETA="$BETA"
DEDT="025"
SCHEME="plm"
STENCIL_WIDTH=$STENCIL_WIDTH
N_DISP_TOTAL=$N_DISP
N_RANDOM_SUBSAMPLES=$N_RAND
N_ELL_BINS=128
STRIDE=1
SEED=42

echo "============================================================================"
echo " STRUCTURE FUNCTION ANALYSIS"
echo " Resolution: \$RESOLUTION, Beta: \$BETA, Stencil Width: \$STENCIL_WIDTH"
echo "============================================================================"
echo " Job ID: \$SLURM_JOB_ID"
echo " Start Time: \$(date)"
echo " Nodes: \$SLURM_JOB_NUM_NODES"
echo "============================================================================"

# Load modules
module reset
module load PrgEnv-gnu

# Activate virtual environment
source /autofs/nccs-svm1_home2/dfielding/SFunctor/venv_frontier/bin/activate

# Set paths
BASE_DIR="/lustre/orion/ast207/proj-shared/dfielding/Production_plm"
SFUNCTOR_DIR="/autofs/nccs-svm1_home2/dfielding/SFunctor"
SIM_NAME="\${SIM_TYPE}_\${RESOLUTION}_beta\${BETA}_dedt\${DEDT}_\${SCHEME}"
RUN_NAME="ndisp\${N_DISP_TOTAL}_nrand\${N_RANDOM_SUBSAMPLES}_nell\${N_ELL_BINS}_sw\${STENCIL_WIDTH}"
WORK_DIR="\${BASE_DIR}/sfunctor_results/results_\${SIM_NAME}/\${RUN_NAME}_job\${SLURM_JOB_ID}"

mkdir -p \$WORK_DIR
cd \$WORK_DIR

# Save configuration
cat > config.txt << CONFIG_EOF
SIM_NAME=\$SIM_NAME
RESOLUTION=\$RESOLUTION
BETA=\$BETA
STENCIL_WIDTH=\$STENCIL_WIDTH
N_DISP_TOTAL=\$N_DISP_TOTAL
N_RANDOM_SUBSAMPLES=\$N_RANDOM_SUBSAMPLES
N_ELL_BINS=\$N_ELL_BINS
STRIDE=\$STRIDE
N_NODES=\$SLURM_JOB_NUM_NODES
CONFIG_EOF

# Generate slice list
SLICE_LIST="\${BASE_DIR}/sfunctor_results/slice_list_\${SIM_NAME}.txt"
if [ ! -f "\$SLICE_LIST" ]; then
    SLICE_DIR="\${BASE_DIR}/sfunctor_results/slice_\${SIM_NAME}"
    [ ! -d "\$SLICE_DIR" ] && SLICE_DIR="\${BASE_DIR}/slice_\${SIM_NAME}"
    ls \$SLICE_DIR/*.npz > \$SLICE_LIST 2>/dev/null || exit 1
fi

TOTAL_SLICES=\$(wc -l < \$SLICE_LIST)
echo " Total slices: \$TOTAL_SLICES"

# Generate displacements
echo " Generating displacements..."
python \$SFUNCTOR_DIR/scripts/production/generate_displacements.py \\
    --n_disp_total \$N_DISP_TOTAL \\
    --n_ell_bins \$N_ELL_BINS \\
    --Nres \$RESOLUTION \\
    --seed \$SEED

# Get nodes
NODES=(\$(scontrol show hostname \$SLURM_JOB_NODELIST))
TOTAL_NODES=\${#NODES[@]}

# Process slices
echo ""
echo " Processing Slices"
echo "============================================================================"

mapfile -t ALL_SLICES < \$SLICE_LIST
SLICE_COUNT=0

for SLICE_PATH in "\${ALL_SLICES[@]}"; do
    SLICE_NAME=\$(basename \$SLICE_PATH .npz)
    SLICE_COUNT=\$((SLICE_COUNT + 1))
    
    echo " [\$SLICE_COUNT/\$TOTAL_SLICES] Processing: \$SLICE_NAME"
    
    SLICE_OUTPUT_DIR="\$WORK_DIR/\${SLICE_NAME}_histograms"
    mkdir -p \$SLICE_OUTPUT_DIR
    
    # Launch on all nodes
    for NODE_ID in \$(seq 0 \$((TOTAL_NODES - 1))); do
        NODE=\${NODES[\$NODE_ID]}
        
        srun --exclusive -N 1 -n 1 -w \$NODE --cpus-per-task=56 \\
            python \$SFUNCTOR_DIR/scripts/production/run_node_analysis.py \\
            --slice \$SLICE_PATH \\
            --displacements \$WORK_DIR/displacements.npz \\
            --output_dir \$SLICE_OUTPUT_DIR \\
            --stride \$STRIDE \\
            --N_random_subsamples \$N_RANDOM_SUBSAMPLES \\
            --stencil_width \$STENCIL_WIDTH \\
            --n_processes 56 \\
            --node_id \$NODE_ID \\
            --total_nodes \$TOTAL_NODES \\
            > \$SLICE_OUTPUT_DIR/node_\${NODE_ID}.log 2>&1 &
    done
    
    wait
done

# Merge histograms
echo ""
echo " Merging histograms..."

for dir in \$WORK_DIR/*_histograms; do
    [ -d "\$dir" ] || continue
    SLICE_NAME=\$(basename "\$dir" "_histograms")
    
    python \$SFUNCTOR_DIR/scripts/production/combine_histograms_fast.py \\
        --pattern "\${dir}/histogram_*.npz" \\
        --output "\$WORK_DIR/sf_results_\${SLICE_NAME}.npz" \\
        --mode node \\
        --workers 32
    
    [ -f "\$WORK_DIR/sf_results_\${SLICE_NAME}.npz" ] && rm -rf "\$dir"
done

# Final merge
echo " Final merge..."
python \$SFUNCTOR_DIR/scripts/production/combine_histograms_fast.py \\
    --pattern "\$WORK_DIR/sf_results_\${SIM_TYPE}*.npz" \\
    --output "\$WORK_DIR/sf_results_all_slices.npz" \\
    --mode slice \\
    --workers 32

# Generate plots
echo " Generating plots..."
python \$SFUNCTOR_DIR/plotting_scripts/plot_structure_functions.py \$WORK_DIR/sf_results_all_slices.npz

echo ""
echo "============================================================================"
echo " COMPLETE"
echo " Results: \$WORK_DIR"
echo " End Time: \$(date)"
echo "============================================================================"
EOF
    
    chmod +x $JOB_SCRIPT
    
    # Submit job
    echo " Submitting: $DESCRIPTION"
    JOB_OUTPUT=$(sbatch $JOB_SCRIPT 2>&1)
    
    if [ $? -eq 0 ]; then
        JOB_ID=$(echo $JOB_OUTPUT | awk '{print $4}')
        echo "   ✓ Submitted: Job ID $JOB_ID"
        SUBMITTED_JOBS+=("$JOB_ID:$DESCRIPTION")
    else
        echo "   ✗ Failed: $JOB_OUTPUT"
        FAILED_SUBMISSIONS+=("$DESCRIPTION")
    fi
}

#============================================================================
# BETA SURVEY AT 5120 (12 jobs)
#============================================================================

echo ""
echo " BETA SURVEY AT 5120 RESOLUTION"
echo "============================================================================"

# Stencil width = 2
submit_job 5120 1 2 50000 50000 64 "5120 β=1 SW=2"
submit_job 5120 6 2 50000 50000 64 "5120 β=6 SW=2"
submit_job 5120 25 2 50000 50000 64 "5120 β=25 SW=2"
submit_job 5120 100 2 50000 50000 64 "5120 β=100 SW=2"

# Stencil width = 3
submit_job 5120 1 3 50000 50000 64 "5120 β=1 SW=3"
submit_job 5120 6 3 50000 50000 64 "5120 β=6 SW=3"
submit_job 5120 25 3 50000 50000 64 "5120 β=25 SW=3"
submit_job 5120 100 3 50000 50000 64 "5120 β=100 SW=3"

# Stencil width = 5
submit_job 5120 1 5 50000 50000 64 "5120 β=1 SW=5"
submit_job 5120 6 5 50000 50000 64 "5120 β=6 SW=5"
submit_job 5120 25 5 50000 50000 64 "5120 β=25 SW=5"
submit_job 5120 100 5 50000 50000 64 "5120 β=100 SW=5"

#============================================================================
# RESOLUTION STUDY AT BETA=25 (11 jobs - excluding done 10240 SW=3)
#============================================================================

echo ""
echo " RESOLUTION STUDY AT BETA=25"
echo "============================================================================"

# Stencil width = 2
submit_job 640 25 2 12500 12500 2 "640 β=25 SW=2"
submit_job 2560 25 2 25000 25000 32 "2560 β=25 SW=2"
# 5120 β=25 SW=2 already in beta survey above
submit_job 10240 25 2 100000 100000 128 "10240 β=25 SW=2"

# Stencil width = 3
submit_job 640 25 3 12500 12500 2 "640 β=25 SW=3"
submit_job 2560 25 3 25000 25000 32 "2560 β=25 SW=3"
# 5120 β=25 SW=3 already in beta survey above
# 10240 β=25 SW=3 is marked as done [x]

# Stencil width = 5
submit_job 640 25 5 12500 12500 2 "640 β=25 SW=5"
submit_job 2560 25 5 25000 25000 32 "2560 β=25 SW=5"
# 5120 β=25 SW=5 already in beta survey above
submit_job 10240 25 5 100000 100000 128 "10240 β=25 SW=5"

#============================================================================
# SUMMARY
#============================================================================

echo ""
echo "============================================================================"
echo " SUBMISSION SUMMARY"
echo "============================================================================"
echo " Total jobs submitted: ${#SUBMITTED_JOBS[@]}"
echo " Failed submissions: ${#FAILED_SUBMISSIONS[@]}"

if [ ${#SUBMITTED_JOBS[@]} -gt 0 ]; then
    echo ""
    echo " Successfully Submitted:"
    for job in "${SUBMITTED_JOBS[@]}"; do
        echo "   $job"
    done
fi

if [ ${#FAILED_SUBMISSIONS[@]} -gt 0 ]; then
    echo ""
    echo " Failed:"
    for job in "${FAILED_SUBMISSIONS[@]}"; do
        echo "   $job"
    done
fi

echo ""
echo " Monitor with: squeue -u $USER"
echo "============================================================================"
