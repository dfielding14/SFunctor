#!/bin/bash
#SBATCH -A AST207
#SBATCH -J SF_EXTRACT
#SBATCH -o sf_extract_%j.out
#SBATCH -t 0:30:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks=12

# SLURM script to extract slices with work distribution
# Each task extracts one slice

echo "========================================"
echo "Slice Extraction"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Tasks: $SLURM_NTASKS"
echo "Date: $(date)"
echo "========================================"

# Configuration
SIM_NAME="Turb_640_beta25_dedt025_plm"
AXES=(1 2 3)
POSITIONS=(-0.375 -0.125 0.125 0.375)
FILE_NUMBERS=(100)

# Load required modules
module reset
module load gcc/9.3.0 python/.3.11-anaconda3

# Activate virtual environment
source /ccs/home/dfielding/SFunctor/venv_sfunctor/bin/activate

# Change to SFunctor directory
cd /autofs/nccs-svm1_home2/dfielding/SFunctor

# Calculate total number of jobs
TOTAL_JOBS=$((${#FILE_NUMBERS[@]} * ${#AXES[@]} * ${#POSITIONS[@]}))
echo "Total extraction jobs: $TOTAL_JOBS"

# Launch all extraction jobs in parallel using srun
job_id=0
for file_num in "${FILE_NUMBERS[@]}"; do
    for axis in "${AXES[@]}"; do
        for pos in "${POSITIONS[@]}"; do
            # Launch each job on a specific task
            srun --exclusive -N1 -n1 --cpu-bind=none bash -c "
                if [ \$SLURM_PROCID -eq $job_id ]; then
                    echo \"Task $job_id: Extracting file $file_num, axis $axis, position $pos\"
                    python extractor.py \
                        --sim_name \"$SIM_NAME\" \
                        --axis $axis \
                        --position $pos \
                        --file_number $file_num
                fi
            " &
            ((job_id++))
        done
    done
done

# Wait for all background jobs to complete
wait

echo ""
echo "========================================"
echo "Extraction Complete"
echo "========================================"

# List extracted files
echo "Extracted slices:"
ls -lh slice_data/*.npz 2>/dev/null | tail -20

echo ""
echo "Total slices: $(ls slice_data/*.npz 2>/dev/null | wc -l)"