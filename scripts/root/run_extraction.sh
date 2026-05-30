#!/bin/bash
#SBATCH -A AST207
#SBATCH -J SF_EXTRACT
#SBATCH -o logs/sf_extract.%j.out
#SBATCH -t 6:00:00
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=12

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
SIM_NAME="Turb_5120_beta1_dedt025_plm"
AXES=(1 2 3)
# Different positions for different axes
POSITIONS_AXIS1=(-0.375 -0.125 0.125 0.375)
POSITIONS_AXIS2=(-0.375 -0.125 0.125 0.375)
POSITIONS_AXIS3=(-1.125 -0.375 0.375 1.125)
FILE_NUMBERS=(-1)  # List of file numbers to extract (use -1 for the final file)

# Base directory where data and inputs are located
BASE_DIR="/lustre/orion/ast207/proj-shared/dfielding/Production_plm"

# Load required modules
module reset
module load gcc/9.3.0 python/.3.11-anaconda3

# Activate virtual environment
source /ccs/home/dfielding/SFunctor/venv_sfunctor/bin/activate

# Save current directory
SFUNCTOR_DIR="/ccs/home/dfielding/SFunctor"

# Check that required directories exist
if [ ! -d "${BASE_DIR}/data/data_${SIM_NAME}" ]; then
    echo "Error: Data directory not found: ${BASE_DIR}/data/data_${SIM_NAME}"
    exit 1
fi

if [ ! -f "${BASE_DIR}/inputs/${SIM_NAME}.athinput" ]; then
    echo "Error: Input file not found: ${BASE_DIR}/inputs/${SIM_NAME}.athinput"
    exit 1
fi

echo "Base directory: $BASE_DIR"
echo "Data directory exists: ${BASE_DIR}/data/data_${SIM_NAME}"
echo "Input file exists: ${BASE_DIR}/inputs/${SIM_NAME}.athinput"
echo ""

# ---------------------------------------------------------
# 1. Build list of argument strings – one element per slice
# ---------------------------------------------------------
ARGS_LIST=()
for file_num in "${FILE_NUMBERS[@]}"; do
    for axis in "${AXES[@]}"; do
        # Select the appropriate positions array based on the axis
        if [ $axis -eq 1 ]; then
            POSITIONS=("${POSITIONS_AXIS1[@]}")
        elif [ $axis -eq 2 ]; then
            POSITIONS=("${POSITIONS_AXIS2[@]}")
        elif [ $axis -eq 3 ]; then
            POSITIONS=("${POSITIONS_AXIS3[@]}")
        fi
        
        for pos in "${POSITIONS[@]}"; do
            ARGS_LIST+=("--sim_name $SIM_NAME --axis $axis --position $pos --file_number $file_num")
        done
    done
done

TOTAL_JOBS=${#ARGS_LIST[@]}
echo "Total extraction jobs: $TOTAL_JOBS"
echo ""

# Flatten list into a pipe-separated string so it can be exported
ARGS_STRING=$(printf "%s|" "${ARGS_LIST[@]}")
ARGS_STRING=${ARGS_STRING%|}          # trim trailing "|"
export ARGS_STRING                    # make visible to every srun rank

# ---------------------------------------------------------
# 2. Launch one rank per argument set
# ---------------------------------------------------------
srun --ntasks=$TOTAL_JOBS --chdir="$BASE_DIR" bash -c '
  # Re-inflate the list inside each task
  IFS="|" read -ra ALL_ARGS <<< "$ARGS_STRING"
  ARGS="${ALL_ARGS[$SLURM_PROCID]}"

  echo "Task $SLURM_PROCID: python '"${SFUNCTOR_DIR}"'/scripts/production/extractor.py $ARGS"
  python '"${SFUNCTOR_DIR}"'/scripts/production/extractor.py $ARGS
'
echo ""
echo "========================================"
echo "Extraction Complete"
echo "========================================"

# List extracted files
SLICE_DIR="${BASE_DIR}/sfunctor_results/slice_${SIM_NAME}"
echo "Extracted slices:"
ls -lh ${SLICE_DIR}/*.npz 2>/dev/null | tail -20

echo ""
echo "Total slices: $(ls ${SLICE_DIR}/*.npz 2>/dev/null | wc -l)"
echo ""
echo "Slices saved to: ${SLICE_DIR}"

# Create slice list file for distributed analysis
SLICE_LIST="${BASE_DIR}/sfunctor_results/slice_list_${SIM_NAME}.txt"
ls ${SLICE_DIR}/*.npz > $SLICE_LIST 2>/dev/null
echo ""
echo "Slice list created: $SLICE_LIST"
echo "Number of slices in list: $(wc -l < $SLICE_LIST)"