#!/bin/bash
# Simple test script for slice extraction

echo "Testing slice extraction"
echo "======================="

# Set paths
SFUNCTOR_DIR="/autofs/nccs-svm1_home2/dfielding/SFunctor"
BASE_DIR="/lustre/orion/ast207/proj-shared/dfielding/Production_plm"
SIM_NAME="Turb_640_beta25_dedt025_plm"
FILE_NUM=24

# Load required modules
module reset
module load gcc/9.3.0 python/.3.11-anaconda3

# Activate virtual environment
source /ccs/home/dfielding/SFunctor/venv_sfunctor/bin/activate

# Change to base directory where data is located
cd $BASE_DIR

echo "Working from: $(pwd)"
echo "Testing extraction for: $SIM_NAME, file $FILE_NUM"
echo ""

# Test extraction along different axes
echo "Extracting test slices..."

# Extract along z-axis (xy plane)
echo "Extracting z-axis slice..."
python ${SFUNCTOR_DIR}/extractor.py \
    --sim_name "$SIM_NAME" \
    --axis 3 \
    --position 0.0 \
    --file_number $FILE_NUM

# Extract along y-axis (xz plane)
echo ""
echo "Extracting y-axis slice..."
python ${SFUNCTOR_DIR}/extractor.py \
    --sim_name "$SIM_NAME" \
    --axis 2 \
    --position 0.0 \
    --file_number $FILE_NUM

# Extract along x-axis (yz plane)
echo ""
echo "Extracting x-axis slice..."
python ${SFUNCTOR_DIR}/extractor.py \
    --sim_name "$SIM_NAME" \
    --axis 1 \
    --position 0.0 \
    --file_number $FILE_NUM

echo ""
echo "Extraction complete. Checking results..."
echo ""

# List extracted files
SLICE_DIR="${BASE_DIR}/sfunctor_results/slice_${SIM_NAME}"
echo "Extracted slices in ${SLICE_DIR}:"
ls -lh ${SLICE_DIR}/${SIM_NAME}_axis*_file00${FILE_NUM}.npz 2>/dev/null

# Check contents of one slice
echo ""
echo "Checking slice contents:"
python -c "
import numpy as np
import glob

# Find a test slice
slices = glob.glob('${SLICE_DIR}/${SIM_NAME}_axis3_*_file00${FILE_NUM}.npz')
if slices:
    data = np.load(slices[0])
    print(f'File: {slices[0].split(\"/\")[-1]}')
    print(f'Keys: {list(data.keys())}')
    if 'dens' in data:
        print(f'Shape of density: {data[\"dens\"].shape}')
    print(f'Available fields: {[k for k in data.keys() if not k.startswith(\"_\")]}')
else:
    print('No slice files found!')
"