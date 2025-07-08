#!/bin/bash
# Simple test script for slice extraction

echo "Testing slice extraction"
echo "======================="

# Set paths
SFUNCTOR_DIR="/autofs/nccs-svm1_home2/dfielding/SFunctor"
TEST_DATA="/lustre/orion/ast207/proj-shared/dfielding/Production_plm/data/Turb.hydro.00100.bin"
OUTPUT_DIR="./test_extraction_$(date +%Y%m%d_%H%M%S)"

# Create output directory
mkdir -p $OUTPUT_DIR

# Test extraction along different axes
echo "Extracting test slices..."

# Extract along z-axis (xy plane)
python ${SFUNCTOR_DIR}/extract_2d_slice.py \
    --input "$TEST_DATA" \
    --output "$OUTPUT_DIR/test_slice_z.npz" \
    --axis 3 \
    --position 0.5

# Extract along y-axis (xz plane)
python ${SFUNCTOR_DIR}/extract_2d_slice.py \
    --input "$TEST_DATA" \
    --output "$OUTPUT_DIR/test_slice_y.npz" \
    --axis 2 \
    --position 0.5

# Extract along x-axis (yz plane)
python ${SFUNCTOR_DIR}/extract_2d_slice.py \
    --input "$TEST_DATA" \
    --output "$OUTPUT_DIR/test_slice_x.npz" \
    --axis 1 \
    --position 0.5

echo ""
echo "Extraction complete. Results in: $OUTPUT_DIR"
ls -lh $OUTPUT_DIR/*.npz

# Check contents of one slice
echo ""
echo "Checking slice contents:"
python -c "
import numpy as np
data = np.load('$OUTPUT_DIR/test_slice_z.npz')
print(f'Keys: {list(data.keys())}')
print(f'Shape of rho: {data[\"rho\"].shape}')
print(f'Axis: {data[\"axis\"]}')
print(f'Position: {data[\"slice_position\"]}')
"