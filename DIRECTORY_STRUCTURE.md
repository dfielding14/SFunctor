# SFunctor Directory Structure

## Overview

The SFunctor pipeline now uses a consistent directory structure under `sfunctor_results/`:

```
sfunctor_results/
├── slice_{SIM_NAME}/          # Extracted 2D slices
│   └── *.npz                  # Individual slice files
├── results_{SIM_NAME}/        # Analysis results
│   └── distributed_run_{JOB_ID}/  # Results from each run
│       ├── displacements.npz      # Generated displacement vectors
│       ├── histograms_*/          # Intermediate histogram files
│       └── sf_results_*.npz       # Final structure function results
└── slice_list_{SIM_NAME}.txt  # List of slice paths for batch processing
```

## Key Changes

1. **Slice Storage**: 
   - Old: `slice_data/` or `sfunctor_results/slices_{SIM_NAME}/`
   - New: `sfunctor_results/slice_{SIM_NAME}/`

2. **Results Storage**:
   - Old: Various locations
   - New: `sfunctor_results/results_{SIM_NAME}/`

3. **Scripts Updated**:
   - `scripts/production/extractor.py`: Now saves to `sfunctor_results/slice_{SIM_NAME}/`
   - `run_extraction.sh`: Creates slice list pointing to new location
   - `run_distributed_analysis.sh`: Saves results to `sfunctor_results/results_{SIM_NAME}/`
   - `test_extraction.sh`: Updated to check new locations
   - `test_distributed_single_slice.sh`: Updated paths

## Usage

1. **Extract slices**:
   ```bash
   sbatch run_extraction.sh
   ```
   This creates slices in `sfunctor_results/slice_{SIM_NAME}/`

2. **Run distributed analysis**:
   ```bash
   sbatch run_distributed_analysis.sh
   ```
   This reads from `slice_{SIM_NAME}/` and saves to `results_{SIM_NAME}/`

## Configuration

All paths are configured in the SLURM scripts via these variables:
- `SIM_NAME`: Simulation name (e.g., "Turb_640_beta25_dedt025_plm")
- `BASE_DIR`: Base directory where data and results are stored
- `SLICE_DIR`: `${BASE_DIR}/sfunctor_results/slice_${SIM_NAME}`
- `WORK_DIR`: `${BASE_DIR}/sfunctor_results/results_${SIM_NAME}/distributed_run_${SLURM_JOB_ID}`