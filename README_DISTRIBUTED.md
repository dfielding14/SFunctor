# Distributed Structure Function Analysis

This is a simplified approach to running structure function analysis across multiple nodes without MPI complexity.

## Overview

The analysis is split into independent components:
1. **Generate displacements** - Create displacement vectors once
2. **Node analysis** - Each node processes a subset of displacements 
3. **Combine results** - Merge partial histograms into final result

## Quick Start

### Test Locally
```bash
./test_distributed_single_slice.sh
```

### Run on Cluster
```bash
sbatch run_distributed_analysis.sh
```

## Scripts

- `generate_displacements.py` - Generate displacement vectors
- `run_node_analysis.py` - Process subset of displacements on one node
- `combine_histograms.py` - Combine partial results
- `run_distributed_analysis.sh` - SLURM orchestration script

## Configuration

Edit parameters in `run_distributed_analysis.sh`:
- `N_DISP_TOTAL` - Total number of displacement vectors
- `N_RANDOM_SUBSAMPLES` - Random samples per displacement
- `STRIDE` - Downsampling factor for slice data

## How It Works

1. Each node gets assigned a range of displacements
2. Nodes work independently with local multiprocessing
3. Results are saved as partial histograms
4. Final step combines all partial results

No MPI, no shared memory issues, simple and robust.