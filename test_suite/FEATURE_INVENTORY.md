# SFunctor Test Suite Inventory (Current API)

## Scope
This inventory reflects the **current unified histogram API** used by SFunctor:
- `hist`: `(N_CHANNELS, n_ell, n_theta, n_phi, n_delta)`
- `channels`: channel names aligned with `sfunctor.core.histograms.Channel`
- `delta_bin_edges`: per-channel delta bin edges

Current `N_CHANNELS` is 26.

## Implemented Test Scripts
- `00_setup_and_validation.py`
  - Validates input slice loading and basic field statistics.
- `01_core_structure_functions.py`
  - Runs `analyze_slice()` and generates core histogram/scale/angle plots.
- `02_physics_calculations.py`
  - Computes derived fields (vA, z±, vorticity/current/curvature/grad rho proxies) and validates histogram channels.
- `03_time_series.py`
  - Pseudo time-series diagnostics across available snapshots.
- `04_anisotropy.py`
  - Scale-dependent anisotropy diagnostics from angular histograms.

## Runner
- `run_comprehensive_tests.py`
  - Executes available numbered scripts.
  - Skips non-implemented modules (`05+`) cleanly.
  - Writes a summary report to `test_suite/results/test_report.txt`.

## Result Artifacts
Outputs are written under:
- `test_suite/results/plots/`
- `test_suite/results/data/`

These are generated artifacts and should generally not be committed.
