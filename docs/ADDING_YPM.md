# Adding Y^± (third-order mixed structure functions) to the pipeline

## Goal
Capture the mixed third-order structure functions

\[ Y^+(\ell, \theta, \phi) = |\delta z^+|^2 \; (\delta z^- \cdot \hat{r}) \]
\[ Y^-(\ell, \theta, \phi) = |\delta z^-|^2 \; (\delta z^+ \cdot \hat{r}) \]

for each displacement bin (\ell, \theta, \phi), accumulating both sum and count so that later we can form averages and higher moments. These should be stored alongside existing histograms and survive combination across nodes/slices.

## Data shapes
- `hist_Ypm_sum`: float64, shape `(2, n_ell, n_theta, n_phi)`
  - index 0: Y^+, index 1: Y^- sums
- `hist_Ypm_count`: int64, shape `(n_ell, n_theta, n_phi)`

## Code changes (high level)

1) **histograms.py**
   - `_compute_histogram_core`: add parameters `hist_Ypm_sum`, `hist_Ypm_count` and accumulate Y^± per sample:
     - Compute \hat{r} = (dx, dy, dz)/r
     - Compute |\delta z^±|^2 and longitudinal components \delta z_L^∓ = \delta z^∓ · \hat{r}` (signed)
     - Y^+ = |\delta z^+|^2 * \delta z^-_L, Y^- = |\delta z^-|^2 * \delta z^+_L
     - Increment sums and count in the (ell, theta, phi) bin if indices valid.
   - All stencil-specific `compute_histogram_for_disp_2D_stencil{2,3,5}` need to allocate and return the Ypm arrays, forwarding them into `_compute_histogram_core`.

2) **sfunctor/core/parallel.py**
   - `_process_batch`: allocate zeros for Ypm sum/count, receive them from `compute_histogram_for_disp_2D`, accumulate, and return `(hist, hist_Ypm_sum, hist_Ypm_count)`.
   - `compute_histograms_shared`: update return signature to include Ypm; adjust multiprocessing reduction to sum the new arrays.

3) **Drivers** (`sfunctor/analysis/single_slice.py`, `sfunctor/analysis/batch.py`, `scripts/production/run_node_analysis.py`)
   - Expect unified `hist` plus the two Ypm arrays from `compute_histograms_shared`.
   - Save `hist_Ypm_sum` and `hist_Ypm_count` into per-node NPZ outputs.

4) **Combine scripts** (`scripts/production/combine_histograms.py`, `..._fast.py`)
   - When loading per-node/per-slice NPZ, if Ypm arrays exist, sum them across inputs.
   - Preserve them in combined outputs as `hist_Ypm_sum` and `hist_Ypm_count` (counts summed, sums summed).
   - Ensure metadata is untouched and falls back gracefully if arrays are absent (for backward compatibility).

5) **Plotting / downstream**
   - No immediate change required; plotting should ignore missing Ypm. Later, we can add S3/Y± visualizations using the saved arrays.

## Testing/checks
- Unit-style check: build a small synthetic slice, run `compute_histograms_shared` with a couple of displacements, and verify shapes/non-zero Ypm bins.
- Integration: rerun a short job (tiny ndisp/nrand) to confirm NPZs now contain `hist_Ypm_sum`/`hist_Ypm_count` and combine step succeeds.
- Backward compatibility: combine/plot should still work on older NPZs without Ypm; code paths should check for key existence.

## Notes
- Bin edges unchanged; we only add new accumulators keyed by (ell, theta, phi).
- Keep Ypm accumulation in float64 to preserve sum accuracy.
- Use the same folded theta/phi definitions as existing histograms: both
  angles are in `[0, pi/2]` because absolute projections identify opposite
  directions.
