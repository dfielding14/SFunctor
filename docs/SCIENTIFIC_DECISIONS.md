# Scientific Decisions And Follow-Up Policy

## Compressible Elsasser Convention

Use fixed-reference-density Elsasser-like variables as the primary
literature-comparison baseline:

```text
z_plus_ref  = u + B / sqrt(rho0)
z_minus_ref = u - B / sqrt(rho0)
```

For production comparisons, pass the physically documented initial uniform
density as `rho0`. Retain the pointwise-density alternatives:

```text
z_plus  = u + B / sqrt(rho)
z_minus = u - B / sqrt(rho)
```

as compressibility-sensitivity diagnostics. Do not silently substitute one
convention for the other. Pointwise density changes both increment amplitudes
and the `e_xi` direction.

## Preliminary 2-D Matrix

The reproducible driver is
`scripts/production/compare_density_conventions.py`. A preliminary matrix was
run on May 30, 2026 using the three available orthogonal
`Turb_320_beta100_dedt025_plm` slices, `rho0=1`, stride `8`, `500` origins per
displacement, seed `23`, wedge widths `10`, `15`, and `20` degrees, fit
interval `[1, 16]`, and minimum count `20`.

Across the `54` local-frame comparisons for `parallel`, `xi`, and `lambda`:

- median symmetric relative `S2` differences ranged from `0.28%` to `6.19%`;
- the maximum absolute fitted-slope difference was `0.0564`;
- no fit failed the exploratory quality gate;
- accepted directional fractions ranged from `1.07%` to `9.55%`.

These values support the baseline/robustness split above for this exploratory
snapshot. They are not a final scientific conclusion. Repeat the matrix across
representative times and beta values before publication analysis.

## Restricted 2-D Coverage

Treat strict slice results as provisional slice-conditioned measurements.
Save `coverage_rows`, inspect accepted fractions and eligible bins, and retain
separate products for each slice normal. Do not silently pool sparse
directions. Definitive statistical-eddy shapes require unrestricted 3-D pair
sampling.

## Rank-Local I/O

`sfunctor/io/rank_manifest.py` is the storage contract for future 3-D work:

1. scan actual rank-file metadata without loading cell payloads;
2. validate contiguous rank-file coverage, common snapshot metadata, block
   geometry, and unique `(level, logical_location)` ownership;
3. group selected block payload reads by rank file;
4. keep rank, local block index, refinement level, geometry, and provenance
   attached to later chunk iterators.

Current 2-D extraction consumes the manifest but intentionally requires an
exact level-0 uniform layout. Add an AMR-aware chunk iterator for 3-D rather
than weakening that assertion.

## Legacy Histogram Censoring

New legacy products save `hist_censoring` and `censor_names`. Calibrate one
fixed set of per-channel delta edges before distributed comparisons. Warn when
underflow plus overflow exceeds `0.1%`; justify quantitative use above `1%`.
Undefined values are reported separately as `invalid`.
