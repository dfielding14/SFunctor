# Scientific Audit Report

Audit date: May 30, 2026. Scope: CPU calculations on existing 2-D slices only.
The full rank-local 3-D subvolume calculation was not started.

## Architecture Summary

See `docs/ARCHITECTURE.md` for the file map and active data flow. The important
architectural correction is separation of two products:

1. The legacy unified histogram remains a broad 26-channel PDF and alignment
   diagnostic with normalized 2-, 3-, and 5-point increment filters.
2. The new strict directional module is the primary two-point
   Chen/Mallet-style calculator for `ell_parallel`, `xi`, and `lambda`.

## Scientific Issues Found And Corrected

| original behavior | correction |
| --- | --- |
| Every legacy channel used one `phi` defined from `delta B_perp`; this cannot represent Mallet-style `z+` and `z-` frames. | Added strict per-`q` bases. Each variant uses its own `delta q_perp` for `e_xi` and measurement. |
| Legacy magnitude channels used full-vector increments, not `|delta q_perp|`. | Strict path directly accumulates `<|delta q_perp|^2>`. |
| Pixel offsets were used as physical separations. | Added explicit Cartesian `cell_sizes` and KJI-aware offset conversion to strict and low-level legacy paths. |
| Undefined legacy `phi` is historically assigned to zero. | Strict path excludes undefined `phi` by reason while retaining theta-only statistics. Legacy behavior is documented as unsuitable for rigorous three-direction cuts. |
| Widths `3` and `5` were described as more accurate stencils. | Documented them as distinct normalized increment filters. Only width `2` is directly comparable to pairwise literature definitions. |
| Compressible Elsasser convention was implicit. | Added and labeled pointwise-density and fixed-`rho0` variants plus `B`, `vA`, `vA_ref`, and `u`. |
| Global-field and local-field conditioning were not available as a controlled comparison. | Strict results retain local primary and global sanity-check frames. |
| Plot wedge-width arguments were ignored and sparse `NaN` cells poisoned weighted averages. | Wired configured widths and made reduction count-first over finite populated cells. |
| Alignment sine ratios were plotted as angles; cross-run comparison inferred first moments from saved second moments. | Primary plotting applies `arcsin`; invalid cross-run reconstruction is disabled explicitly. |

## Numerical And Production Issues Corrected

| original behavior | correction |
| --- | --- |
| Spatial origins used unseeded population-sized `choice(..., replace=False)` inside every legacy offset. | Use seeded `O(k)` replacement draws with stable per-offset seeds independent of worker scheduling. |
| Every offset allocated and returned a full dense 5-D histogram. | Shared path requests a one-`ell` slab and accumulates it into the worker batch histogram. |
| Parent retained all dense worker results before reduction. | Stream unordered batch results into the parent accumulator. |
| Combiners cast exact counts to `float64`, retained large arrays, warned on missing nodes, and did not validate schemas. | Added streaming `int64` combination, compatibility checks, exact node coverage checks, and per-slice metadata retention. |
| Extraction caches passed if each primitive had any finite value. | Reject partially filled caches, quarantine invalid cache files before recomputation, and abort before saving any incomplete produced plane. |
| Extraction could mix latest snapshots across ranks and write non-atomically; lock helper was inactive. | Select one rank-0 snapshot basename, require it on every rank, activate locking, and atomically replace caches. |
| Legacy `compute_vA` silently clamped tiny density and treated non-finite density inconsistently. | Warn for invalid and tiny densities; zero invalid values. Strict path excludes invalid-density pairs explicitly. |
| Rectangular convenience analysis used only the first slice dimension for its offset limit. | Use the minimum in-plane extent. |
| Legacy histogram delta censoring was silent. | Save accepted, underflow, overflow, and invalid counts by channel and angular bin; preserve them through combination and surface warnings in plotting. |
| Strict 2-D directional fits could appear valid despite sparse directional coverage. | Save occupancy rows and gate fitted slopes by configurable accepted counts, fractions, and eligible-bin counts. |
| Slice extraction inferred static Morton rank ownership and repeatedly reparsed rank files for central and derivative-neighbor planes. | Build a file-derived manifest, validate exact level-0 ownership for 2-D extraction, and group selected block payload reads by rank file. |

## Validation Baseline

The trusted loop oracle is `sfunctor/reference.py`. It independently enumerates
periodic origins and offsets and accumulates strict directional statistics.
The optimized implementation agrees with it for counts, sums, sum-of-squares,
attempts, and exclusions.

The regression suite covers:

- KJI extraction and offset mapping for all slice normals on an elongated grid;
- physical-spacing radial placement;
- basis normalization and orthogonality;
- sign invariance of `e_xi` and `e_lambda`;
- local/global frames and every compressible `q` variant;
- invalid density, entirely invalid `B`, canceling `B_loc`, undefined `phi`,
  and sparse bins;
- invalid-cache quarantine before extraction recomputation;
- periodic translation and fixed-seed reproducibility;
- one- versus two-process legacy equality;
- legacy accepted-count equality with histogram sums, explicit underflow,
  overflow, and invalid counters, and censor-side-array combination;
- analytic periodic-sine `S2`;
- controlled `xi` versus `lambda` classification;
- log-log slopes and constant-`S2` aspect ratios;
- rank-manifest metadata scanning, grouped one-read-per-rank payload access,
  and rejection of duplicate ownership and missing rank IDs;
- exact combiner counts, schema mismatch rejection, and incomplete-node rejection.

Focused follow-up suite result: `63 passed`. Final full-suite result:
`128 passed in 2.26s` after running `python -m pytest -q`.

The preliminary three-axis, three-wedge density-convention matrix is recorded
in `docs/SCIENTIFIC_DECISIONS.md`. It is an exploratory one-snapshot check, not
a publication result.

## CPU Benchmarks

See `docs/BENCHMARKING.md` for commands and the measured table. Key local
results:

- one-`ell` legacy slab: `18.60x` faster, `67.72 MiB -> 1.41 MiB` temporary;
- vectorized strict path: `84.60x` faster than loop reference on a real
  strided slice, with exact counts and sums agreeing at `1e-13` tolerance;
- complete benchmark-driver peak RSS: `375680 KiB`.

## Modified Files

The current uncommitted audit worktree contains the following reviewed changes.
This inventory includes earlier preserved axis-ordering work that the audit
integrated rather than discarding:

- configuration and top-level documentation: `.gitignore`, `README.md`,
  `README_DISTRIBUTED.md`, `docs/ADDING_YPM.md`,
  `docs/ANISOTROPIC_STRUCTURE_FUNCTIONS.md`,
  `docs/HISTOGRAM_ANALYSIS_GUIDE.md`, `docs/MEASURING_ANISOTROPY.md`,
  `docs/README.md`, `docs/ARCHITECTURE.md`, `docs/AUDIT_REPORT.md`,
  `docs/BENCHMARKING.md`, `docs/THREE_DIRECTION_ANALYSIS.md`,
  `docs/SCIENTIFIC_DECISIONS.md`;
- strict and reference calculations: `sfunctor/core/directional.py`,
  `sfunctor/analysis/directional.py`, `sfunctor/reference.py`,
  `scripts/production/run_directional_analysis.py`,
  `scripts/production/compare_density_conventions.py`,
  `scripts/benchmarks/benchmark_cpu.py`;
- legacy CPU path and exports: `sfunctor/__init__.py`,
  `sfunctor/core/__init__.py`, `sfunctor/core/histograms.py`,
  `sfunctor/core/parallel.py`, `sfunctor/core/physics.py`,
  `sfunctor/analysis/batch.py`, `sfunctor/analysis/single_slice.py`,
  `sfunctor/utils/cli.py`, `sfunctor/utils/displacements.py`;
- extraction, production, and plotting: `sfunctor/io/__init__.py`,
  `sfunctor/io/extract.py`,
  `sfunctor/io/bin_convert_new.py`, `sfunctor/io/histogram_results.py`,
  `sfunctor/io/rank_manifest.py`,
  `scripts/production/combine_histograms.py`,
  `scripts/production/combine_histograms_fast.py`,
  `scripts/production/run_node_analysis.py`,
  `job_scripts/production/run_slice_extraction_andes.sh`,
  `job_scripts/production/run_slice_extraction_frontier.sh`,
  `plotting_scripts/plot_structure_functions.py`,
  `plotting_scripts/plot_compare_structure_functions.py`;
- regression updates: `tests/axis_ordering_test.py`,
  `tests/histograms_unified_test.py`, `tests/test_directional.py`,
  `tests/test_histogram_results.py`, `tests/test_plotting_reducers.py`,
  `tests/test_rank_manifest.py`,
  `tests/test_sf_io.py`, `tests/test_sf_physics.py`,
  `test_suite/02_physics_calculations.py`.

## Remaining Ambiguities And Limitations

1. Full-compressible-MHD Elsasser variables do not have one literature-mandated
   density convention. Retain and compare both implemented alternatives.
2. A 2-D slice restricts available separation directions. It is not a substitute
   for unrestricted 3-D conditional sampling. Preserve per-axis occupancy and
   results.
3. The legacy histogram retains its historical undefined-`phi -> 0` fallback
   for schema compatibility. Use the strict path for three-direction science.
4. Current 2-D extraction intentionally rejects non-level-0 layouts. The
   manifest preserves refinement metadata, but the later 3-D chunk iterator
   must define AMR and halo behavior explicitly.
5. Physical-spacing legacy runs require displacement files whose radial edges
   use the same units as `cell_sizes`. The strict CLI exposes the same explicit
   contract.

## Later 3-D Extension

Retain the strict result schema and basis implementation. Add a chunked
pair-batch iterator over rank-local subvolumes with explicit halo ownership,
periodic domain coordinates, Cartesian spacings, and per-rank provenance.
Validate its small-volume output against the existing loop oracle before any
distributed optimization. Do not port the legacy dense-histogram architecture
unchanged into 3-D.
