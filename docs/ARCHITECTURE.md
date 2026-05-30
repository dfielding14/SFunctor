# Architecture

## Active Data Flow

1. `sfunctor/io/rank_manifest.py` scans rank-local ownership metadata through
   `sfunctor/io/bin_convert_new.py`, validates one snapshot manifest, and loads
   selected blocks grouped by rank file. `sfunctor/io/extract.py` uses that
   manifest to extract KJI-oriented 2-D planes, computes periodic
   central-difference derived fields, validates every saved plane, and writes
   atomic slice caches.
2. `sfunctor/io/slice_io.py` loads extracted NPZ planes, applies an optional
   stride, and renames AthenaK variables to canonical Python names.
3. `sfunctor/core/physics.py` computes legacy `vA = B / sqrt(rho)` and
   `z+/- = u +/- vA` planes.
4. `sfunctor/core/histograms.py` and `sfunctor/core/parallel.py` accumulate the
   broad legacy count histogram. `scripts/production/run_node_analysis.py`
   writes partial products.
5. `sfunctor/io/histogram_results.py` validates schemas and streams exact
   `int64` combinations. Both production combiner CLIs use it.
6. `plotting_scripts/plot_structure_functions.py` reconstructs approximate
   moments from the legacy delta histograms.
7. `sfunctor/core/directional.py` independently computes strict pairwise
   directional `S2` sums. `sfunctor/reference.py` provides a readable loop
   oracle and `sfunctor/analysis/directional.py` fits slopes and shapes.

## Public Interfaces

- Legacy broad histogram:
  `sfunctor.core.parallel.compute_histograms_shared(...)`
- Strict pairwise directional calculation:
  `sfunctor.core.directional.compute_directional_structure_functions(...)`
- Strict reference oracle:
  `sfunctor.reference.compute_directional_structure_functions_reference(...)`
- Slice extraction:
  `sfunctor.io.extract.extract_2d_slice(...)`
- Rank-file manifest and grouped block reads:
  `sfunctor.io.rank_manifest.build_rank_manifest(...)`,
  `sfunctor.io.rank_manifest.read_blocks_grouped(...)`
- Slice loading:
  `sfunctor.io.slice_io.load_slice_npz(...)`

## AthenaK Storage Contract

AthenaK uses `k,j,i`, equivalently `x3,x2,x1` or `z,y,x`, array order.

| slice normal | stored plane | offset mapping |
| --- | --- | --- |
| `axis=1` | `(k=x3, j=x2)` | `(delta_i, delta_j) -> (0, delta_i dx2, delta_j dx3)` |
| `axis=2` | `(k=x3, i=x1)` | `(delta_i, delta_j) -> (delta_i dx1, 0, delta_j dx3)` |
| `axis=3` | `(j=x2, i=x1)` | `(delta_i, delta_j) -> (delta_i dx1, delta_j dx2, 0)` |

This is tested in `tests/axis_ordering_test.py`,
`tests/histograms_unified_test.py`, and `tests/test_directional.py`.

## Supported Diagnostics

The legacy unified histogram contains 26 channels:

- full-vector magnitudes: `|delta u|`, `|delta B|`, `|delta rho|`,
  `|delta vA|`, `|delta z+|`, `|delta z-|`, `|delta omega|`, `|delta J|`,
  `|delta curvature|`, `|delta grad(rho)|`, and `|delta B| / |B_loc|`;
- perpendicular alignment cross numerators, products, and sine ratios for
  `(u,B)`, `(u,omega)`, `(B,J)`, `(omega,J)`, and `(z+,z-)`.

The strict pairwise directional path contains:

- direct `S2 = <|delta q_perp|^2>` for `all`, `parallel`, perpendicular,
  `xi`, and `lambda` cuts;
- local-`B_loc` and global-mean-`B` frames;
- `q` variants `z_plus`, `z_minus`, `z_plus_ref`, `z_minus_ref`, `B`, `vA`,
  `vA_ref`, and `u`;
- counts, sum-of-squares sampling errors, exclusion counts, occupancy rows,
  and explicit fit-quality gates.

## Legacy Increment Filters

Legacy `stencil_width` values are distinct normalized increment filters:

| width | increment |
| --- | --- |
| `2` | `f(x+r) - f(x)` |
| `3` | `[f(x+r) - 2 f(x) + f(x-r)] / sqrt(3)` |
| `5` | `[f(x-2r) - 4 f(x-r) + 6 f(x) - 4 f(x+r) + f(x+2r)] / sqrt(35)` |

The normalization makes uncorrelated second moments comparable. Widths `3`
and `5` are not higher-accuracy approximations to the two-point statistic.
Only width `2` is directly comparable to the pairwise Chen/Mallet definitions.

## 3-D Extension Boundary

The strict module already separates point-pair geometry from accumulation,
and the rank manifest separates block ownership from slice assembly:
the present origin/target enumerator is 2-D-periodic, while basis construction,
field variants, result storage, fitting, and exclusion accounting are not tied
to one slice orientation. A later chunked 3-D implementation should replace
only pair-batch generation and periodic target lookup, retain explicit
Cartesian spacings, and preserve per-rank provenance.
