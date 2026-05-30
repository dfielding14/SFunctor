# Strict Three-Direction Conditional Structure Functions

## Scope

Use `sfunctor/core/directional.py` for the primary `ell_parallel`, `xi`, and
`lambda` analysis. It implements strict two-point increments on CPU from 2-D
slices. The broad unified histogram remains useful for PDFs, alignment
diagnostics, higher-order normalized increment filters, and robustness checks,
but it is not the primary Chen/Mallet estimator.

The 2-D calculation is slice-conditioned. It cannot replace unrestricted 3-D
separation coverage. Keep per-axis outputs and angular occupancies separate
before pooling results.

## Pairwise Basis

For each periodic pair `x` and `x+r`:

```text
B_loc      = [B(x) + B(x+r)] / 2
e_parallel = B_loc / |B_loc|
delta q    = q(x+r) - q(x)
delta q_perp = delta q - (delta q . e_parallel) e_parallel
e_xi       = delta q_perp / |delta q_perp|
e_lambda   = e_parallel x e_xi
```

The measured quantity is:

```text
S2^q = <|delta q_perp|^2>.
```

Angles are folded onto `[0, pi/2]` because each eddy axis is unoriented:

```text
theta = acos(|r_hat . e_parallel|)
phi   = acos(|r_perp_hat . e_xi|)
```

Default directional cuts are explicit:

| output | cut |
| --- | --- |
| `parallel` | `theta <= 15 deg` |
| perpendicular | `theta >= 75 deg` |
| `xi` | `theta >= 75 deg` and `phi <= 15 deg` |
| `lambda` | `theta >= 75 deg` and `phi >= 75 deg` |

Vary these widths and inspect counts before interpreting slopes.

Because each 2-D slice only supplies in-plane separations, save and inspect
`coverage_rows` separately for every slice normal. Each row records attempted
pairs, accepted pairs, accepted fraction, exclusions, and whether that
separation bin passes the configured fit-quality gate. A directional slope is
reported only when enough eligible bins remain. Do not silently pool sparsely
sampled directions across slice normals.

## Compressible-MHD Variants

The RMHD literature does not uniquely determine one full-compressible-MHD
Elsasser convention. The calculator therefore labels and retains alternatives:

| `q` | definition | density convention |
| --- | --- | --- |
| `z_plus`, `z_minus` | `u +/- B / sqrt(rho)` | pointwise density |
| `z_plus_ref`, `z_minus_ref` | `u +/- B / sqrt(rho0)` | fixed reference density |
| `B` | magnetic field | none |
| `vA` | `B / sqrt(rho)` | pointwise density |
| `vA_ref` | `B / sqrt(rho0)` | fixed reference density |
| `u` | velocity | none |

By default, `rho0` is the mean over finite positive-density cells in the loaded
slice. Production comparisons should pass an explicit physically chosen
`rho0`, such as the initial density or a documented volume mean.

Every variant uses its own `delta q_perp` both to define `e_xi` and to measure
`|delta q_perp|^2`. `B_loc` always defines `e_parallel` in the primary frame.
The optional global-mean-field frame is a sanity check only.

## Exclusions

Undefined geometry is not assigned to an angular endpoint. The result records:

- invalid or non-finite `B`;
- weak `|B_loc|`;
- invalid `q`, including invalid pointwise density;
- weak `|delta q_perp|`, for which `phi` is undefined;
- weak `|r_perp|`, for which `phi` is undefined;
- displacements outside the requested physical `ell` bins.

Theta-only `parallel` and perpendicular accumulators remain usable when `phi`
is undefined. This is deliberately different from the legacy histogram's
historical `phi=0` fallback.

## Outputs

`scripts/production/run_directional_analysis.py` saves:

- exact `counts`, `sums`, and `sums_sq`;
- `s2` and sampling-noise `standard_error`;
- exclusion counts and attempted-pair counts;
- local/global frames and every requested `q`;
- density conventions, physical spacings, wedge angles, random seed, and cost;
- optional fitted slopes over an explicit interval;
- one occupancy row per `q`, frame, direction, and separation bin, plus
  configurable minimum-count, accepted-fraction, and minimum-bin fit gates;
- optional constant-`S2` eddy scales and `xi/lambda`,
  `ell_parallel/lambda` aspect ratios.

Spatial pairs are correlated. The stored standard error is useful for
subsampling diagnostics but is not a complete physical uncertainty estimate.
For scientific error bars, bootstrap spatial blocks or independent slices and
vary fit windows, wedges, slice axes, and random seeds.

## Literature Mapping

Chen et al. (2012), *Three-dimensional structure of solar wind turbulence*,
use a local magnetic-field basis and the perpendicular magnetic fluctuation
direction. Mallet et al. (2016), *Measures of three-dimensional anisotropy and
intermittency in strong Alfvénic turbulence*, use corresponding perpendicular
Elsasser increments for their directional conditional statistics.

The legacy histogram is a Chen-style magnetic-frame approximation because its
single `phi` is always defined from `delta B_perp`. The strict path supports the
per-variable Mallet-style frame requested here.
