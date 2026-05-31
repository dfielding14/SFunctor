# Phase 3 finite-domain 3-D sampler

Phase 3 analyzes the four retained Phase 2 `L_sub = 640` cubes without
periodic wrapping. Extracted arrays remain in AthenaK KJI order:

```text
array[k, j, i] == field[x3, x2, x1]
```

Integer displacements are stored in Cartesian index order:

```text
(di, dj, dk)
```

and converted to Cartesian vectors as:

$$
\mathbf{r}
=
(d_i \Delta x_1,\ d_j \Delta x_2,\ d_k \Delta x_3).
$$

## Pair modes

The primary `nested_core` mode derives one common half-open interior origin
box. Every configured signed displacement is valid from every sampled origin
in that box. This gives every angular direction the same spatial support.

The diagnostic `all_valid_pairs` mode derives a separate valid origin box for
each displacement. It uses more of the cube but can mix finite-domain geometry
with real spatial inhomogeneity. It is retained as a robustness comparison,
not as the primary Phase 3 estimator. The bounded Phase 3 smoke uses the same
sample count for `nested_core` and `all_valid_pairs`, records support for every
individual offset, and repeats the representative all-valid calculation for
three deterministic seeds. This separates measured finite-support differences
from avoidable sampling-depth mismatches and exposes seed sensitivity.

Neither mode applies modulo arithmetic. Endpoints are asserted to remain
inside the cube before array indexing, including for negative displacements.

## Measurements

For each vector variant `q`, the output stores two labeled increment products:

```text
total
perpendicular
```

The total-vector product is:

$$
S_p^q(\ell)
=
\left\langle
\left|
\delta \mathbf{q}
\right|^p
\right\rangle.
$$

The perpendicular-vector product is:

$$
S_{p,\perp}^q(\ell)
=
\left\langle
\left|
\delta \mathbf{q}_{\perp}
\right|^p
\right\rangle.
$$

The perpendicular product is the primary Chen/Mallet-style eddy-geometry
estimator. The total-vector product is a labeled comparison. The existing
direction name `all` means all angular orientations; it does not mean a
total-vector increment.

## Conditioning

The primary pair-scale basis is:

$$
\mathbf{B}_{\mathrm{loc,pair}}
=
\frac{
\mathbf{B}(\mathbf{x})
+
\mathbf{B}(\mathbf{x}+\mathbf{r})
}{2},
$$

$$
\mathbf{e}_{\parallel}
=
\frac{
\mathbf{B}_{\mathrm{loc,pair}}
}{
\left|
\mathbf{B}_{\mathrm{loc,pair}}
\right|
},
$$

$$
\delta \mathbf{q}_{\perp}
=
\delta \mathbf{q}
-
(\delta \mathbf{q} \cdot \mathbf{e}_{\parallel})
\mathbf{e}_{\parallel},
$$

$$
\mathbf{e}_{\xi}
=
\frac{
\delta \mathbf{q}_{\perp}
}{
\left|
\delta \mathbf{q}_{\perp}
\right|
},
\qquad
\mathbf{e}_{\lambda}
=
\mathbf{e}_{\parallel}
\times
\mathbf{e}_{\xi}.
$$

Angles use absolute dot products, so the statistical axes are unoriented. The
result is invariant under sign reversal of `e_xi` or `e_lambda`. A
`subvolume_mean` magnetic frame is also saved as a diagnostic comparison:

$$
\mathbf{B}_{\mathrm{mean,sub}}
=
\left\langle
\mathbf{B}
\right\rangle_{\mathrm{sub}}.
$$

## Field variants

The API supports:

```text
B
u
vA
vA_ref
z_plus
z_minus
z_plus_ref
z_minus_ref
```

with:

$$
\mathbf{v}_A
=
\frac{
\mathbf{B}
}{
\sqrt{\rho}
},
\qquad
\mathbf{z}^{\pm}
=
\mathbf{u}
\pm
\mathbf{v}_A.
$$

Reference-density variants use an explicit fixed `rho0` and remain labeled
diagnostics. AthenaK code units absorb the usual `4 pi` factor where
appropriate. The smoke campaign intentionally evaluates only `B`, `u`, and
`p = 2`.

## Pair accounting

Each result records:

- sampled geometry-valid pairs;
- analytic valid-origin population;
- full cube-origin population;
- boundary or nested-core population excluded from support;
- displacement count per separation bin;
- offset-resolved support, including each displacement vector and separation
  bin, so angular support can be inspected rather than inferred from
  separation-bin totals;
- physical exclusions for invalid fields and degenerate directions;
- directional accepted counts.

Sampling standard errors are retained as diagnostics. They are not physical
uncertainty bars because nearby spatial pairs are correlated.

The `profile-controls` driver action records bounded one-factor timing
comparisons on one approved real cube. It varies pair batch size, sampled-pair
count, separation census, number of $p$ values, and `B` versus `u` execution.
The in-memory sampler has no separate I/O chunk control:
`pair_batch_size` is its bounded work-chunk control.

The smoke campaign computes alternate fitted slopes over
$8 \le \ell \le 64$, $16 \le \ell \le 96$, and $8 \le \ell \le 96$ cells.
The resulting spread is a fit-window sensitivity diagnostic. It is not a
physical uncertainty estimate, and unstable smoke fits must not be promoted
to scientific conclusions.

For scalar fields, Phase 3 does not implement a longitudinal/transverse
decomposition. The vector API stores labeled total-vector and
perpendicular-vector increments only. Scalar extensions are explicitly
deferred.

## Eddy dimensions

`sfunctor.analysis.finite_domain.constant_sp_shapes(...)` infers
`ell_parallel`, `xi`, and `lambda` by matching directional curves at fixed
`S_p`, then reports:

```text
ell_parallel / lambda
xi / lambda
```

The reducer rejects sparse bins, absent crossings, and multiple crossings
rather than silently selecting an ambiguous branch. A later scientific
campaign must restrict interpretation to stable intervals and add spatial
block uncertainty estimates where needed.
