# Anisotropic Structure Functions in SFunctor

This document defines the anisotropic structure functions used throughout the SFunctor pipeline, including:

- the increment definition `δf(x, r)`
- the two angles `(θ, φ)` used to condition averages
- the derived length scales `(ℓ∥, ℓ⊥, ξ, λ)` used to quantify anisotropy

It also explains how these definitions map to what the code bins in the unified histogram output.

> **Scope:** this document describes the broad legacy unified histogram. Its
> `phi` frame is always defined from `delta B_perp`, and its magnitude channels
> use full-vector increments. For the primary strict two-point Chen/Mallet-style
> calculation with per-variable `delta q_perp`, explicit exclusions, and
> compressible Elsasser alternatives, use `docs/THREE_DIRECTION_ANALYSIS.md`.

## 1) Field increments and structure functions

Let `f(x)` be a scalar field (e.g. density `ρ`) or a vector field (e.g. velocity `v`, magnetic field `B`). For a displacement vector `r`, define the (two-point) increment

```
δf(x, r) = f(x + r) − f(x).
```

For vector fields, SFunctor typically bins **magnitudes** such as `|δv|`, `|δB|`, `|δz^±|`, etc. (See `sfunctor/core/histograms.py` for the exact channel definitions.)

The (p-th order) structure function of a scalar increment magnitude `Δ(x, r)` is

```
S_p(r) = ⟨ Δ(x, r)^p ⟩,
```

where `⟨·⟩` denotes an average over spatial locations `x` (and, in practice, over random subsamples of `x`).

### Anisotropic (conditioned) structure functions

In magnetized turbulence, the statistics depend on the direction of `r` relative to the magnetic field and (often) relative to local fluctuation directions. We therefore consider **conditioned** structure functions:

```
S_p(ℓ, θ, φ) = ⟨ Δ(x, r)^p  |  |r|∈bin(ℓ),  θ(r,B_loc)∈bin(θ),  φ(r,δB,B_loc)∈bin(φ) ⟩.
```

SFunctor does not store `S_p` directly; it stores histograms of `Δ` in each `(ℓ, θ, φ)` bin. See `docs/HISTOGRAM_ANALYSIS_GUIDE.md` for how to compute moments from those histograms.

## 2) Local mean magnetic field `B_loc`

The key concept is the **local** (scale-dependent) mean field used to define “parallel” and “perpendicular”.

For a given location `x` and displacement `r`, define `B_loc(x, r)` as a simple average of `B` near the points involved in the increment. In the current code, this depends on the finite-difference stencil width:

- **2-point stencil**: `B_loc = (B(x) + B(x+r))/2`
- **3-point stencil** (along ±r): `B_loc = (B(x−r) + B(x) + B(x+r))/3`
- **5-point stencil** (along ±r, with standard 5-point weights): a weighted average of `B(x±2r), B(x±r), B(x)`

We then form the unit vector

```
b̂ = B_loc / |B_loc|.
```

When `|B_loc|` is extremely small, the code skips the sample (because “parallel/perpendicular” becomes ill-defined).

## 3) The displacement magnitude `ℓ` and the angle `θ`

Let

```
ℓ = |r|.
```

Define the first angle `θ` as the angle between `r` and the local mean field direction `b̂`:

```
cos θ = | r̂ · b̂ |,
θ = arccos(| r̂ · b̂ |),
```

where `r̂ = r/|r|`. The absolute value folds `θ` into `[0, π/2]` (parallel and anti-parallel are treated the same).

Interpretation:

- `θ ≈ 0` means `r` is (anti-)parallel to `B_loc` (a “parallel” separation)
- `θ ≈ π/2` means `r` is perpendicular to `B_loc` (a “perpendicular” separation)

## 4) The perpendicular plane and the angle `φ`

Define the projection operator into the plane perpendicular to `b̂`:

```
a_⊥ = a − (a · b̂) b̂.
```

SFunctor defines the second angle `φ` inside this perpendicular plane using the direction of the **perpendicular magnetic increment** `δB_⊥`. Concretely:

1) Form

```
r_⊥   = r − (r · b̂) b̂
δB_⊥  = δB − (δB · b̂) b̂
```

2) Then define

```
cos φ = | r̂_⊥ · δB̂_⊥ |,
φ = arccos(| r̂_⊥ · δB̂_⊥ |),
```

where `r̂_⊥ = r_⊥/|r_⊥|` and `δB̂_⊥ = δB_⊥/|δB_⊥|`. Again, the absolute value folds `φ` into `[0, π/2]`.

Interpretation:

- `φ ≈ 0`: `r_⊥` is aligned with `δB_⊥` in the perpendicular plane
- `φ ≈ π/2`: `r_⊥` is perpendicular to `δB_⊥` in that plane

If either `|r_⊥|` or `|δB_⊥|` is zero, the legacy histogram sets `φ = 0`.
This historical fallback can contaminate a `xi` endpoint cut and must not be
used as the rigorous three-direction estimator. The strict pairwise path
records the undefined geometry as an exclusion while preserving theta-only
statistics.

## 5) Derived length scales: `(ℓ∥, ℓ⊥, ξ, λ)`

Once `(ℓ, θ, φ)` are defined, it is natural to work with **derived length scales** corresponding to an eddy-oriented coordinate system.

### Parallel and perpendicular lengths: `ℓ∥` and `ℓ⊥`

Define

```
ℓ∥  = ℓ cos θ = |r · b̂|
ℓ⊥  = ℓ sin θ = |r_⊥|.
```

So `ℓ` is decomposed into a part along the local mean field and a part perpendicular to it.

### Splitting the perpendicular length into `(ξ, λ)`

Within the perpendicular plane, we define two orthogonal directions:

- `ê_ξ` along the perpendicular increment direction `δB̂_⊥`
- `ê_λ` perpendicular to both `b̂` and `ê_ξ` (i.e. `ê_λ ∝ b̂ × ê_ξ`)

Then the perpendicular displacement can be decomposed as

```
ξ = | r_⊥ · ê_ξ | = ℓ⊥ cos φ
λ = | r_⊥ · ê_λ | = ℓ⊥ sin φ.
```

These are non-negative by construction (because `θ` and `φ` are folded with absolute values).

Geometric picture (eddy-aligned frame):

- `ℓ∥` measures elongation along the magnetic field
- `ξ` measures the “width” in the perpendicular plane along the local fluctuation direction
- `λ` measures the “thickness” across the fluctuation direction in the perpendicular plane

This 3-scale geometry is the natural setting for Boldyrev-type “sheet-like” eddies.

## 6) How this maps to SFunctor histograms

SFunctor bins counts into `(ℓ, θ, φ, Δ)` for each channel. In the unified output:

- `hist[channel, i_ell, i_theta, i_phi, i_delta]` stores the number of samples whose increment magnitude fell into that `Δ` bin for that `(ℓ, θ, φ)` bin.
- `ell_bin_edges`, `theta_bin_edges`, `phi_bin_edges` define the binning in `(ℓ, θ, φ)`.
- `delta_bin_edges[channel]` defines the `Δ`-bin edges for each channel.

To compute `S_p(ℓ, θ, φ)` from the histogram, you compute a moment over the `Δ` dimension:

```
S_p(ℓ_i, θ_j, φ_k) ≈ ( Σ_n  N_{i,j,k}(Δ_n)  Δ_n^p ) / ( Σ_n N_{i,j,k}(Δ_n) )
```

with `Δ_n` chosen as a representative value in each bin (often the geometric mean for log-spaced bins).

For how to compute moments robustly (and how to collapse bins into “parallel/perp/ξ/λ” curves), see:

- `docs/HISTOGRAM_ANALYSIS_GUIDE.md`
- `docs/MEASURING_ANISOTROPY.md`
