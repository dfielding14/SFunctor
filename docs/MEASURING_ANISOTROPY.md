# Measuring Eddy Anisotropy from Structure Functions

This document explains how to extract anisotropy scalings from anisotropic structure functions, with an emphasis on the common “equal-structure-function” mapping:

> find `ℓ∥(ℓ⊥)` such that `S_p^∥(ℓ∥) = S_p^⊥(ℓ⊥)`

and the analogous in-plane mapping between `(ξ, λ)`.

The goal is to produce directly comparable scalings to theoretical pictures like Goldreich–Sridhar and Boldyrev.

## 1) Why “equal-structure-function” mapping?

In an inertial-range cascade, it is often more meaningful to compare **eddies of the same fluctuation strength** rather than eddies at the same geometric separation.

If `S_p(ℓ)` is monotone in `ℓ` over the range of interest, then fixing a value of the structure function (or of a moment) selects a family of eddies of comparable amplitude. Matching those values between “parallel” and “perpendicular” directions provides an empirical anisotropy relation.

This avoids having to pick a single “eddy size” definition; instead you compare scales at equal `S_p`.

## 2) Step A: build directional structure functions

Starting from histogram output, construct 1D curves for each direction you care about.

### 2.1 Parallel vs perpendicular directions

Define (using θ bins):

- `S_p^∥(ℓ)` using θ bins near `0`
- `S_p^⊥(ℓ)` using θ bins near `π/2`

In practice:

1) pick a small set of θ bins at the low/high ends
2) sum histogram counts over φ (and over the chosen θ bins)
3) compute the moment over Δ

This yields two functions sampled at the same `ℓ` bin centers:

```
S_p^∥(ℓ_i),   S_p^⊥(ℓ_i).
```

### 2.2 In-plane directions: ξ vs λ

For in-plane anisotropy, focus on `θ ≈ π/2` and define:

- `S_p^ξ(ℓ)` using φ bins near `0`
- `S_p^λ(ℓ)` using φ bins near `π/2`

These measure statistics of increments along and across the local perpendicular fluctuation direction.

## 3) Step B: map `ℓ∥(ℓ⊥)` by matching `S_p`

Assume you have two monotone curves:

```
S_p^∥(ℓ)  and  S_p^⊥(ℓ).
```

The equal-structure-function mapping proceeds as follows:

1) choose a set of target values `S_target` spanning the overlap of the two curves
2) invert each curve to get the corresponding scale at that `S_target`:

```
ℓ∥(S_target)  from  S_p^∥(ℓ)
ℓ⊥(S_target)  from  S_p^⊥(ℓ)
```

3) eliminate `S_target` to get the anisotropy relation:

```
ℓ∥(ℓ⊥).
```

### 3.1 Practical inversion (interpolation)

Because you have discrete samples at `ℓ_i`, you usually:

- restrict to a range where the curves are smooth and monotone (often the inertial range)
- interpolate in log-log space to reduce curvature:

```
log ℓ  ↔  log S_p
```

Then inversion is just interpolation `log ℓ(log S)` on that grid.

### 3.2 Extracting a scaling exponent

If `ℓ∥ ∝ ℓ⊥^α`, then:

```
log ℓ∥ = α log ℓ⊥ + const.
```

So you fit a line in log-log space over the inertial range of the mapping.

## 4) Step C: in-plane anisotropy via `ξ(λ)`

Repeat the same idea using the in-plane directional curves:

```
S_p^ξ(ℓ)  and  S_p^λ(ℓ)
```

to obtain a mapping:

```
ξ(λ)  such that  S_p^ξ(ξ) = S_p^λ(λ).
```

In Boldyrev-type sheet-like turbulence, this in-plane anisotropy is a key additional diagnostic beyond `ℓ∥(ℓ⊥)`.

## 5) Which p?

Common choices:

- `p = 2`: connects naturally to energy spectra and variance-like measures
- `p = 1`: can be more robust to intermittency/heavy tails
- higher `p`: probes intermittency, but requires more statistics

For anisotropy scalings (GS/Boldyrev), `p=2` is the typical starting point.

## 6) Caveats and best practices

### 6.1 Local vs global mean field

Anisotropy in strong MHD turbulence is best captured relative to a **local** mean field at the scale of interest. SFunctor uses a local `B_loc` built from nearby stencil points.

If you instead used a global mean `B0`, you would generally underestimate anisotropy at small scales.

### 6.2 θ/φ binning choices affect “directional” curves

Your “parallel” curve depends on how narrow the θ window is near 0, and your “perpendicular” curve depends on how narrow the window is near `π/2`. Narrower bins are cleaner but noisier.

A good practice is to:

- try multiple θ/φ window widths
- verify the inferred exponent is stable within uncertainties

### 6.3 2D-slice limitation

SFunctor analyzes 2D slices, so displacements lie in the slice plane. This can bias the sampling of `θ` and `ℓ∥` if the magnetic field is strongly oriented out of that plane.

If you compare different slice orientations (axis=1/2/3) and get consistent anisotropy scalings, that increases confidence.

## 7) Minimal algorithm sketch (pseudo-code)

1) Load `hist`, `delta_bin_edges`, `ell_bin_edges`, `theta_bin_edges`, `phi_bin_edges`.
2) Choose a channel and an order `p`.
3) Define index sets:
   - `Θ∥`: θ bins near 0
   - `Θ⊥`: θ bins near π/2
   - optionally, `Φξ` near 0 and `Φλ` near π/2
4) Compute directional curves by summing counts and taking moments.
5) Invert curves via interpolation to get `ℓ∥(S)` and `ℓ⊥(S)`.
6) Fit `log ℓ∥` vs `log ℓ⊥` to extract α.

For concrete histogram-to-moment code, see `docs/HISTOGRAM_ANALYSIS_GUIDE.md`.

