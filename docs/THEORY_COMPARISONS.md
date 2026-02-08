# Theory Comparisons: Goldreich–Sridhar vs Boldyrev-Type Pictures

This document summarizes common theoretical expectations for anisotropy in strong, magnetized turbulence and how to compare them to measurements from anisotropic structure functions.

It is intentionally “comparison oriented”: it focuses on scaling relations you can directly test with `S_p`-based anisotropy mappings.

## 1) Coordinate system and measured scalings

Using the eddy-aligned scales defined in `docs/ANISOTROPIC_STRUCTURE_FUNCTIONS.md`:

- `ℓ∥`: separation along the local mean magnetic field
- `ℓ⊥`: separation perpendicular to the local mean magnetic field
- `ξ`: separation in the perpendicular plane along the local fluctuation direction (often taken to be `δB_⊥`)
- `λ`: separation in the perpendicular plane perpendicular to that fluctuation direction

The core anisotropy diagnostics are power-law relations (in an inertial range):

```
ℓ∥ ∝ ℓ⊥^α
ξ  ∝ λ^β
```

and, for alignment-based theories, a scale-dependent alignment angle:

```
θ_align(ℓ⊥) ∝ ℓ⊥^γ.
```

## 2) Goldreich–Sridhar (GS95) strong turbulence

### 2.1 Critical balance (concept)

GS95 posits “critical balance”: the linear Alfvén time along the magnetic field and the nonlinear interaction time across the field are comparable at each scale.

In wave-number form, this yields:

```
k∥ v_A  ~  k⊥ δu⊥,
```

with `δu⊥` the characteristic fluctuation amplitude at perpendicular scale `~ 1/k⊥`.

### 2.2 Canonical scaling expectations

Under standard assumptions (Kolmogorov-like cascade in the perpendicular direction), GS95 gives:

- perpendicular spectrum: `E(k⊥) ∝ k⊥^{-5/3}`
- anisotropy: `k∥ ∝ k⊥^{2/3}`  ↔  `ℓ∥ ∝ ℓ⊥^{2/3}`

In the **perpendicular plane**, GS95 does not require a strong anisotropy between two perpendicular directions; a common working assumption is approximate isotropy in that plane:

```
ξ ∼ λ  (β ≈ 1)
```

though in practice coherent structures and intermittency can introduce departures.

### 2.3 What to compare in SFunctor outputs

From structure functions:

1) build `S_p^∥(ℓ)` and `S_p^⊥(ℓ)` (see `docs/MEASURING_ANISOTROPY.md`)
2) perform equal-`S_p` mapping to get `ℓ∥(ℓ⊥)`
3) fit `ℓ∥ ∝ ℓ⊥^α` and compare `α` to `2/3`

## 3) Boldyrev dynamic alignment (sheet-like eddies)

### 3.1 Core idea

Boldyrev-type pictures introduce **scale-dependent alignment** between velocity and magnetic fluctuations (in Alfvén units), which reduces nonlinear interaction strength at small scales. This changes both spectral and anisotropy scalings.

### 3.2 Canonical scaling expectations (one common form)

A widely cited set of Boldyrev scalings is:

- perpendicular spectrum: `E(k⊥) ∝ k⊥^{-3/2}`
- alignment angle: `θ_align(ℓ⊥) ∝ ℓ⊥^{1/4}`
- parallel anisotropy: `ℓ∥ ∝ ℓ⊥^{1/2}` (i.e. `α = 1/2`)
- in-plane anisotropy (sheet-like eddies): `ξ ∝ λ^{3/4}` (i.e. `β = 3/4`)

Different derivations/variants exist in the literature, but these exponents are common reference values for comparison.

### 3.3 What to compare in SFunctor outputs

1) Measure `ℓ∥(ℓ⊥)` via equal-`S_p` mapping and compare `α` to `1/2`.
2) Measure `ξ(λ)` via equal-`S_p` mapping in the perpendicular plane and compare `β` to `3/4`.
3) Use alignment channels to estimate alignment angles. For example, the ratio channel

```
R_vB = |δv_⊥ × δB_⊥| / (|δv_⊥||δB_⊥|)
```

is `sin(θ_{vB})`. You can compute `⟨R_vB⟩(ℓ, θ, φ)` and convert to an angle with `arcsin`.

If `θ_{vB}(ℓ⊥) ∝ ℓ⊥^{1/4}` over a clear range, that supports dynamic alignment behavior.

## 4) Practical comparison strategy

To make comparisons robust:

1) Identify an inertial-range window (avoid forcing and dissipation scales).
2) Check that `S_p^∥(ℓ)` and `S_p^⊥(ℓ)` are monotone there.
3) Use multiple `p` (often start with `p=2`, then test `p=1` or `p=3`).
4) Vary the θ/φ bin windows defining “parallel/perp/ξ/λ” and check stability of exponents.
5) Compare across slice orientations (axis=1/2/3) if possible.

## 5) What agreement/disagreement can mean

If you measure:

- `α` close to `2/3` and `β` close to `1`: broadly GS95-like
- `α` close to `1/2` and `β` close to `3/4` with scale-dependent alignment: Boldyrev-like

But real simulations often show:

- finite-Re effects (short inertial ranges)
- intermittency (p-dependent scalings)
- changes in scaling across regimes (e.g. weak→strong transition)

So it is normal to report:

- best-fit exponents with uncertainty and fit range
- sensitivity tests for bin/window choices

## 6) Next steps

For implementation details of the angle definitions used in SFunctor, see:

- `docs/ANISOTROPIC_STRUCTURE_FUNCTIONS.md`

For how to extract anisotropy mappings from binned histograms, see:

- `docs/MEASURING_ANISOTROPY.md`
- `docs/HISTOGRAM_ANALYSIS_GUIDE.md`

