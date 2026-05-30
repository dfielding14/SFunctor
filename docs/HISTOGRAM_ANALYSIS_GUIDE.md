# Histogram Analysis Guide (Unified `hist` Output)

> **Scope:** this guide describes the legacy binned-histogram approximation.
> For the exact pairwise `l_parallel`, `xi`, and `lambda` calculation used for
> primary three-direction science, use `docs/THREE_DIRECTION_ANALYSIS.md`.

SFunctor stores **binned counts** rather than directly storing structure functions. This is intentional: it preserves the full distribution of increment magnitudes, and allows you to compute many statistics (moments, PDFs, alignment measures) offline without re-running the expensive increment sampling.

This document explains how to go from the unified histogram output to:

- structure functions `S_p(ℓ, θ, φ)` for any order `p`
- 1D “directional” structure functions (parallel vs perpendicular, etc.)
- derived length scales `(ℓ∥, ℓ⊥, ξ, λ)` from bin centers

## 1) What is saved in `*.npz`

Typical combined outputs (and per-node partials) contain:

- `hist`: integer counts with shape `(N_CHANNELS, n_ell, n_theta, n_phi, n_delta)`
- `channels`: list/array of channel names (same order as `hist` axis 0)
- `ell_bin_edges`, `theta_bin_edges`, `phi_bin_edges`: bin edges for `(ℓ, θ, φ)`
- `delta_bin_edges`: per-channel Δ bin edges (saved as an object array)
- `hist_censoring`: optional integer counts with shape
  `(N_CHANNELS, n_ell, n_theta, n_phi, 4)` for accepted, underflow, overflow,
  and invalid values
- `censor_names`: labels for the final `hist_censoring` axis
- `metadata`: dict with run configuration (optional)

The unified format is used by:

- `scripts/production/run_node_analysis.py` (partial node histograms)
- `scripts/production/combine_histograms*.py` (combining to slice-level and all-slice outputs)
- `plotting_scripts/plot_structure_functions.py` (plotting from unified outputs)

## 2) Turning a histogram into a moment `⟨Δ^p⟩`

Fix a channel (e.g. `D_V`) and a bin `(i_ell, i_theta, i_phi)`. Let

- `N_n = hist[channel, i_ell, i_theta, i_phi, n]` be the count in Δ-bin `n`
- `Δ_edges = delta_bin_edges[channel]` be the bin edges

For log-spaced Δ bins, a common representative bin value is the geometric mean:

```
Δ_n = sqrt(Δ_edges[n] * Δ_edges[n+1]).
```

Then the p-th order structure function estimate in that bin is:

```
S_p ≈ (Σ_n N_n Δ_n^p) / (Σ_n N_n).
```

Practical notes:

- Always guard against empty bins (`Σ_n N_n = 0`).
- Inspect `hist_censoring` before interpreting moments. A moment reconstructed
  from a materially censored histogram is biased.
- For ratio channels bounded in `[0, 1]` you may prefer arithmetic bin centers.

### Censoring policy

New production files record each geometrically binned channel value as one of:

- `accepted`: retained in `hist`;
- `underflow`: smaller than the first channel edge;
- `overflow`: larger than the final channel edge;
- `invalid`: non-finite or undefined, such as an alignment ratio with a zero
  denominator.

Use a small calibration run to choose fixed edges before launching distributed
jobs. Do not auto-adjust edges independently by node or snapshot. Treat a
censored fraction above `0.1%` as a warning and justify quantitative use above
`1%` explicitly. Older output files may not contain these counters.

### Minimal Python snippet

```python
import numpy as np
from sfunctor.core.histograms import Channel

data = np.load("sf_results_slice.npz", allow_pickle=True)
hist = data["hist"]
delta_edges = np.asarray(data["delta_bin_edges"][Channel.D_V], dtype=float)

counts = hist[Channel.D_V]  # (n_ell, n_theta, n_phi, n_delta)
delta_centers = np.sqrt(delta_edges[:-1] * delta_edges[1:])  # (n_delta,)

p = 2
numer = (counts * (delta_centers**p)[None, None, None, :]).sum(axis=-1)
denom = counts.sum(axis=-1)
S2 = np.where(denom > 0, numer / denom, np.nan)  # (n_ell, n_theta, n_phi)
```

## 3) Reducing `(ℓ, θ, φ)` bins to 1D curves

Most anisotropy analyses start by collapsing a subset of angular bins to form 1D “directional” structure functions.

### Example: parallel vs perpendicular

Let `θ` be binned on `[0, π/2]`. Define:

- **parallel bins**: `θ` near 0 (e.g. the first one or two `θ` bins)
- **perpendicular bins**: `θ` near `π/2` (e.g. the last one or two `θ` bins)

Then you can form a parallel structure function curve by averaging (or summing counts and re-taking the moment) over those bins:

```
S_p^∥(ℓ_i) = moment_over_Δ( Σ_{θ∈∥, φ∈all} hist(ℓ_i, θ, φ, Δ) )
S_p^⊥(ℓ_i) = moment_over_Δ( Σ_{θ∈⊥, φ∈all} hist(ℓ_i, θ, φ, Δ) )
```

Summing counts first and then computing the moment is preferred (it correctly weights by sample counts).

### Example: in-plane directions (ξ vs λ)

For in-plane anisotropy you typically focus on `θ ≈ π/2` (perpendicular separations) and split by `φ`:

- `φ ≈ 0` corresponds to displacements aligned with `δB_⊥` → **ξ direction**
- `φ ≈ π/2` corresponds to displacements perpendicular to `δB_⊥` → **λ direction**

So:

```
S_p^ξ(ℓ_i) = moment_over_Δ( Σ_{θ∈⊥, φ∈ξ} hist(ℓ_i, θ, φ, Δ) )
S_p^λ(ℓ_i) = moment_over_Δ( Σ_{θ∈⊥, φ∈λ} hist(ℓ_i, θ, φ, Δ) )
```

## 4) Converting bin centers to `(ℓ∥, ℓ⊥, ξ, λ)`

The histogram bins are stored in `(ℓ, θ, φ)`. If you want to plot against derived lengths, you can map bin centers:

- `ℓ_center = 0.5(ℓ_edges[i] + ℓ_edges[i+1])` (often fine because ℓ bins are linear in this pipeline; if they were log-spaced you’d prefer geometric centers)
- `θ_center = 0.5(θ_edges[j] + θ_edges[j+1])`
- `φ_center = 0.5(φ_edges[k] + φ_edges[k+1])`

Then:

```
ℓ∥  = ℓ_center cos θ_center
ℓ⊥  = ℓ_center sin θ_center
ξ   = ℓ_center sin θ_center cos φ_center
λ   = ℓ_center sin θ_center sin φ_center
```

This is an approximation (because bins have width), but is often sufficient for scaling analyses.

## 5) Alignment channels (optional but useful for Boldyrev-type tests)

Some channels are explicitly constructed so that their “ratio” value is a sine of an alignment angle:

```
R = |a_⊥ × b_⊥| / (|a_⊥||b_⊥|) = sin(θ_align)
```

These are stored as histogram channels of the ratio `R` itself (binned on Δ in `[0, 1]`).

You can compute a mean alignment angle at each `(ℓ, θ, φ)` bin via:

1) compute `⟨R⟩` from the ratio histogram
2) map to an angle with `θ_align ≈ arcsin(clip(⟨R⟩, 0, 1))`

Depending on your application, you may instead want a median or a percentile of `R`.

## 6) Next: anisotropy extraction

Once you can compute `S_p^∥(ℓ)` and `S_p^⊥(ℓ)`, you can extract an anisotropy scaling relation `ℓ∥(ℓ⊥)` by matching equal values of `S_p`.

That procedure is described in detail in `docs/MEASURING_ANISOTROPY.md`.
