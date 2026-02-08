# Worked Example: Extracting `ℓ∥(ℓ⊥)` and `ξ(λ)` from `sf_results_*.npz`

This document walks through a practical, end-to-end example of how to:

1) load SFunctor’s unified histogram output
2) compute a 2nd-order structure function `S2` from histograms
3) build “directional” curves (parallel/perp and ξ/λ)
4) compute equal-`S2` anisotropy mappings
5) fit scaling exponents and compare to GS95/Boldyrev expectations

It is written to be copy-paste runnable (modulo providing your own `sf_results_*.npz` file).

## 0) Assumptions and file input

You have a file like:

```
sf_results_all_slices.npz
```

containing at least: `hist`, `channels`, `ell_bin_edges`, `theta_bin_edges`, `phi_bin_edges`, `delta_bin_edges`.

## 1) Helper: moment from counts

We will compute `S2` by summing counts first and then taking the Δ-moment.

```python
import numpy as np

def moment_from_counts(counts: np.ndarray, delta_edges: np.ndarray, p: int) -> np.ndarray:
    """
    counts: (..., n_delta)
    delta_edges: (n_delta + 1,)
    returns: (...,) with NaN where empty
    """
    delta_centers = np.sqrt(delta_edges[:-1] * delta_edges[1:])  # geometric mean for log bins
    numer = (counts * (delta_centers**p)[None, ...]).sum(axis=-1)
    denom = counts.sum(axis=-1)
    return np.where(denom > 0, numer / denom, np.nan)
```

## 2) Load a result file and pick a channel

```python
import numpy as np
from sfunctor.core.histograms import Channel

path = "sf_results_all_slices.npz"
data = np.load(path, allow_pickle=True)

hist = data["hist"]  # (N_CHANNELS, n_ell, n_theta, n_phi, n_delta)
channels = list(data["channels"])

ell_edges = np.asarray(data["ell_bin_edges"], dtype=float)
theta_edges = np.asarray(data["theta_bin_edges"], dtype=float)
phi_edges = np.asarray(data["phi_bin_edges"], dtype=float)
delta_bin_edges = data["delta_bin_edges"]

# Choose a channel to analyze: |δv| is a common starting point.
ch = Channel.D_V
ch_idx = channels.index(ch.name)
delta_edges = np.asarray(delta_bin_edges[ch_idx], dtype=float)

ell_centers = 0.5 * (ell_edges[:-1] + ell_edges[1:])
theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
```

## 3) Build directional `S2(ℓ)` curves

We will define “parallel” as the lowest θ bins and “perpendicular” as the highest θ bins.

Pick windows (you should test sensitivity by varying these):

```python
p = 2

n_theta = len(theta_centers)
n_phi = len(phi_centers)

# Example windows:
theta_parallel_bins = np.arange(0, 2)              # first 2 θ bins (near 0)
theta_perp_bins = np.arange(n_theta - 2, n_theta)  # last 2 θ bins (near π/2)

# For in-plane anisotropy, we will also define φ windows:
phi_xi_bins = np.arange(0, 2)                      # near 0
phi_lambda_bins = np.arange(n_phi - 2, n_phi)      # near π/2
```

Now compute `S2^∥(ℓ)` and `S2^⊥(ℓ)`:

```python
counts = hist[ch_idx]  # (n_ell, n_theta, n_phi, n_delta)

# Sum counts over φ and the chosen θ bins first, then take the Δ-moment.
counts_par = counts[:, theta_parallel_bins, :, :].sum(axis=(1, 2))  # (n_ell, n_delta)
counts_perp = counts[:, theta_perp_bins, :, :].sum(axis=(1, 2))     # (n_ell, n_delta)

S2_par = moment_from_counts(counts_par, delta_edges, p=p)    # (n_ell,)
S2_perp = moment_from_counts(counts_perp, delta_edges, p=p)  # (n_ell,)
```

### In-plane curves `S2^ξ(ℓ)` and `S2^λ(ℓ)`

Restrict to `θ ≈ π/2` bins and split by `φ`:

```python
counts_xi = counts[:, theta_perp_bins, phi_xi_bins, :].sum(axis=(1, 2))        # (n_ell, n_delta)
counts_lambda = counts[:, theta_perp_bins, phi_lambda_bins, :].sum(axis=(1, 2))# (n_ell, n_delta)

S2_xi = moment_from_counts(counts_xi, delta_edges, p=p)
S2_lambda = moment_from_counts(counts_lambda, delta_edges, p=p)
```

## 4) Equal-`S2` mapping via interpolation

We want `ℓ∥(ℓ⊥)` such that `S2_par(ℓ∥) = S2_perp(ℓ⊥)`.

The simplest approach is:

1) pick a set of target `S2` values in the overlap region
2) invert each curve by interpolation to get `ℓ(S2)`

```python
def invert_monotone(x: np.ndarray, y: np.ndarray, y_targets: np.ndarray) -> np.ndarray:
    """
    Invert y(x) -> x(y) assuming y is monotone over the chosen range.
    Interpolates in log-log space (common for SF scalings).
    """
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]

    # Sort by y for a proper inversion.
    order = np.argsort(y)
    y = y[order]
    x = x[order]

    logx = np.log(x)
    logy = np.log(y)
    logy_t = np.log(y_targets)

    logx_t = np.interp(logy_t, logy, logx, left=np.nan, right=np.nan)
    return np.exp(logx_t)


# Choose overlap range for targets.
mask_par = np.isfinite(S2_par) & (S2_par > 0)
mask_perp = np.isfinite(S2_perp) & (S2_perp > 0)

Smin = max(np.nanmin(S2_par[mask_par]), np.nanmin(S2_perp[mask_perp]))
Smax = min(np.nanmax(S2_par[mask_par]), np.nanmax(S2_perp[mask_perp]))

S_targets = np.geomspace(Smin, Smax, 30)

ell_par_of_S = invert_monotone(ell_centers, S2_par, S_targets)
ell_perp_of_S = invert_monotone(ell_centers, S2_perp, S_targets)
```

Now you have paired points `(ℓ⊥, ℓ∥)`:

```python
ell_perp = ell_perp_of_S
ell_par = ell_par_of_S
mask_map = np.isfinite(ell_perp) & np.isfinite(ell_par) & (ell_perp > 0) & (ell_par > 0)
ell_perp = ell_perp[mask_map]
ell_par = ell_par[mask_map]
```

### In-plane mapping `ξ(λ)`

```python
mask_xi = np.isfinite(S2_xi) & (S2_xi > 0)
mask_lam = np.isfinite(S2_lambda) & (S2_lambda > 0)

Smin2 = max(np.nanmin(S2_xi[mask_xi]), np.nanmin(S2_lambda[mask_lam]))
Smax2 = min(np.nanmax(S2_xi[mask_xi]), np.nanmax(S2_lambda[mask_lam]))
S_targets2 = np.geomspace(Smin2, Smax2, 30)

xi_of_S = invert_monotone(ell_centers, S2_xi, S_targets2)
lam_of_S = invert_monotone(ell_centers, S2_lambda, S_targets2)

mask2 = np.isfinite(xi_of_S) & np.isfinite(lam_of_S) & (xi_of_S > 0) & (lam_of_S > 0)
xi = xi_of_S[mask2]
lam = lam_of_S[mask2]
```

## 5) Fit scaling exponents

Fit a line in log-log space:

```python
def fit_powerlaw(x: np.ndarray, y: np.ndarray):
    lx = np.log10(x)
    ly = np.log10(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    return slope, intercept

alpha, c0 = fit_powerlaw(ell_perp, ell_par)
beta, c1 = fit_powerlaw(lam, xi)

print("ell_parallel ~ ell_perp^alpha, alpha =", alpha)
print("xi ~ lambda^beta, beta =", beta)
```

Interpretation (typical reference values):

- GS95: `α ≈ 2/3`, `β ≈ 1`
- Boldyrev: `α ≈ 1/2`, `β ≈ 3/4`

## 6) Plot quick diagnostics

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].loglog(ell_perp, ell_par, "o", label="mapping")
ax[0].loglog(ell_perp, (10**c0) * ell_perp**alpha, "-", label=f"fit α={alpha:.2f}")
ax[0].set_xlabel(r"$\\ell_\\perp$")
ax[0].set_ylabel(r"$\\ell_\\parallel$")
ax[0].legend()

ax[1].loglog(lam, xi, "o", label="mapping")
ax[1].loglog(lam, (10**c1) * lam**beta, "-", label=f"fit β={beta:.2f}")
ax[1].set_xlabel(r"$\\lambda$")
ax[1].set_ylabel(r"$\\xi$")
ax[1].legend()

fig.tight_layout()
plt.show()
```

## 7) Notes and recommended refinements

This example is intentionally minimal. For production-quality measurements you should:

- choose an inertial-range window explicitly (fit only there)
- test multiple `(θ, φ)` window widths for robustness
- check monotonicity and remove bins dominated by noise/empty counts
- repeat for multiple channels and orders `p`
- compare across slice orientations (axis=1/2/3) if possible

Conceptual background and definitions:

- `docs/ANISOTROPIC_STRUCTURE_FUNCTIONS.md`
- `docs/MEASURING_ANISOTROPY.md`
- `docs/THEORY_COMPARISONS.md`

