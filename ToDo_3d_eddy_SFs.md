**Agent instructions: anisotropic structure functions S₂(ℓ, θ, ϕ) using existing θ, ϕ binning**

---

### 0. Definitions and geometry (from the simulation code)

You have access to pairwise increments between points, with:

* Separation vector **r** = (dx, dy, dz); magnitude ℓ = r.
* Local mean magnetic field **B̄** = (Bmx, Bmy, Bmz).
* Magnetic increment δB = (dBx, dBy, dBz) and velocity increment δv = (dvx, dvy, dvz) (or other quantities Q).

Angles are computed in the upstream code as:

```python
Bmean_mag = (Bmx * Bmx + Bmy * Bmy + Bmz * Bmz) ** 0.5
cos_theta = abs(dx * Bmx + dy * Bmy + dz * Bmz) / (r * Bmean_mag)
theta_val = arccos(cos_theta)                    # 0 ≤ theta_val ≤ π/2

B_unit = np.array([Bmz, Bmy, Bmx]) / Bmean_mag

dB_vec = np.array([dBz, dBy, dBx])
displacement_vec = np.array([dz, dy, dx])

dB_perp = _perp(dB_vec, B_unit)
displacement_perp = _perp(displacement_vec, B_unit)

cos_phi = abs((displacement_perp * dB_perp).sum()) / (displacement_perp_mag * dB_perp_mag)
phi_val = arccos(cos_phi)                        # 0 ≤ phi_val ≤ π/2
```

Interpretation:

* **θ** (`theta_val`): angle between separation vector **r** and local mean field **B̄**, with parallel and anti-parallel identified:

  * θ ≈ 0 → along ±B̄ (parallel direction **L**).
  * θ ≈ π/2 → in the plane perpendicular to B̄.

* **ϕ** (`phi_val`): angle in the perpendicular plane between:

  * the perpendicular separation displacement_perp, and
  * the perpendicular fluctuation dB_perp.
    With ϕ ∈ [0, π/2]:
  * ϕ ≈ 0 → separation direction aligned with δB⊥.
  * ϕ ≈ π/2 → separation direction perpendicular to δB⊥.

We use this to identify the classic 3D eddy axes:

* **L**: direction along B̄ (small θ).
* **ξ**: direction along δB⊥ in the perpendicular plane (small ϕ, θ ≈ π/2).
* **λ**: direction perpendicular to both B̄ and δB⊥ (ϕ ≈ π/2, θ ≈ π/2).

You already accumulate pair counts in bins of (dQ, ℓ, θ, ϕ) for multiple quantities Q.

---

### 1. Bin definitions (use existing θ, ϕ binning)

Use the given uniform binning:

```python
n_theta_bins = 18
theta_bin_edges = np.linspace(0, np.pi / 2, n_theta_bins + 1)  # shape (19,)
n_phi_bins = 16
phi_bin_edges = np.linspace(0, np.pi / 2, n_phi_bins + 1)      # shape (17,)
```

Define bin centers:

```python
theta_centers = 0.5 * (theta_bin_edges[:-1] + theta_bin_edges[1:])  # (18,)
phi_centers   = 0.5 * (phi_bin_edges[:-1]   + phi_bin_edges[1:])    # (16,)
```

Because the bins are uniform:

* Δθ = (π/2) / 18 ≈ 5°
* Δϕ = (π/2) / 16 ≈ 5.6°

For anisotropy cuts, use **3-bin wedges** (~15–17° half-width) by default.

Define the following index sets:

```python
# θ indices
n_theta_parallel_bins = 3                # ~15° around 0
theta_L_idx    = np.arange(0, n_theta_parallel_bins)          # [0, 1, 2]
n_theta_perp_bins     = 3                # ~15° near π/2
theta_perp_idx = np.arange(n_theta_bins - n_theta_perp_bins, n_theta_bins)  # [15, 16, 17]

# ϕ indices
n_phi_small_bins  = 3                    # ~17° around 0
phi_xi_idx        = np.arange(0, n_phi_small_bins)            # [0, 1, 2]
n_phi_large_bins  = 3                    # ~17° near π/2
phi_lambda_idx    = np.arange(n_phi_bins - n_phi_large_bins, n_phi_bins)    # [13, 14, 15]
```

These integers (3-bin wedges) should be configurable, but treat the above as the default.

---

### 2. Data you operate on

For each quantity Q (e.g. Q ∈ {B components, v components, |B|, |v|, etc.}), you have a 4D histogram over (ℓ, θ, ϕ, dQ):

* `counts_Q[i_ell, j_theta, k_phi, m_dQ]` = number of pairs with:

  * ℓ in ℓ-bin i_ell,
  * θ in θ-bin j_theta,
  * ϕ in φ-bin k_phi,
  * dQ in dQ-bin m_dQ.

You also have ℓ-bin edges `ell_edges` and dQ-bin centers `dQ_centers_Q` (or equivalent).

---

### 3. Compute S₂(ℓ, θ, ϕ) for each Q

For each quantity Q:

1. Compute the total number of pairs per (ℓ, θ, ϕ):

   ```python
   N_pairs_Q[i_ell, j_theta, k_phi] = counts_Q[i_ell, j_theta, k_phi, :].sum(axis=-1)
   ```

2. Compute the second-order structure function for each (ℓ, θ, ϕ):

   ```python
   # dQ_centers_Q is shape (n_dQ_bins,)
   # counts_Q is shape (n_ell_bins, n_theta_bins, n_phi_bins, n_dQ_bins)
   dQ2 = dQ_centers_Q**2  # (n_dQ_bins,)

   numerator   = (counts_Q * dQ2[None, None, None, :]).sum(axis=-1)
   denominator = N_pairs_Q  # as above

   S2_Q = numerator / denominator   # shape (n_ell_bins, n_theta_bins, n_phi_bins)
   # Handle denominator == 0 by setting S2_Q[...] = NaN or masking.
   ```

3. Compute ℓ-bin centers for plotting:

   ```python
   ell_centers = 0.5 * (ell_edges[:-1] + ell_edges[1:])
   ```

---

### 4. Define angular masks for L, ⊥, ξ, λ, and φ-only

Construct boolean masks over (θ, ϕ) index space of shape `(n_theta_bins, n_phi_bins)`:

```python
import numpy as np

L_mask       = np.zeros((n_theta_bins, n_phi_bins), dtype=bool)
perp_mask    = np.zeros((n_theta_bins, n_phi_bins), dtype=bool)
xi_mask      = np.zeros((n_theta_bins, n_phi_bins), dtype=bool)
lambda_mask  = np.zeros((n_theta_bins, n_phi_bins), dtype=bool)
phi_small_mask = np.zeros((n_theta_bins, n_phi_bins), dtype=bool)
phi_90_mask    = np.zeros((n_theta_bins, n_phi_bins), dtype=bool)

# L: θ small, all ϕ
L_mask[theta_L_idx, :] = True

# ⊥: θ near π/2, all ϕ
perp_mask[theta_perp_idx, :] = True

# ξ: θ near π/2 AND ϕ small (separation ‖ δB⊥)
xi_mask[np.ix_(theta_perp_idx, phi_xi_idx)] = True

# λ: θ near π/2 AND ϕ near π/2 (separation ⟂ δB⊥)
lambda_mask[np.ix_(theta_perp_idx, phi_lambda_idx)] = True

# φ-only masks, averaged over all θ:
phi_small_mask[:, phi_xi_idx]     = True    # ϕ small, all θ
phi_90_mask[:,   phi_lambda_idx]  = True    # ϕ ≈ π/2, all θ
```

---

### 5. Angularly averaged structure functions per Q

Define a helper for weighted averages over (θ, ϕ) at fixed ℓ:

```python
def angular_average(S2_slice, N_slice, mask):
    # S2_slice, N_slice: shape (n_theta_bins, n_phi_bins)
    weights = N_slice * mask
    w_sum = weights.sum()
    if w_sum <= 0:
        return np.nan
    return (S2_slice * weights).sum() / w_sum
```

For each quantity Q and each ℓ-bin index `i_ell`, compute:

* **Isotropic (angle-averaged):**

  ```python
  S2_iso_Q[i_ell] = angular_average(S2_Q[i_ell], N_pairs_Q[i_ell],
                                    np.ones_like(L_mask, dtype=bool))
  ```

* **L (parallel to B̄):**

  ```python
  S2_L_Q[i_ell] = angular_average(S2_Q[i_ell], N_pairs_Q[i_ell], L_mask)
  ```

* **Perpendicular shell (all ϕ):**

  ```python
  S2_perp_Q[i_ell] = angular_average(S2_Q[i_ell], N_pairs_Q[i_ell], perp_mask)
  ```

* **ξ (perpendicular, separation aligned with δB⊥):**

  ```python
  S2_xi_Q[i_ell] = angular_average(S2_Q[i_ell], N_pairs_Q[i_ell], xi_mask)
  ```

* **λ (perpendicular, separation ⟂ δB⊥):**

  ```python
  S2_lambda_Q[i_ell] = angular_average(S2_Q[i_ell], N_pairs_Q[i_ell], lambda_mask)
  ```

* **φ-only cuts, averaged over all θ:**

  ```python
  S2_phi_small_Q[i_ell] = angular_average(S2_Q[i_ell], N_pairs_Q[i_ell], phi_small_mask)
  S2_phi_90_Q[i_ell]    = angular_average(S2_Q[i_ell], N_pairs_Q[i_ell], phi_90_mask)
  ```

Repeat this for every ℓ-bin to obtain 1D arrays:

* `S2_iso_Q(ell_centers)`
* `S2_L_Q(ell_centers)`
* `S2_perp_Q(ell_centers)`
* `S2_xi_Q(ell_centers)`
* `S2_lambda_Q(ell_centers)`
* `S2_phi_small_Q(ell_centers)`
* `S2_phi_90_Q(ell_centers)`

---

### 6. Apply the procedure for **all quantities Q**

You will be given or can infer a list of all Q fields for which (dQ, ℓ, θ, ϕ) histograms exist.

For **each** Q in this list:

1. Load (or receive) `counts_Q` and `dQ_centers_Q`.
2. Compute `N_pairs_Q`, `S2_Q` as in Sections 3–5.
3. Compute all directional structure functions:
   `S2_iso_Q`, `S2_L_Q`, `S2_perp_Q`, `S2_xi_Q`, `S2_lambda_Q`, `S2_phi_small_Q`, `S2_phi_90_Q`.
4. Store these arrays for Q in an output structure (e.g. dictionary keyed by Q) and write them to disk in a structured format (e.g. HDF5/npz) with clear names:

   * `S2_iso_<Q>`, `S2_L_<Q>`, `S2_perp_<Q>`, `S2_xi_<Q>`, `S2_lambda_<Q>`, `S2_phi_small_<Q>`, `S2_phi_90_<Q>`, plus `ell_centers`.

---

### 7. Plotting deliverables for each Q

For every quantity Q, generate at least the following plots (log–log in ℓ):

1. **θ-based anisotropy plot** (L vs ⊥ vs isotropic):

   * x-axis: `ell_centers` (ℓ).
   * y-axis: S₂(ℓ).
   * Curves:

     * `S2_iso_Q` — label: `"angle-avg S₂(<Q>)"`.
     * `S2_L_Q` — label: `"L (‖B, θ small)"`.
     * `S2_perp_Q` — label: `"⊥ (θ≈90°, all ϕ)"`.
   * Use log–log scales.
   * Include legend, axis labels, and a title like `"Anisotropic S₂ for <Q>"`.

2. **Perpendicular-plane ξ–λ plot**:

   * x-axis: `ell_centers`.
   * y-axis: S₂(ℓ).
   * Curves:

     * `S2_perp_Q` as reference.
     * `S2_xi_Q` — label: `"ξ (⊥, ϕ small, separation ‖ δB⊥)"`.
     * `S2_lambda_Q` — label: `"λ (⊥, ϕ≈90°, separation ⟂ δB⊥)"`.
   * Log–log scales, legend, labels, appropriate title.

3. **Optional φ-only diagnostic plot (all θ):**

   * x-axis: `ell_centers`.
   * y-axis: S₂(ℓ).
   * Curves:

     * `S2_phi_small_Q` — label: `"ϕ small (alignment, all θ)"`.
     * `S2_phi_90_Q` — label: `"ϕ≈90° (perp to δB⊥, all θ)"`.
   * Log–log, legend, labels.

For each Q, save the figures to disk with systematic filenames, e.g.:

* `S2_<Q>_theta_anisotropy.png` / `.pdf`
* `S2_<Q>_xi_lambda_anisotropy.png` / `.pdf`
* (φ-only diagnostic plot omitted per latest decision)

---

### 8. Summary of final deliverables

For every quantity Q that has a (dQ, ℓ, θ, ϕ) histogram:

1. **Data products:**

   * 1D arrays as functions of ℓ:

     * `S2_iso_<Q>`, `S2_L_<Q>`, `S2_perp_<Q>`, `S2_xi_<Q>`, `S2_lambda_<Q>`, `S2_phi_small_<Q>`, `S2_phi_90_<Q>`.
   * Shared `ell_centers` array.
   * All saved in a structured file (e.g. HDF5/npz) for downstream analysis.

2. **Plots:**

   * Log–log plot of S₂ vs ℓ comparing isotropic, L, and ⊥ for Q.
     * Drop plot titles; y-label `S_{2}(Q)` using the specific channel label.
     * Legend labels: `isotropic (ℓ)`, `parallel (ℓ_∥)`, `perpendicular (ℓ_⊥)`.
     * Add a second panel showing ℓ_⊥ vs ℓ_∥ by matching S₂_⊥(ℓ_⊥) = S₂_∥(ℓ_∥) with log-space interpolation; fit a power law over 32 < ℓ_∥ < ℓ_max/8 (skip fit if <3 valid points).
   * Log–log plot of S₂ vs ℓ comparing ξ, λ, and parallel for Q.
     * Same y-label rule and no titles.
     * Second panel: ξ vs ℓ_∥ and λ vs ℓ_∥ using log-space interpolation to match S₂ curves; fit power laws over 32 < ℓ_∥ < ℓ_max/8 (skip if <3 points).
   * φ-only diagnostic plot omitted per latest decision.
   * All plots labeled with Q, axes, and curve meaning, and saved to disk with systematic filenames (e.g., `dboverbmeanloc_S2_theta_anisotropy.png`, `dboverbmeanloc_S2_xi_lambda_anisotropy.png`).

3. **Output organization:**

   * Place S₂ npz outputs in a dedicated subdirectory under the plot output directory (e.g. `anisotropic_S2/`).
   * Apply similar subdirectory organization to other plot artifacts for clarity (e.g., split raw vs KDE outputs; drop unused hist2d/hist2d_normalized).
