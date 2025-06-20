import cmasher as cmr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from numba import jit, njit

matplotlib.rc("font", family="sans-serif", size=12)
matplotlib.rcParams["xtick.direction"] = "in"
matplotlib.rcParams["ytick.direction"] = "in"
matplotlib.rcParams["xtick.top"] = True
matplotlib.rcParams["ytick.right"] = True
matplotlib.rcParams["xtick.minor.visible"] = True
matplotlib.rcParams["ytick.minor.visible"] = True
matplotlib.rcParams["lines.dash_capstyle"] = "round"
matplotlib.rcParams["figure.dpi"] = 200
plt.clf()


import multiprocess as mulitp

PoolProcesses = mulitp.cpu_count() - 4
print(PoolProcesses)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--stride", type=int, default=1)
parser.add_argument("--file_name", type=str, default="")
parser.add_argument("--n_disp_total", type=int, default=1000)
parser.add_argument("--N_random_subsamples", type=int, default=1000)
parser.add_argument("--n_ell_bins", type=int, default=128)
args = parser.parse_args()
stride = args.stride
file_name = args.file_name
n_disp_total = args.n_disp_total
N_random_subsamples = args.N_random_subsamples
n_ell_bins = args.n_ell_bins


from sf_histograms import Channel, compute_histogram_for_disp_2D

# Local helper module for I/O --------------------------------------------------
from sf_io import load_slice_npz, parse_slice_metadata

# Local helper modules
from sf_physics import compute_vA, compute_z_plus_minus

# -----------------------------------------------------------------------------
# 1. Data loading & metadata ---------------------------------------------------
# -----------------------------------------------------------------------------

# Parse axis & plasma beta from filename
axis, beta = parse_slice_metadata(file_name)

# Load slice data with consistent stride for all fields
slice_data = load_slice_npz(file_name, stride=stride)

# Alias commonly-used variables (legacy names kept for now) --------------------
rho = slice_data["rho"]

B_x = slice_data["B_x"]
B_y = slice_data["B_y"]
B_z = slice_data["B_z"]

v_x = slice_data["v_x"]
v_y = slice_data["v_y"]
v_z = slice_data["v_z"]

# Optional: vorticity & current, available but not yet used in this script.
omega_x = slice_data["omega_x"]
omega_y = slice_data["omega_y"]
omega_z = slice_data["omega_z"]

j_x = slice_data["j_x"]
j_y = slice_data["j_y"]
j_z = slice_data["j_z"]

# Clean-up to free the namespace ------------------------------------------------
del slice_data

# -----------------------------------------------------------------------------
# 2. Derived fields ------------------------------------------------------------
# -----------------------------------------------------------------------------

vA_x, vA_y, vA_z = compute_vA(B_x, B_y, B_z, rho)
(z_plus_x, z_plus_y, z_plus_z), (z_minus_x, z_minus_y, z_minus_z) = (
    compute_z_plus_minus(v_x, v_y, v_z, vA_x, vA_y, vA_z)
)


def find_ell_bin_edges(r_min, r_max, n_ell_bins):
    # Binary search to find the right number of points
    n_points_low = n_ell_bins + 1  # Minimum possible number of points
    n_points_high = 3 * n_ell_bins  # Start with a reasonable upper bound

    while n_points_low <= n_points_high:
        n_points_mid = (n_points_low + n_points_high) // 2
        ell_bin_edges = np.unique(
            np.around(np.geomspace(r_min, r_max, n_points_mid)).astype(int)
        )

        if len(ell_bin_edges) < n_ell_bins + 1:
            n_points_low = n_points_mid + 1
        elif len(ell_bin_edges) > n_ell_bins + 1:
            n_points_high = n_points_mid - 1
        else:
            # Found exactly the right number of points
            break

    # If we exited the loop without finding exactly n_ell_bins + 1 points,
    # take the closest result (should be rare)
    if len(ell_bin_edges) != n_ell_bins + 1:
        print(
            f"Warning: Could not find exactly {n_ell_bins + 1} unique bin edges. Using {len(ell_bin_edges)} instead."
        )
    return ell_bin_edges


N_res = rho.shape[0]  # size of the grid (e.g., 10240 later)
r_min = 1.0
r_max = N_res // 4
ell_bin_edges = find_ell_bin_edges(r_min, r_max, n_ell_bins)
ell_bin_edges = np.array(ell_bin_edges, dtype=np.float64)


# Calculate how many displacements per ell-bin
n_per_bin = n_disp_total // n_ell_bins

disp_list = []

for i in range(n_ell_bins):
    # Get the lower and upper edge for the current bin
    r_low = ell_bin_edges[i]
    r_high = ell_bin_edges[i + 1]

    # Generate n_per_bin radii logarithmically spaced between the bin edges.
    r_values = np.geomspace(r_low, r_high, n_per_bin)

    # Generate independent random angles for this bin in [0, π)
    angles = np.random.uniform(0, np.pi, n_per_bin)

    # Compute the displacements and round to nearest integer
    dx = np.rint(r_values * np.cos(angles)).astype(np.int32)
    dy = np.rint(r_values * np.sin(angles)).astype(np.int32)

    # Stack them into a (n_per_bin, 2) array for this bin.
    disp_bin = np.column_stack((dx, dy))

    # Optionally: if you want to enforce uniqueness per bin, you might remove duplicates here.
    # However, note that removing duplicates might leave you with fewer than n_per_bin entries.
    disp_bin = np.unique(disp_bin, axis=0)

    disp_list.append(disp_bin)

# Combine displacements from all bins.
displacements = np.vstack(disp_list)

# -------------------------------
# Step 2: Remove mirror duplicates (e.g. [1,0] and [-1,0])
# -------------------------------
# Define a canonical representation:
# If the first element is negative, or if it is zero and the second element is negative, use the negated vector.
mask = (displacements[:, 0] < 0) | (
    (displacements[:, 0] == 0) & (displacements[:, 1] < 0)
)
canonical = np.where(mask[:, None], -displacements, displacements)

# Remove duplicates based on canonical representation.
unique_canonical = np.unique(canonical, axis=0)

displacements = unique_canonical[
    np.argsort(np.sqrt(unique_canonical[:, 0] ** 2 + unique_canonical[:, 1] ** 2))
]
n_disp = displacements.shape[0]

# -------------------------------
# Step 3: Define histogram bin edges
# -------------------------------
ell_bin_centers = (ell_bin_edges[:-1] + ell_bin_edges[1:]) / 2.0

# θ bins: here we choose 18 bins covering 0 to π.
n_theta_bins = 18
theta_bin_edges = np.linspace(0, np.pi / 2, n_theta_bins + 1)
theta_bin_centers = (theta_bin_edges[:-1] + theta_bin_edges[1:]) / 2.0

# ϕ bins: here we choose 18 bins covering 0 to π.
n_phi_bins = 18
phi_bin_edges = np.linspace(0, np.pi, n_phi_bins + 1)
phi_bin_centers = (phi_bin_edges[:-1] + phi_bin_edges[1:]) / 2.0

# sf bins: choose a range for the SF values. bins will be raised to the power of p_orders
sf_min = 1e-6
sf_max = 1e1
n_sf_bins = 500
sf_bin_edges = np.logspace(np.log10(sf_min), np.log10(sf_max), n_sf_bins + 1)
sf_bin_centers = (sf_bin_edges[:-1] + sf_bin_edges[1:]) / 2.0

# product bins: choose a range for the SF values. bins will be raised to the power of p_orders
product_min = 1e-10
product_max = 1e1
n_product_bins = 500
product_bin_edges = np.logspace(
    np.log10(product_min), np.log10(product_max), n_product_bins + 1
)
product_bin_centers = (product_bin_edges[:-1] + product_bin_edges[1:]) / 2.0


# -----------------------------------------------------------------------------
# Histogram kernel imported from sf_histograms --------------------------------
# -----------------------------------------------------------------------------


# -------------------------------
# Step 4: Loop over all displacements with multiprocess and sum the histograms.
# -------------------------------
@njit
def process_displacement_3(i_disp):
    dx, dy = displacements[i_disp]
    return compute_histogram_for_disp_2D(
        v_x,
        v_y,
        v_z,
        B_x,
        B_y,
        B_z,
        rho,
        j_x,
        j_y,
        j_z,
        dx,
        dy,
        axis,
        N_random_subsamples,
        ell_bin_edges,
        theta_bin_edges,
        phi_bin_edges,
        sf_bin_edges,
        product_bin_edges,
        3,
    )


# For testing, try one displacement:
hist_single = process_displacement_3(0)
print("Histogram shape for a single displacement:", hist_single.shape)


def process_batch_3(batch_indices):
    result = process_displacement_3(batch_indices[0])
    for i in batch_indices[1:]:
        result += process_displacement_3(i)
    return result


# Create batches (fewer, larger batches)
batch_size = max(1, n_disp // PoolProcesses)  # Aim for ~4 batches per process
batches = [range(i, min(i + batch_size, n_disp)) for i in range(0, n_disp, batch_size)]
print(batch_size, batches[0], batches[-1])

pool = mulitp.Pool(processes=PoolProcesses)
batch_results = pool.map(process_batch_3, batches)
pool.close()
pool.join()

# Flatten results
hist_3 = np.sum(np.array(batch_results), axis=0)
np.savez(
    f'slice_data/hist_3_{file_name[:-4].split("/")[-1]}_ndisps_{n_disp_total}_Nsubsamples_{N_random_subsamples}.npz',
    hist_3=hist_3,
    ell_bin_edges=ell_bin_edges,
    theta_bin_edges=theta_bin_edges,
    phi_bin_edges=phi_bin_edges,
    sf_bin_edges=sf_bin_edges,
    product_bin_edges=product_bin_edges,
    # save the centers too
    ell_bin_centers=ell_bin_centers,
    theta_bin_centers=theta_bin_centers,
    phi_bin_centers=phi_bin_centers,
    sf_bin_centers=sf_bin_centers,
    product_bin_centers=product_bin_centers,
)


hist_3 = hist_3[:, :, :, :, :-1]
sf_bin_centers = sf_bin_centers[:-1]
product_bin_centers = product_bin_centers[:-1]

# some test plots
bsf_theta_ell = (
    np.sum(np.sum(hist_3[1], axis=(2)) * sf_bin_centers**2, axis=-1)
    / np.sum(hist_3[1], axis=(2, 3))
) ** (1 / 2)
vsf_theta_ell = (
    np.sum(np.sum(hist_3[0], axis=(2)) * sf_bin_centers**2, axis=-1)
    / np.sum(hist_3[0], axis=(2, 3))
) ** (1 / 2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 4), sharey=True, sharex=True)
plot = ax1.contourf(
    theta_bin_centers * 180 / np.pi,
    ell_bin_centers,
    bsf_theta_ell,
    levels=np.linspace(0, 2, 21),
)
cb = fig.colorbar(plot, ax=ax1, location="top", shrink=0.8)
cb.set_label(r"$\langle|\delta B (\ell)|^2 : \theta \rangle^{1/2}$")
ax1.semilogy()

plot = ax2.contourf(
    theta_bin_centers * 180 / np.pi,
    ell_bin_centers,
    vsf_theta_ell,
    levels=np.linspace(0, 2, 21),
)
cb = fig.colorbar(plot, ax=ax2, location="top", shrink=0.8)
cb.set_label(r"$\langle|\delta v (\ell)|^2 : \theta \rangle^{1/2}$")
ax2.semilogy()

ax1.set_xlabel(r"$\theta$")
ax2.set_xlabel(r"$\theta$")
ax1.set_ylabel(r"$\ell$")

fig.tight_layout()

plt.savefig(
    f'slice_data/SF_theta_ell_{file_name[:-4].split("/")[-1]}_ndisps_{n_disp_total}_Nsubsamples_{N_random_subsamples}.png',
    dpi=200,
)
plt.clf()


p_orders = np.arange(1, 11, 1)
vsfp = np.array(
    [
        (
            np.sum(np.sum(hist_3[1], axis=(1, 2)) * sf_bin_centers**p, axis=1)
            / np.sum(hist_3[1], axis=(1, 2, 3))
        )
        ** (1 / p)
        for p in p_orders
    ]
)

plot = plt.pcolormesh(
    ell_bin_centers,
    sf_bin_centers,
    np.sum(hist_3[1], axis=(1, 2)).T / np.sum(np.sum(hist_3[1], axis=(1, 2)).T, axis=0),
    norm=LogNorm(),
    cmap=cmr.arctic_r,
)
cb = plt.colorbar(plot)
cb.set_label(r"$N(\delta v|\ell)$")
plt.grid()
plt.axhline(1, color="magenta", linestyle="--", lw=1)
for i, line in enumerate(vsfp):
    plt.loglog(
        ell_bin_centers, line, color="0.7", linewidth=1.5 - i * 0.1, alpha=1 - i * 0.1
    )
plt.ylabel(r"$|\delta v|$")
plt.xlabel(r"$\ell$")
plt.ylim(bottom=1e-3)
plt.savefig(
    f'slice_data/SF_v_ell_{file_name[:-4].split("/")[-1]}_ndisps_{n_disp_total}_Nsubsamples_{N_random_subsamples}.png',
    dpi=200,
)
plt.clf()


B0 = np.sqrt(2 / beta)

p_orders = np.arange(1, 11, 1)
vsfp = np.array(
    [
        (
            np.sum(np.sum(hist_3[0], axis=(1, 2)) * sf_bin_centers**p, axis=1)
            / np.sum(hist_3[0], axis=(1, 2, 3))
        )
        ** (1 / p)
        for p in p_orders
    ]
)
plot = plt.pcolormesh(
    ell_bin_centers,
    sf_bin_centers,
    np.sum(hist_3[0], axis=(1, 2)).T / np.sum(np.sum(hist_3[0], axis=(1, 2)).T, axis=0),
    norm=LogNorm(),
    cmap=cmr.amethyst_r,
)
cb = plt.colorbar(plot)
cb.set_label(r"$N(\delta B|\ell)$")
plt.grid()
plt.axhline(B0, color="cyan", linestyle="--", lw=1)
for i, line in enumerate(vsfp):
    plt.loglog(
        ell_bin_centers, line, color="0.7", linewidth=1.5 - i * 0.1, alpha=1 - i * 0.1
    )
plt.ylabel(r"$|\delta B|$")
plt.xlabel(r"$\ell$")
plt.ylim(bottom=1e-3)

plt.savefig(
    f'slice_data/SF_b_ell_{file_name[:-4].split("/")[-1]}_ndisps_{n_disp_total}_Nsubsamples_{N_random_subsamples}.png',
    dpi=200,
)
plt.clf()


average_vperp_cross_bperp = np.sum(
    np.sum(hist_3[2], axis=(1, 2)) * product_bin_centers, axis=1
) / np.sum(hist_3[2], axis=(1, 2, 3))
average_vperp_bperp = np.sum(
    np.sum(hist_3[3], axis=(1, 2)) * product_bin_centers, axis=1
) / np.sum(hist_3[3], axis=(1, 2, 3))

# plt.loglog(ell_bin_centers,average_vperp_cross_bperp, label=r"$\langle|\delta v_\perp \times \delta B_\perp|\rangle$")
# plt.loglog(ell_bin_centers,average_vperp_bperp, label=r"$\langle |\delta v_\perp| |\delta B_\perp| \rangle$")
plt.loglog(
    ell_bin_centers,
    average_vperp_cross_bperp / average_vperp_bperp,
    label=r"$\frac{\langle|\delta v_\perp \times \delta B_\perp|\rangle}{\langle |\delta v_\perp| |\delta B_\perp| \rangle}$",
)
plt.loglog(
    ell_bin_centers,
    0.1 * ell_bin_centers ** (1 / 4),
    color="black",
    label=r"$\ell^{1/4}$",
)
plt.loglog(
    ell_bin_centers,
    0.2 * ell_bin_centers ** (1 / 8),
    color="grey",
    label=r"$\ell^{1/8}$",
)
plt.loglog(
    ell_bin_centers,
    0.2 * ell_bin_centers ** (1 / 16),
    color="0.25",
    label=r"$\ell^{1/16}$",
)
plt.legend()
plt.savefig(
    f'slice_data/theta_vb_ell_{file_name[:-4].split("/")[-1]}_ndisps_{n_disp_total}_Nsubsamples_{N_random_subsamples}.png',
    dpi=200,
)
plt.clf()
