import numpy as np
from numba import njit, jit
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cmasher as cmr


import matplotlib
matplotlib.rc('font', family='sans-serif', size=12)
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['ytick.right'] = True
matplotlib.rcParams['xtick.minor.visible'] = True
matplotlib.rcParams['ytick.minor.visible'] = True
matplotlib.rcParams['lines.dash_capstyle'] = 'round'
matplotlib.rcParams['figure.dpi'] = 200



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



data = np.load(file_name)

rho = data['dens'][::stride,::stride]
B_x = data['bcc1'][::stride,::stride]
B_y = data['bcc2'][::stride,::stride]
B_z = data['bcc3'][::stride,::stride]
v_x = data['velx'][::stride,::stride]
v_y = data['vely'][::stride,::stride]
v_z = data['velz'][::stride,::stride]

del data

#get axis from file name
axis = int(file_name.split('_')[2][-1])
beta = int(file_name.split('_')[6][4:])



def find_ell_bin_edges(r_min, r_max, n_ell_bins):
    # Binary search to find the right number of points
    n_points_low = n_ell_bins + 1  # Minimum possible number of points
    n_points_high = 3 * n_ell_bins  # Start with a reasonable upper bound

    while n_points_low <= n_points_high:
        n_points_mid = (n_points_low + n_points_high) // 2
        ell_bin_edges = np.unique(np.around(np.geomspace(r_min, r_max, n_points_mid)).astype(int))

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
        print(f"Warning: Could not find exactly {n_ell_bins + 1} unique bin edges. Using {len(ell_bin_edges)} instead.")
    return ell_bin_edges



N_res = rho.shape[0]   # size of the grid (e.g., 10240 later)
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
    r_high = ell_bin_edges[i+1]

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
mask = (displacements[:, 0] < 0) | ((displacements[:, 0] == 0) & (displacements[:, 1] < 0))
canonical = np.where(mask[:, None], -displacements, displacements)

# Remove duplicates based on canonical representation.
unique_canonical = np.unique(canonical, axis=0)

displacements = unique_canonical[np.argsort(np.sqrt(unique_canonical[:,0]**2 + unique_canonical[:,1]**2))]
n_disp = displacements.shape[0]

# -------------------------------
# Step 3: Define histogram bin edges
# -------------------------------
ell_bin_centers = (ell_bin_edges[:-1] + ell_bin_edges[1:]) / 2.0

# θ bins: here we choose 18 bins covering 0 to π.
n_theta_bins = 18
theta_bin_edges = np.linspace(0, np.pi/2, n_theta_bins + 1)
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
product_bin_edges = np.logspace(np.log10(product_min), np.log10(product_max), n_product_bins + 1)
product_bin_centers = (product_bin_edges[:-1] + product_bin_edges[1:]) / 2.0


@njit
def find_bin_index_binary(value, bin_edges):
    left, right = 0, len(bin_edges) - 2
    while left <= right:
        mid = (left + right) // 2
        if bin_edges[mid] <= value < bin_edges[mid + 1]:
            return mid
        elif value < bin_edges[mid]:
            right = mid - 1
        else:
            left = mid + 1
    return -1

# -------------------------------
# Step 3: Numba-accelerated function for a single displacement.
# This function computes the SF2 for each spatial point for the given displacement,
# calculates the local mean magnetic field, the local theta, and bins the SF2 value
# into a 3D histogram of shape (n_ell_bins, n_theta_bins, n_sf2_bins).
# (For a given displacement, the ℓ bin is fixed.)
# -------------------------------
@njit
def compute_histogram_for_disp_2D(v_x, v_y, v_z, B_x, B_y, B_z, rho,
                                  delta_i, delta_j, slice_axis,
                                  N_random_subsamples,
                                  ell_bin_edges, theta_bin_edges, phi_bin_edges,
                                  sf_bin_edges, product_bin_edges,
                                  stencil_width=2):
    N, M            = v_x.shape
    n_ell_bins      = ell_bin_edges.shape[0] - 1
    n_theta_bins    = theta_bin_edges.shape[0] - 1
    n_phi_bins      = phi_bin_edges.shape[0] - 1
    n_sf_bins       = sf_bin_edges.shape[0] - 1

    if slice_axis == 1:
        dx = 0
        dy = delta_i
        dz = delta_j
    elif slice_axis == 2:
        dx = delta_i
        dy = 0
        dz = delta_j
    elif slice_axis == 3:
        dx = delta_i
        dy = delta_j
        dz = 0

    # 1  hist the vsfp
    # 2  hist the bsfp
    # 3  hist the vperp_cross_bperp
    # 4  hist the vperp_bperp
    n_hist_elements = 4

    # Create an empty histogram with shape: (quantity being averaged, ℓ bins, θ bins, quantity bins, number of p orders)
    hist = np.zeros((n_hist_elements, n_ell_bins, n_theta_bins, n_phi_bins, n_sf_bins), dtype=np.int64)

    # Compute displacement magnitude r.
    r = (delta_i*delta_i + delta_j*delta_j) ** 0.5
    # Determine ℓ bin index (for this displacement) using the provided edges.
    ell_idx = -1
    for i in range(n_ell_bins):
        if ell_bin_edges[i] <= r < ell_bin_edges[i+1]:
            ell_idx = i
            break
    if ell_idx == -1:
        return hist

    # Sample unique flat indices from the grid
    flat_indices = np.random.choice(M*N, size=N_random_subsamples, replace=False)

    # Convert flat indices to 2D coordinates
    random_points_y = flat_indices // N  # row indices
    random_points_x = flat_indices % N   # column indices

    # Loop over every spatial position (with periodic boundary conditions)
    for i in random_points_x:
        for j in random_points_y:
            ip = (i + delta_i) % N
            jp = (j + delta_j) % M
            im = (i - delta_i) % N
            jm = (j - delta_j) % M

            if stencil_width == 3:
                # Compute finite-difference for the velocity (three-point stencil)
                dvx = v_x[jp, ip] - 2.0 * v_x[j, i] + v_x[jm, im]
                dvy = v_y[jp, ip] - 2.0 * v_y[j, i] + v_y[jm, im]
                dvz = v_z[jp, ip] - 2.0 * v_z[j, i] + v_z[jm, im]

                # Compute finite-difference for the magnetic field (three-point stencil)
                dBx = B_x[jp, ip] - 2.0 * B_x[j, i] + B_x[jm, im]
                dBy = B_y[jp, ip] - 2.0 * B_y[j, i] + B_y[jm, im]
                dBz = B_z[jp, ip] - 2.0 * B_z[j, i] + B_z[jm, im]

                # Compute local mean magnetic field (using all three components)
                Bmx = (B_x[jp, ip] + B_x[j, i] + B_x[jm, im]) / 3.0
                Bmy = (B_y[jp, ip] + B_y[j, i] + B_y[jm, im]) / 3.0
                Bmz = (B_z[jp, ip] + B_z[j, i] + B_z[jm, im]) / 3.0
            elif stencil_width == 2:
                # Compute finite-difference for the velocity (two-point stencil)
                dvx = v_x[jp, ip] - v_x[j, i]
                dvy = v_y[jp, ip] - v_y[j, i]
                dvz = v_z[jp, ip] - v_z[j, i]

                # Compute finite-difference for the magnetic field (two-point stencil)
                dBx = B_x[jp, ip] - B_x[j, i]
                dBy = B_y[jp, ip] - B_y[j, i]
                dBz = B_z[jp, ip] - B_z[j, i]

                # Compute local mean magnetic field (using two components)
                Bmx = (B_x[jp, ip] + B_x[j, i]) / 2.0
                Bmy = (B_y[jp, ip] + B_y[j, i]) / 2.0
                Bmz = (B_z[jp, ip] + B_z[j, i]) / 2.0

            Bmean_mag = (Bmx*Bmx + Bmy*Bmy + Bmz*Bmz) ** 0.5
            if Bmean_mag == 0.0:
                continue

            cos_theta = np.abs(dx * Bmx + dy * Bmy + dz * Bmz) / (r * Bmean_mag)
            theta_val = np.arccos(cos_theta)

            # calculate the angle between the component of dB perpendicular to mean local B and the component of the displacement perpendicular to mean local B
            # then find the index
            dB_perp = np.array([dBz, dBy, dBx]) - (dBz*Bmz + dBy*Bmy + dBx*Bmx) * np.array([Bmz, Bmy, Bmx]) / Bmean_mag**2
            dB_perp_mag = np.sqrt(dB_perp[0]**2 + dB_perp[1]**2 + dB_perp[2]**2)

            dv_perp = np.array([dvz, dvy, dvx]) - (dvz*Bmz + dvy*Bmy + dvx*Bmx) * np.array([Bmz, Bmy, Bmx]) / Bmean_mag**2
            dv_perp_mag = np.sqrt(dv_perp[0]**2 + dv_perp[1]**2 + dv_perp[2]**2)

            displacement_perp = np.array([dz, dy, dx]) - (dz*Bmz + dy*Bmy + dx*Bmx) * np.array([Bmz, Bmy, Bmx]) / Bmean_mag**2
            displacement_perp_mag = np.sqrt(displacement_perp[0]**2 + displacement_perp[1]**2 + displacement_perp[2]**2)

            cos_phi = (displacement_perp[0] * dB_perp[0] + displacement_perp[1] * dB_perp[1] + displacement_perp[2] * dB_perp[2]) / (displacement_perp_mag * dB_perp_mag)
            phi_val = np.arccos(cos_phi)

            # Compute vperp_cross_bperp and vperp_bperp
            vperp_cross_bperp = np.sqrt(np.sum(np.cross(dv_perp, dB_perp)**2))
            vperp_bperp = dv_perp_mag * dB_perp_mag

            # Compute magnitude of velocity and magnetic field fluctuations
            dv = np.sqrt(dvx**2 + dvy**2 + dvz**2)
            dB = np.sqrt(dBx**2 + dBy**2 + dBz**2)

            # Determine theta bin index.
            theta_idx = find_bin_index_binary(theta_val, theta_bin_edges)

            # Determine theta bin index.
            phi_idx = find_bin_index_binary(phi_val, phi_bin_edges)

            # For each p order, determine the quantity bin.
            v_idx = find_bin_index_binary(dv, sf_bin_edges)
            hist[0, ell_idx, theta_idx, phi_idx, v_idx] += 1

            b_idx = find_bin_index_binary(dB, sf_bin_edges)
            hist[1, ell_idx, theta_idx, phi_idx, b_idx] += 1

            vperp_cross_bperp_idx = find_bin_index_binary(vperp_cross_bperp, product_bin_edges)
            hist[2, ell_idx, theta_idx, phi_idx, vperp_cross_bperp_idx] += 1

            vperp_bperp_idx = find_bin_index_binary(vperp_bperp, product_bin_edges)
            hist[3, ell_idx, theta_idx, phi_idx, vperp_bperp_idx] += 1

    return hist


# # -------------------------------
# # Step 4: Loop over all displacements with multiprocess and sum the histograms.
# # -------------------------------
# @njit
# def process_displacement_2(i_disp):
#     dx, dy = displacements[i_disp]
#     if axis == 1:
#         delta = [0, dx, dy]
#     elif axis == 2:
#         delta = [dx, 0, dy]
#     elif axis == 3:
#         delta = [dx, dy, 0]
#     return compute_histogram_for_disp_2D(v_x, v_y, v_z, B_x, B_y, B_z, rho,
#                                       delta[0], delta[1], delta[2],
#                                       N_random_subsamples,
#                                       ell_bin_edges, theta_bin_edges, phi_bin_edges, sf_bin_edges, product_bin_edges,2)
# -------------------------------
# Step 4: Loop over all displacements with multiprocess and sum the histograms.
# -------------------------------
@njit
def process_displacement_3(i_disp):
    dx, dy = displacements[i_disp]
    return compute_histogram_for_disp_2D(v_x, v_y, v_z, B_x, B_y, B_z, rho,
                                      dx, dy, axis,
                                      N_random_subsamples,
                                      ell_bin_edges, theta_bin_edges, phi_bin_edges, sf_bin_edges, product_bin_edges,3)

# For testing, try one displacement:
hist_single = process_displacement_3(0)
print("Histogram shape for a single displacement:", hist_single.shape)


# def process_batch_2(batch_indices):
#     result = process_displacement_2(batch_indices[0])
#     for i in batch_indices[1:]:
#         result += process_displacement_2(i)
#     return result

def process_batch_3(batch_indices):
    result = process_displacement_3(batch_indices[0])
    for i in batch_indices[1:]:
        result += process_displacement_3(i)
    return result

# Create batches (fewer, larger batches)
batch_size = max(1, n_disp // PoolProcesses)  # Aim for ~4 batches per process
batches = [range(i, min(i+batch_size, n_disp)) for i in range(0, n_disp, batch_size)]
print(batch_size,batches[0],batches[-1])

pool = mulitp.Pool(processes=PoolProcesses)
batch_results = pool.map(process_batch_3, batches)
pool.close()
pool.join()

# Flatten results
hist_3 = np.sum(np.array(batch_results), axis=0)
np.savez(f'slice_data/hist_3_{file_name[:-4].split("/")[-1]}_ndisps_{n_disp_total}_Nsubsamples_{N_random_subsamples}.npz',
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
            product_bin_centers=product_bin_centers)


hist_3 = hist_3[:,:,:,:,:-1]
sf_bin_centers = sf_bin_centers[:-1]
product_bin_centers = product_bin_centers[:-1]

# some test plots 
bsf_theta_ell = (np.sum(np.sum(hist_3[1], axis=(2))*sf_bin_centers**2, axis=-1) / np.sum(hist_3[1], axis=(2,3)))**(1/2)
vsf_theta_ell = (np.sum(np.sum(hist_3[0], axis=(2))*sf_bin_centers**2, axis=-1) / np.sum(hist_3[0], axis=(2,3)))**(1/2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 4), sharey=True, sharex=True)
plot = ax1.contourf(theta_bin_centers *180/np.pi, ell_bin_centers, bsf_theta_ell, levels=np.linspace(0,2,21))
cb = fig.colorbar(plot, ax=ax1, location='top', shrink=0.8)
cb.set_label(r"$\langle|\delta B (\ell)|^2 : \theta \rangle^{1/2}$")
ax1.semilogy()

plot = ax2.contourf(theta_bin_centers *180/np.pi, ell_bin_centers, vsf_theta_ell, levels=np.linspace(0,2,21))
cb = fig.colorbar(plot, ax=ax2, location='top', shrink=0.8)
cb.set_label(r"$\langle|\delta v (\ell)|^2 : \theta \rangle^{1/2}$")
ax2.semilogy()

ax1.set_xlabel(r"$\theta$")
ax2.set_xlabel(r"$\theta$")
ax1.set_ylabel(r"$\ell$")

fig.tight_layout()

plt.savefig(f'slice_data/SF_theta_ell_{file_name[:-4].split("/")[-1]}_ndisps_{n_disp_total}_Nsubsamples_{N_random_subsamples}.png', dpi=200)
plt.clf()



p_orders = np.arange(1,11,1)
vsfp = np.array([ (np.sum(np.sum(hist_3[1], axis=(1,2))*sf_bin_centers**p, axis=1) / np.sum(hist_3[1], axis=(1,2,3)))**(1/p) for p in p_orders])

plot = plt.pcolormesh(ell_bin_centers, sf_bin_centers, np.sum(hist_3[1], axis=(1,2)).T/np.sum(np.sum(hist_3[1], axis=(1,2)).T,axis=0), norm=LogNorm(), cmap=cmr.arctic_r)
cb = plt.colorbar(plot)
cb.set_label(r"$N(\delta v|\ell)$")
plt.grid()
plt.axhline(1, color='magenta', linestyle='--', lw=1)
for i, line in enumerate(vsfp):
    plt.loglog(ell_bin_centers, line, color='0.7', linewidth=1.5-i*0.1, alpha=1-i*0.1)
plt.ylabel(r'$|\delta v|$')
plt.xlabel(r'$\ell$')
plt.ylim(bottom = 1e-3)
plt.savefig(f'slice_data/SF_v_ell_{file_name[:-4].split("/")[-1]}_ndisps_{n_disp_total}_Nsubsamples_{N_random_subsamples}.png', dpi=200)
plt.clf()




B0 = np.sqrt(2/beta)

p_orders = np.arange(1,11,1)
vsfp = np.array([ (np.sum(np.sum(hist_3[0], axis=(1,2))*sf_bin_centers**p, axis=1) / np.sum(hist_3[0], axis=(1,2,3)))**(1/p) for p in p_orders])
plot = plt.pcolormesh(ell_bin_centers, sf_bin_centers, np.sum(hist_3[0], axis=(1,2)).T/np.sum(np.sum(hist_3[0], axis=(1,2)).T,axis=0), norm=LogNorm(), cmap=cmr.amethyst_r)
cb = plt.colorbar(plot)
cb.set_label(r"$N(\delta B|\ell)$")
plt.grid()
plt.axhline(B0, color='cyan', linestyle='--', lw=1)
for i, line in enumerate(vsfp):
    plt.loglog(ell_bin_centers, line, color='0.7', linewidth=1.5-i*0.1, alpha=1-i*0.1)
plt.ylabel(r'$|\delta B|$')
plt.xlabel(r'$\ell$')
plt.ylim(bottom = 1e-3)

plt.savefig(f'slice_data/SF_b_ell_{file_name[:-4].split("/")[-1]}_ndisps_{n_disp_total}_Nsubsamples_{N_random_subsamples}.png', dpi=200)
plt.clf()







average_vperp_cross_bperp = np.sum(np.sum(hist_3[2], axis=(1,2))*product_bin_centers, axis=1) / np.sum(hist_3[2], axis=(1,2,3))
average_vperp_bperp = np.sum(np.sum(hist_3[3], axis=(1,2))*product_bin_centers, axis=1) / np.sum(hist_3[3], axis=(1,2,3))

# plt.loglog(ell_bin_centers,average_vperp_cross_bperp, label=r"$\langle|\delta v_\perp \times \delta B_\perp|\rangle$")
# plt.loglog(ell_bin_centers,average_vperp_bperp, label=r"$\langle |\delta v_\perp| |\delta B_\perp| \rangle$")
plt.loglog(ell_bin_centers,average_vperp_cross_bperp/average_vperp_bperp, label=r"$\frac{\langle|\delta v_\perp \times \delta B_\perp|\rangle}{\langle |\delta v_\perp| |\delta B_\perp| \rangle}$")
plt.loglog(ell_bin_centers,0.1*ell_bin_centers**(1/4), color='black', label=r"$\ell^{1/4}$")
plt.loglog(ell_bin_centers,0.2*ell_bin_centers**(1/8), color='grey', label=r"$\ell^{1/8}$")
plt.loglog(ell_bin_centers,0.2*ell_bin_centers**(1/16), color='0.25', label=r"$\ell^{1/16}$")
plt.legend()
plt.savefig(f'slice_data/theta_vb_ell_{file_name[:-4].split("/")[-1]}_ndisps_{n_disp_total}_Nsubsamples_{N_random_subsamples}.png', dpi=200)
plt.clf()




