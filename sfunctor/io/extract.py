import matplotlib
matplotlib.rc('font', family='serif')
matplotlib.rc('mathtext', fontset='cm')
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['ytick.right'] = True
matplotlib.rcParams['xtick.minor.visible'] = True
matplotlib.rcParams['ytick.minor.visible'] = True
matplotlib.rcParams['lines.dash_capstyle'] = 'round'
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import glob
import numpy as np
import cmasher as cmr
import subprocess
import os
import sys
import matplotlib.patheffects as patheffects
import time

import bin_convert_new as bc

def Morton_array_to_int(arr):
    """
    Convert a three-element array [a, b, c] into an integer (0 <= n < 2^18)
    by interleaving the bits of the 6-bit numbers a, b, and c.

    The array [a, b, c] represents a point in a 64x64x64 grid.
    The integer n is constructed as an 18-bit number:
       n = b17 b16 ... b0

    The bits are interleaved so that:
      - Bits from 'a' go to positions 3*i + 0,
      - Bits from 'b' go to positions 3*i + 1,
      - Bits from 'c' go to positions 3*i + 2,
    for i = 0 to 5 (with bit i being the i-th least significant bit of a, b, or c).
    """
    if len(arr) != 3:
        raise ValueError("Input array must have exactly three elements [a, b, c].")

    a, b, c = arr
    # Check that a, b, and c are in the valid range for 6-bit numbers.
    for coord in (a, b, c):
        if coord < 0 or coord >= 64:
            raise ValueError("Each coordinate must be in the range 0 to 63 (inclusive).")

    n = 0
    # Each coordinate is a 6-bit number, so we iterate over 6 bits.
    for i in range(6):
        # Extract the i-th bit from each coordinate and put it in the proper position:
        n |= ((a >> i) & 1) << (3 * i + 0)
        n |= ((b >> i) & 1) << (3 * i + 1)
        n |= ((c >> i) & 1) << (3 * i + 2)
    return n

def Morton_int_to_array(n):
    """
    Convert an integer (0 <= n < 2^18) into a three-element array [a, b, c]
    by de-interleaving its 18-bit binary representation.

    The 18-bit number is written as:
        b17 b16 ... b0
    Then we assign:
      - a gets bits at positions 0, 3, 6, 9, 12, 15 (least significant bits of each coordinate)
      - b gets bits at positions 1, 4, 7, 10, 13, 16
      - c gets bits at positions 2, 5, 8, 11, 14, 17 (most significant bits of each coordinate)

    Each coordinate is a 6-bit number (range 0 to 63), so the resulting array
    represents a point in a 64x64x64 grid.
    """
    if n < 0 or n >= (1 << 18):
        raise ValueError("n must be in the range 0 to 2^18 - 1 (i.e. 0 <= n < 262144).")

    a = b = c = 0
    # There are 6 bits for each coordinate since 18/3 = 6.
    for i in range(6):
        # Extract the bit for coordinate a from position (3*i + 0)
        a |= ((n >> (3 * i + 0)) & 1) << i
        # Extract the bit for coordinate b from position (3*i + 1)
        b |= ((n >> (3 * i + 1)) & 1) << i
        # Extract the bit for coordinate c from position (3*i + 2)
        c |= ((n >> (3 * i + 2)) & 1) << i

    return [c, b, a]

def extract_2d_slice(sim_name, axis, slice_value, file_number=None, *, save=True, cache_dir="slice_data"):
    """
    Extract a 2D slice from the 3D domain at a given value along the specified axis.
    The function always returns the following variables:
      ['dens', 'velx', 'vely', 'velz', 'bcc1', 'bcc2', 'bcc3']

    axis: 1, 2, or 3 (for x1, x2, x3)
    slice_value: value along the axis to slice at
    file_number: optional file number to use instead of the latest file
    If *save* is True (default) the resulting slice dictionary is cached to
    ``./slice_data`` (created on-demand).  Subsequent calls with identical
    arguments will reload the cached ``*.npz`` instead of recomputing.

    Returns
    -------
    dict
        Mapping of variable names → 2-D numpy arrays for the requested slice.
    """
    input_file_name = f"inputs/{sim_name}.athinput"
    input_file = bc.athinput(input_file_name)
    nx1_meshblock = input_file['meshblock']['nx1']
    nx2_meshblock = input_file['meshblock']['nx2']
    nx3_meshblock = input_file['meshblock']['nx3']
    Nres = input_file['mesh']['nx1']
    N_meshblocks = int(Nres**3 / (nx3_meshblock*nx2_meshblock*nx1_meshblock))
    Nranks = int(len(glob.glob(f'data/data_{sim_name}/bin/rank_*/')))

    # Fixed list of variables we will extract
    varnames = ['dens', 'velx', 'vely', 'velz', 'bcc1', 'bcc2', 'bcc3']

    # Prepare the global grid for the slice
    x1f = np.linspace(-0.5, 0.5, Nres+1)
    x2f = np.linspace(-0.5, 0.5, Nres+1)
    x3f = np.linspace(-0.5, 0.5, Nres+1)
    if axis == 1:
        slice_axis = x1f
        other_axes = (x2f, x3f)
    elif axis == 2:
        slice_axis = x2f
        other_axes = (x1f, x3f)
    elif axis == 3:
        slice_axis = x3f
        other_axes = (x1f, x2f)
    else:
        raise ValueError("axis must be 1, 2, or 3")

    # Find the index in the axis closest to the slice_value
    slice_idx = np.searchsorted(slice_axis, slice_value) - 1
    if slice_idx < 0 or slice_idx >= Nres:
        raise ValueError("slice_value is outside the domain")

    # Prepare empty arrays for each variable
    slice_shape = (Nres, Nres)
    slice_data = {var: np.full(slice_shape, np.nan) for var in varnames}

    # Loop over all meshblocks (by GID)
    for gid in range(N_meshblocks):
        # Determine meshblock logical location (i3, i2, i1) from GID
        i3, i2, i1 = Morton_int_to_array(gid)
        # Compute meshblock bounds in each direction
        mb_x1_min = -0.5 + i1 * nx1_meshblock / Nres
        mb_x1_max = mb_x1_min + nx1_meshblock / Nres
        mb_x2_min = -0.5 + i2 * nx2_meshblock / Nres
        mb_x2_max = mb_x2_min + nx2_meshblock / Nres
        mb_x3_min = -0.5 + i3 * nx3_meshblock / Nres
        mb_x3_max = mb_x3_min + nx3_meshblock / Nres

        # Does this meshblock contain the slice?
        if axis == 1 and not (mb_x1_min <= slice_value <= mb_x1_max):
            continue
        if axis == 2 and not (mb_x2_min <= slice_value <= mb_x2_max):
            continue
        if axis == 3 and not (mb_x3_min <= slice_value <= mb_x3_max):
            continue

        # Find which rank and local_mb_idx this meshblock belongs to
        mb_per_rank_base = N_meshblocks // Nranks
        remainder_mbs = N_meshblocks % Nranks
        current_gid_start_for_rank = 0
        for r_idx in range(Nranks):
            num_mbs_for_this_rank = mb_per_rank_base + (1 if r_idx < remainder_mbs else 0)
            if gid < current_gid_start_for_rank + num_mbs_for_this_rank:
                i_rank = r_idx
                local_mb_idx = gid - current_gid_start_for_rank
                break
            current_gid_start_for_rank += num_mbs_for_this_rank

        files = np.sort(glob.glob(f'data/data_{sim_name}/bin/rank_{i_rank:08d}/Turb.full_mhd_w_bcc.*.bin'))
        if len(files) == 0:
            continue
        # Select file based on file_number parameter
        if file_number is not None:
            if file_number < 0 or file_number >= len(files):
                raise ValueError(f"file_number {file_number} is out of range. Available files: {len(files)}")
            selected_file = files[file_number]
        else:
            selected_file = files[-1]
        # Read meshblock data
        mb_data = bc.read_single_rank_binary_as_athdf(selected_file, meshblock_index_in_file=local_mb_idx)
        # Find the index in the meshblock that matches the slice
        if axis == 1:
            x1f_mb = mb_data['x1f']
            idx = np.searchsorted(x1f_mb, slice_value) - 1
            if 0 <= idx < mb_data[varnames[0]].shape[2]:
                j_start = int(np.floor((mb_x2_min + 0.5) * Nres))
                k_start = int(np.floor((mb_x3_min + 0.5) * Nres))
                for var in varnames:
                    slice_data[var][k_start:k_start+nx3_meshblock, j_start:j_start+nx2_meshblock] = mb_data[var][:, :, idx]
        elif axis == 2:
            x2f_mb = mb_data['x2f']
            idx = np.searchsorted(x2f_mb, slice_value) - 1
            if 0 <= idx < mb_data[varnames[0]].shape[1]:
                i_start = int(np.floor((mb_x1_min + 0.5) * Nres))
                k_start = int(np.floor((mb_x3_min + 0.5) * Nres))
                for var in varnames:
                    slice_data[var][k_start:k_start+nx3_meshblock, i_start:i_start+nx1_meshblock] = mb_data[var][:, idx, :]
        elif axis == 3:
            x3f_mb = mb_data['x3f']
            idx = np.searchsorted(x3f_mb, slice_value) - 1
            if 0 <= idx < mb_data[varnames[0]].shape[0]:
                i_start = int(np.floor((mb_x1_min + 0.5) * Nres))
                j_start = int(np.floor((mb_x2_min + 0.5) * Nres))
                for var in varnames:
                    slice_data[var][j_start:j_start+nx2_meshblock, i_start:i_start+nx1_meshblock] = mb_data[var][idx, :, :]

    # ----------------------------------------------------------------------------------
    # Compute vorticity (omega = curl v) and current (J = curl B) on the extracted slice
    # ----------------------------------------------------------------------------------
    # In order to evaluate derivatives along the slice-normal direction we also need the
    # neighbouring slices (±1 index).  We therefore re-extract those two planes for the
    # velocity and magnetic-field components only.
    #
    # NOTE: throughout we follow the AthenaK convention of k,j,i → z,y,x indexing.
    # The orientation of the in-plane axes depends on the slice orientation:
    #   axis==1  →  slice data shape (k=z, j=y)
    #   axis==2  →  slice data shape (k=z, i=x)
    #   axis==3  →  slice data shape (j=y, i=x)
    # ----------------------------------------------------------------------------------
    needed_vec_vars = ['velx', 'vely', 'velz', 'bcc1', 'bcc2', 'bcc3', 'dens']

    # Convenience: ensure we always have the primitive vector variables in the output
    for v in needed_vec_vars:
        if v not in slice_data:
            slice_data[v] = np.full(slice_shape, np.nan)

    # Helper to fill a slice_data-like dict for a given slice value but only for the
    # variables listed in `needed_vec_vars`.
    def _fill_single_slice(target_value):
        out = {var: np.full(slice_shape, np.nan) for var in needed_vec_vars}
        # Re-run the meshblock loop (copy–pasted inner logic but stripped to essentials)
        for gid in range(N_meshblocks):
            i3, i2, i1 = Morton_int_to_array(gid)
            mb_x1_min = -0.5 + i1 * nx1_meshblock / Nres
            mb_x1_max = mb_x1_min + nx1_meshblock / Nres
            mb_x2_min = -0.5 + i2 * nx2_meshblock / Nres
            mb_x2_max = mb_x2_min + nx2_meshblock / Nres
            mb_x3_min = -0.5 + i3 * nx3_meshblock / Nres
            mb_x3_max = mb_x3_min + nx3_meshblock / Nres

            if axis == 1 and not (mb_x1_min <= target_value <= mb_x1_max):
                continue
            if axis == 2 and not (mb_x2_min <= target_value <= mb_x2_max):
                continue
            if axis == 3 and not (mb_x3_min <= target_value <= mb_x3_max):
                continue

            # Map meshblock → rank, local index
            mb_per_rank_base = N_meshblocks // Nranks
            remainder_mbs = N_meshblocks % Nranks
            current_gid_start_for_rank = 0
            for r_idx in range(Nranks):
                num_mbs_for_this_rank = mb_per_rank_base + (1 if r_idx < remainder_mbs else 0)
                if gid < current_gid_start_for_rank + num_mbs_for_this_rank:
                    i_rank = r_idx
                    local_mb_idx = gid - current_gid_start_for_rank
                    break
                current_gid_start_for_rank += num_mbs_for_this_rank

            files = np.sort(glob.glob(f'data/data_{sim_name}/bin/rank_{i_rank:08d}/Turb.full_mhd_w_bcc.*.bin'))
            if len(files) == 0:
                continue
            selected_file = files[file_number] if file_number is not None else files[-1]
            mb_data = bc.read_single_rank_binary_as_athdf(selected_file, meshblock_index_in_file=local_mb_idx)

            if axis == 1:
                x1f_mb = mb_data['x1f']
                idx = np.searchsorted(x1f_mb, target_value) - 1
                if 0 <= idx < mb_data[needed_vec_vars[0]].shape[2]:
                    j_start = int(np.floor((mb_x2_min + 0.5) * Nres))
                    k_start = int(np.floor((mb_x3_min + 0.5) * Nres))
                    for var in needed_vec_vars:
                        out[var][k_start:k_start+nx3_meshblock, j_start:j_start+nx2_meshblock] = mb_data[var][:, :, idx]
            elif axis == 2:
                x2f_mb = mb_data['x2f']
                idx = np.searchsorted(x2f_mb, target_value) - 1
                if 0 <= idx < mb_data[needed_vec_vars[0]].shape[1]:
                    i_start = int(np.floor((mb_x1_min + 0.5) * Nres))
                    k_start = int(np.floor((mb_x3_min + 0.5) * Nres))
                    for var in needed_vec_vars:
                        out[var][k_start:k_start+nx3_meshblock, i_start:i_start+nx1_meshblock] = mb_data[var][:, idx, :]
            elif axis == 3:
                x3f_mb = mb_data['x3f']
                idx = np.searchsorted(x3f_mb, target_value) - 1
                if 0 <= idx < mb_data[needed_vec_vars[0]].shape[0]:
                    i_start = int(np.floor((mb_x1_min + 0.5) * Nres))
                    j_start = int(np.floor((mb_x2_min + 0.5) * Nres))
                    for var in needed_vec_vars:
                        out[var][j_start:j_start+nx2_meshblock, i_start:i_start+nx1_meshblock] = mb_data[var][idx, :, :]
        return out

    # Identify neighbouring slice indices with periodic wrapping
    axis_edges = {
        1: x1f,
        2: x2f,
        3: x3f,
    }[axis]
    axis_centres = 0.5 * (axis_edges[:-1] + axis_edges[1:])
    slice_idx_minus = (slice_idx - 1) % Nres
    slice_idx_plus  = (slice_idx + 1) % Nres
    slice_val_minus = axis_centres[slice_idx_minus]
    slice_val_plus  = axis_centres[slice_idx_plus]

    minus_slice = _fill_single_slice(slice_val_minus)
    plus_slice  = _fill_single_slice(slice_val_plus)

    # Grid spacing (assumed uniform but computed from edges)
    dx = x1f[1] - x1f[0]
    dy = x2f[1] - x2f[0]
    dz = x3f[1] - x3f[0]

    # Short-hands for central & neighbour planes
    vxc, vyc, vzc = slice_data['velx'], slice_data['vely'], slice_data['velz']
    vxp, vyp, vzp = plus_slice['velx'], plus_slice['vely'], plus_slice['velz']
    vxm, vym, vzm = minus_slice['velx'], minus_slice['vely'], minus_slice['velz']
    bxc, byc, bzc = slice_data['bcc1'], slice_data['bcc2'], slice_data['bcc3']
    bxp, byp, bzp = plus_slice['bcc1'], plus_slice['bcc2'], plus_slice['bcc3']
    bxm, bym, bzm = minus_slice['bcc1'], minus_slice['bcc2'], minus_slice['bcc3']

    # Derivative helpers (periodic via np.roll)
    def d_dy(arr):
        if axis == 1:
            return (np.roll(arr, -1, axis=1) - np.roll(arr, 1, axis=1)) / (2 * dy)
        elif axis == 3:
            return (np.roll(arr, -1, axis=0) - np.roll(arr, 1, axis=0)) / (2 * dy)
        else:  # axis == 2 → y is off-plane
            return (plus_slice[arr_name] - minus_slice[arr_name]) / (2 * dy)  # placeholder, replaced below

    def d_dz(arr):
        if axis in (1, 2):  # within plane, k dimension is first axis
            return (np.roll(arr, -1, axis=0) - np.roll(arr, 1, axis=0)) / (2 * dz)
        else:  # axis == 3, z is off-plane
            return (plus_slice[arr_name] - minus_slice[arr_name]) / (2 * dz)  # placeholder

    # Because the mapping of (x,y,z) derivatives depends on the slice orientation
    # we treat each case explicitly to keep the logic clear.
    if axis == 1:
        # --------------------------------------
        # x-normal slice → in-plane (y,z)
        # --------------------------------------
        dvy_dz = (np.roll(vyc, -1, axis=0) - np.roll(vyc, 1, axis=0)) / (2 * dz)
        dvz_dy = (np.roll(vzc, -1, axis=1) - np.roll(vzc, 1, axis=1)) / (2 * dy)
        dvx_dy = (np.roll(vxc, -1, axis=1) - np.roll(vxc, 1, axis=1)) / (2 * dy)
        dvx_dz = (np.roll(vxc, -1, axis=0) - np.roll(vxc, 1, axis=0)) / (2 * dz)
        dvz_dx = (vzp - vzm) / (2 * dx)
        dvy_dx = (vyp - vym) / (2 * dx)

        db_y_dz = (np.roll(byc, -1, axis=0) - np.roll(byc, 1, axis=0)) / (2 * dz)
        db_z_dy = (np.roll(bzc, -1, axis=1) - np.roll(bzc, 1, axis=1)) / (2 * dy)
        db_x_dy = (np.roll(bxc, -1, axis=1) - np.roll(bxc, 1, axis=1)) / (2 * dy)
        db_x_dz = (np.roll(bxc, -1, axis=0) - np.roll(bxc, 1, axis=0)) / (2 * dz)
        db_z_dx = (bzp - bzm) / (2 * dx)
        db_y_dx = (byp - bym) / (2 * dx)

    elif axis == 2:
        # --------------------------------------
        # y-normal slice → in-plane (x,z)
        # --------------------------------------
        # Here central slice shape (k=z, i=x)
        dvz_dy = (vzp - vzm) / (2 * dy)
        dvy_dz = (np.roll(vyc, -1, axis=0) - np.roll(vyc, 1, axis=0)) / (2 * dz)
        dvx_dz = (np.roll(vxc, -1, axis=0) - np.roll(vxc, 1, axis=0)) / (2 * dz)
        dvz_dx = (np.roll(vzc, -1, axis=1) - np.roll(vzc, 1, axis=1)) / (2 * dx)
        dvx_dy = (vxp - vxm) / (2 * dy)
        dvy_dx = (np.roll(vyc, -1, axis=1) - np.roll(vyc, 1, axis=1)) / (2 * dx)

        db_y_dz = (np.roll(byc, -1, axis=0) - np.roll(byc, 1, axis=0)) / (2 * dz)
        db_z_dy = (bzp - bzm) / (2 * dy)
        db_x_dz = (np.roll(bxc, -1, axis=0) - np.roll(bxc, 1, axis=0)) / (2 * dz)
        db_z_dx = (np.roll(bzc, -1, axis=1) - np.roll(bzc, 1, axis=1)) / (2 * dx)
        db_x_dy = (bxp - bxm) / (2 * dy)
        db_y_dx = (np.roll(byc, -1, axis=1) - np.roll(byc, 1, axis=1)) / (2 * dx)

    else:  # axis == 3
        # --------------------------------------
        # z-normal slice → in-plane (x,y)
        # --------------------------------------
        dvy_dz = (vyp - vym) / (2 * dz)
        dvz_dy = (np.roll(vzc, -1, axis=0) - np.roll(vzc, 1, axis=0)) / (2 * dy)
        dvx_dz = (vxp - vxm) / (2 * dz)
        dvz_dx = (np.roll(vzc, -1, axis=1) - np.roll(vzc, 1, axis=1)) / (2 * dx)
        dvx_dy = (np.roll(vxc, -1, axis=0) - np.roll(vxc, 1, axis=0)) / (2 * dy)
        dvy_dx = (np.roll(vyc, -1, axis=1) - np.roll(vyc, 1, axis=1)) / (2 * dx)

        db_y_dz = (byp - bym) / (2 * dz)
        db_z_dy = (np.roll(bzc, -1, axis=0) - np.roll(bzc, 1, axis=0)) / (2 * dy)
        db_x_dz = (bxp - bxm) / (2 * dz)
        db_z_dx = (np.roll(bzc, -1, axis=1) - np.roll(bzc, 1, axis=1)) / (2 * dx)
        db_x_dy = (np.roll(bxc, -1, axis=0) - np.roll(bxc, 1, axis=0)) / (2 * dy)
        db_y_dx = (np.roll(byc, -1, axis=1) - np.roll(byc, 1, axis=1)) / (2 * dx)

    # Curl of velocity → vorticity
    omega_x = dvz_dy - dvy_dz
    omega_y = dvx_dz - dvz_dx
    omega_z = dvy_dx - dvx_dy

    # Curl of magnetic field → current density (up to constants)
    J_x = db_z_dy - db_y_dz
    J_y = db_x_dz - db_z_dx
    J_z = db_y_dx - db_x_dy


    # ------------------------------------------------------------------
    # Magnetic curvature vector: (b·∇)b where b = B/|B|
    # ------------------------------------------------------------------
    # Get density from the slice data
    rho = slice_data['dens']
      
    # Compute |B| with small epsilon to avoid division by zero
    B_mag = np.sqrt(bxc**2 + byc**2 + bzc**2 + 1e-10)

    # Unit magnetic field vector
    bx_unit = bxc / B_mag
    by_unit = byc / B_mag
    bz_unit = bzc / B_mag

    # Need derivatives of unit vector components
    # For neighboring slices, compute unit vectors
    B_mag_plus = np.sqrt(bxp**2 + byp**2 + bzp**2 + 1e-10)
    B_mag_minus = np.sqrt(bxm**2 + bym**2 + bzm**2 + 1e-10)

    bx_unit_plus = bxp / B_mag_plus
    by_unit_plus = byp / B_mag_plus
    bz_unit_plus = bzp / B_mag_plus

    bx_unit_minus = bxm / B_mag_minus
    by_unit_minus = bym / B_mag_minus
    bz_unit_minus = bzm / B_mag_minus

    # Compute derivatives of unit vector components based on slice orientation
    if axis == 1:
        # x-normal slice → in-plane (y,z)
        dbx_dx = (bx_unit_plus - bx_unit_minus) / (2 * dx)
        dbx_dy = (np.roll(bx_unit, -1, axis=1) - np.roll(bx_unit, 1, axis=1)) / (2 * dy)
        dbx_dz = (np.roll(bx_unit, -1, axis=0) - np.roll(bx_unit, 1, axis=0)) / (2 * dz)
        
        dby_dx = (by_unit_plus - by_unit_minus) / (2 * dx)
        dby_dy = (np.roll(by_unit, -1, axis=1) - np.roll(by_unit, 1, axis=1)) / (2 * dy)
        dby_dz = (np.roll(by_unit, -1, axis=0) - np.roll(by_unit, 1, axis=0)) / (2 * dz)
        
        dbz_dx = (bz_unit_plus - bz_unit_minus) / (2 * dx)
        dbz_dy = (np.roll(bz_unit, -1, axis=1) - np.roll(bz_unit, 1, axis=1)) / (2 * dy)
        dbz_dz = (np.roll(bz_unit, -1, axis=0) - np.roll(bz_unit, 1, axis=0)) / (2 * dz)
        
    elif axis == 2:
        # y-normal slice → in-plane (x,z)
        dbx_dx = (np.roll(bx_unit, -1, axis=1) - np.roll(bx_unit, 1, axis=1)) / (2 * dx)
        dbx_dy = (bx_unit_plus - bx_unit_minus) / (2 * dy)
        dbx_dz = (np.roll(bx_unit, -1, axis=0) - np.roll(bx_unit, 1, axis=0)) / (2 * dz)
        
        dby_dx = (np.roll(by_unit, -1, axis=1) - np.roll(by_unit, 1, axis=1)) / (2 * dx)
        dby_dy = (by_unit_plus - by_unit_minus) / (2 * dy)
        dby_dz = (np.roll(by_unit, -1, axis=0) - np.roll(by_unit, 1, axis=0)) / (2 * dz)
        
        dbz_dx = (np.roll(bz_unit, -1, axis=1) - np.roll(bz_unit, 1, axis=1)) / (2 * dx)
        dbz_dy = (bz_unit_plus - bz_unit_minus) / (2 * dy)
        dbz_dz = (np.roll(bz_unit, -1, axis=0) - np.roll(bz_unit, 1, axis=0)) / (2 * dz)
        
    else:  # axis == 3
        # z-normal slice → in-plane (x,y)
        dbx_dx = (np.roll(bx_unit, -1, axis=1) - np.roll(bx_unit, 1, axis=1)) / (2 * dx)
        dbx_dy = (np.roll(bx_unit, -1, axis=0) - np.roll(bx_unit, 1, axis=0)) / (2 * dy)
        dbx_dz = (bx_unit_plus - bx_unit_minus) / (2 * dz)
        
        dby_dx = (np.roll(by_unit, -1, axis=1) - np.roll(by_unit, 1, axis=1)) / (2 * dx)
        dby_dy = (np.roll(by_unit, -1, axis=0) - np.roll(by_unit, 1, axis=0)) / (2 * dy)
        dby_dz = (by_unit_plus - by_unit_minus) / (2 * dz)
        
        dbz_dx = (np.roll(bz_unit, -1, axis=1) - np.roll(bz_unit, 1, axis=1)) / (2 * dx)
        dbz_dy = (np.roll(bz_unit, -1, axis=0) - np.roll(bz_unit, 1, axis=0)) / (2 * dy)
        dbz_dz = (bz_unit_plus - bz_unit_minus) / (2 * dz)

    # Compute (b·∇)b components
    curv_x = bx_unit * dbx_dx + by_unit * dbx_dy + bz_unit * dbx_dz
    curv_y = bx_unit * dby_dx + by_unit * dby_dy + bz_unit * dby_dz
    curv_z = bx_unit * dbz_dx + by_unit * dbz_dy + bz_unit * dbz_dz

    # ------------------------------------------------------------------
    # Density gradient: ∇ρ
    # ------------------------------------------------------------------
    # Need density from neighboring slices
    rho_plus = plus_slice.get('dens', np.full_like(rho, np.nan))
    rho_minus = minus_slice.get('dens', np.full_like(rho, np.nan))

    if axis == 1:
        # x-normal slice → in-plane (y,z)
        grad_rho_x = (rho_plus - rho_minus) / (2 * dx)
        grad_rho_y = (np.roll(rho, -1, axis=1) - np.roll(rho, 1, axis=1)) / (2 * dy)
        grad_rho_z = (np.roll(rho, -1, axis=0) - np.roll(rho, 1, axis=0)) / (2 * dz)
        
    elif axis == 2:
        # y-normal slice → in-plane (x,z)
        grad_rho_x = (np.roll(rho, -1, axis=1) - np.roll(rho, 1, axis=1)) / (2 * dx)
        grad_rho_y = (rho_plus - rho_minus) / (2 * dy)
        grad_rho_z = (np.roll(rho, -1, axis=0) - np.roll(rho, 1, axis=0)) / (2 * dz)
        
    else:  # axis == 3
        # z-normal slice → in-plane (x,y)
        grad_rho_x = (np.roll(rho, -1, axis=1) - np.roll(rho, 1, axis=1)) / (2 * dx)
        grad_rho_y = (np.roll(rho, -1, axis=0) - np.roll(rho, 1, axis=0)) / (2 * dy)
        grad_rho_z = (rho_plus - rho_minus) / (2 * dz)

    slice_data['vortx'] = omega_x
    slice_data['vorty'] = omega_y
    slice_data['vortz'] = omega_z
    slice_data['currx'] = J_x
    slice_data['curry'] = J_y
    slice_data['currz'] = J_z
    slice_data['curvx'] = curv_x
    slice_data['curvy'] = curv_y
    slice_data['curvz'] = curv_z
    slice_data['grad_rho_x'] = grad_rho_x
    slice_data['grad_rho_y'] = grad_rho_y
    slice_data['grad_rho_z'] = grad_rho_z

    # ------------------------------------------------------------------
    # Cache handling ----------------------------------------------------
    # ------------------------------------------------------------------
    if save:
        os.makedirs(cache_dir, exist_ok=True)

        def _safe_float_to_str(val: float) -> str:
            """Convert *val* to a filename-safe string (avoids minus/dot)."""
            s = f"{val:.8f}"  # high precision, trimmed later
            s = s.rstrip("0").rstrip(".")  # remove trailing zeros/dot
            s = s.replace("-", "m").replace(".", "p")
            return s or "0"

        slice_str = _safe_float_to_str(slice_value)

        # Determine explicit file index for naming when file_number is None
        if file_number is None:
            rank0_pattern = f"data/data_{sim_name}/bin/rank_00000000/Turb.full_mhd_w_bcc.*.bin"
            rank0_files = sorted(glob.glob(rank0_pattern))
            if rank0_files:
                last_fname = os.path.splitext(rank0_files[-1])[0]
                # Expected pattern ....<number>
                idx_str = last_fname.split(".")[-1]
                if idx_str.isdigit():
                    file_part = f"{int(idx_str):04d}"
                else:
                    file_part = "latest"
            else:
                file_part = "latest"
        else:
            file_part = f"{file_number:04d}"

        cache_fname = os.path.join(
            cache_dir,
            f"{sim_name}_axis{axis}_slice{slice_str}_file{file_part}.npz",
        )

        if os.path.exists(cache_fname):
            with np.load(cache_fname, allow_pickle=True) as npz:
                return {k: npz[k] for k in npz.files}

    # ------------------------------------------------------------------
    # Save to cache if requested ----------------------------------------
    # ------------------------------------------------------------------

    if save:
        try:
            np.savez(cache_fname, **slice_data)
        except Exception as err:
            # Do not fail the main path if caching fails; just warn.
            print(f"[extract_2d_slice] Warning: could not write cache '{cache_fname}': {err}")

    return slice_data
