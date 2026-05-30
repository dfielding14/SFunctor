import glob
import os
import socket
import time

import numpy as np

from . import bin_convert_new as bc
from .rank_manifest import build_rank_manifest, read_blocks_grouped


def _safe_float_to_str(val: float) -> str:
    """Convert *val* to a filename-safe string (avoids minus/dot)."""
    s = f"{val:.8f}"  # high precision, trimmed later
    s = s.rstrip("0").rstrip(".")  # remove trailing zeros/dot
    s = s.replace("-", "m").replace(".", "p")
    return s or "0"


def _determine_file_part(sim_name: str, file_number: int):
    """Determine file_part string for cache filename based on available data files."""
    if file_number is None:
        rank0_pattern = f"data/data_{sim_name}/bin/rank_00000000/Turb.full_mhd_w_bcc.*.bin"
        rank0_files = sorted(glob.glob(rank0_pattern))
        if rank0_files:
            last_fname = os.path.splitext(rank0_files[-1])[0]
            idx_str = last_fname.split(".")[-1]
            if idx_str.isdigit():
                return f"{int(idx_str):04d}"
            return "latest"
        return "latest"

    if file_number == -1:
        rank0_pattern = f"data/data_{sim_name}/bin/rank_00000000/Turb.full_mhd_w_bcc.*.bin"
        rank0_files = sorted(glob.glob(rank0_pattern))
        if rank0_files:
            last_fname = os.path.splitext(rank0_files[-1])[0]
            idx_str = last_fname.split(".")[-1]
            if idx_str.isdigit():
                return f"{int(idx_str):04d}"
            return "final"
        return "final"

    return f"{file_number:04d}"


def _build_cache_fname(sim_name: str, axis: int, slice_value: float, file_number, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    slice_str = _safe_float_to_str(slice_value)
    file_part = _determine_file_part(sim_name, file_number)
    return os.path.join(
        cache_dir,
        f"{sim_name}_axis{axis}_slice{slice_str}_file{file_part}.npz",
    )


def _acquire_cache_lock(cache_fname: str, *, lock_wait: int, lock_poll: int, lock_stale: int):
    """Acquire a lock to prevent duplicate extractions.

    Returns
    -------
    tuple
        (lock_acquired: bool, cache_available: bool, lock_path: str)
    """
    lock_path = f"{cache_fname}.lock"
    start_time = time.time()
    notified_wait = False

    while True:
        if os.path.exists(cache_fname):
            return False, True, lock_path

        if os.path.exists(lock_path):
            # Remove stale locks
            if lock_stale > 0 and (time.time() - os.path.getmtime(lock_path)) > lock_stale:
                try:
                    os.remove(lock_path)
                    print(f"[extract_2d_slice] Removed stale lock {lock_path}")
                    continue
                except OSError:
                    pass

            # Wait for lock to clear or cache to appear
            if lock_wait > 0 and (time.time() - start_time) < lock_wait:
                if not notified_wait:
                    wait_left = lock_wait - int(time.time() - start_time)
                    print(
                        f"[extract_2d_slice] Lock found at {lock_path}; waiting up to "
                        f"{lock_wait}s (remaining ~{wait_left}s)"
                    )
                    notified_wait = True
                time.sleep(max(1, lock_poll))
                continue

            raise RuntimeError(
                f"[extract_2d_slice] Lock {lock_path} still present after "
                f"{int(time.time() - start_time)}s; aborting to avoid contention."
            )

        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(f"host={socket.gethostname()} pid={os.getpid()} start={time.time()}\n")
            print(f"[extract_2d_slice] Acquired lock {lock_path}")
            return True, False, lock_path
        except FileExistsError:
            # Loop and check again
            continue

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


def _build_meshblock_locations(n_blocks_x1, n_blocks_x2, n_blocks_x3):
    """Return logical meshblock locations in AthenaK Morton/GID order."""
    if max(n_blocks_x1, n_blocks_x2, n_blocks_x3) > 64:
        raise ValueError("Morton meshblock indexing supports at most 64 blocks per axis")

    locations = []
    for i3 in range(n_blocks_x3):
        for i2 in range(n_blocks_x2):
            for i1 in range(n_blocks_x1):
                morton_id = Morton_array_to_int([i1, i2, i3])
                locations.append((morton_id, i3, i2, i1))
    locations.sort()
    return [(i3, i2, i1) for _, i3, i2, i1 in locations]


def _cell_index(edges, value):
    """Return the cell immediately below *value*, clamped to the domain."""
    return int(np.clip(np.searchsorted(edges, value) - 1, 0, len(edges) - 2))


def _load_valid_cache(cache_fname, expected_shape, varnames):
    """Load a cache only if it has the expected geometry and finite primitives."""
    try:
        with np.load(cache_fname, allow_pickle=True) as npz:
            data = {k: npz[k] for k in npz.files}
    except Exception as err:
        print(f"[extract_2d_slice] Ignoring unreadable cache {cache_fname}: {err}")
        return None

    for var in varnames:
        if var not in data or data[var].shape != expected_shape:
            print(f"[extract_2d_slice] Ignoring cache with invalid {var} shape: {cache_fname}")
            return None
        if not np.isfinite(data[var]).all():
            print(f"[extract_2d_slice] Ignoring cache with incomplete or non-finite {var}: {cache_fname}")
            return None
    return data


def extract_2d_slice(
    sim_name,
    axis,
    slice_value,
    file_number=None,
    *,
    save=True,
    cache_dir="slice_data",
    lock_wait=1800,
    lock_poll=5,
    lock_stale=7200,
):
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
    start_time = time.time()
    print(f"[extract_2d_slice] Start sim={sim_name} axis={axis} slice={slice_value} file_number={file_number} save={save}")
    if axis not in (1, 2, 3):
        raise ValueError("axis must be 1, 2, or 3")

    cache_fname = None
    cache_status = "computed"
    lock_path = None
    lock_acquired = False

    input_file_name = f"inputs/{sim_name}.athinput"
    input_file = bc.athinput(input_file_name)
    nx1_meshblock = input_file['meshblock']['nx1']
    nx2_meshblock = input_file['meshblock']['nx2']
    nx3_meshblock = input_file['meshblock']['nx3']
    x1_min = input_file['mesh']['x1min']
    x2_min = input_file['mesh']['x2min']
    x3_min = input_file['mesh']['x3min']
    x1_max = input_file['mesh']['x1max']
    x2_max = input_file['mesh']['x2max']
    x3_max = input_file['mesh']['x3max']
    nx1 = input_file['mesh']['nx1']
    nx2 = input_file['mesh']['nx2']
    nx3 = input_file['mesh']['nx3']
    n_blocks_x1, rem_x1 = divmod(nx1, nx1_meshblock)
    n_blocks_x2, rem_x2 = divmod(nx2, nx2_meshblock)
    n_blocks_x3, rem_x3 = divmod(nx3, nx3_meshblock)
    if rem_x1 or rem_x2 or rem_x3:
        raise ValueError("Global mesh dimensions must be divisible by meshblock dimensions")
    meshblock_locations = _build_meshblock_locations(n_blocks_x1, n_blocks_x2, n_blocks_x3)
    N_meshblocks = len(meshblock_locations)
    Nranks = int(len(glob.glob(f'data/data_{sim_name}/bin/rank_*/')))
    if Nranks == 0:
        raise FileNotFoundError(f"No rank directories found for simulation {sim_name}")
    print(f"[extract_2d_slice] Grid nx=({nx1}, {nx2}, {nx3}) blocks={N_meshblocks} ranks={Nranks} file_part={_determine_file_part(sim_name, file_number)}")

    # Fixed list of variables we will extract
    varnames = ['dens', 'velx', 'vely', 'velz', 'bcc1', 'bcc2', 'bcc3']

    # Prepare the global grid for the slice
    x1f = np.linspace(x1_min, x1_max, nx1 + 1)
    x2f = np.linspace(x2_min, x2_max, nx2 + 1)
    x3f = np.linspace(x3_min, x3_max, nx3 + 1)
    if axis == 1:
        slice_axis = x1f
        slice_shape = (nx3, nx2)
    elif axis == 2:
        slice_axis = x2f
        slice_shape = (nx3, nx1)
    else:
        slice_axis = x3f
        slice_shape = (nx2, nx1)

    # Find the index in the axis closest to the slice_value
    if slice_value <= slice_axis[0]:
        print(f"[extract_2d_slice] Warning: slice_value {slice_value} below domain [{slice_axis[0]}, {slice_axis[-1]}]; clamping to first cell")
    if slice_value >= slice_axis[-1]:
        print(f"[extract_2d_slice] Warning: slice_value {slice_value} above domain [{slice_axis[0]}, {slice_axis[-1]}]; clamping to last cell")
    slice_idx = _cell_index(slice_axis, slice_value)
    slice_centres = 0.5 * (slice_axis[:-1] + slice_axis[1:])
    slice_coord = slice_centres[slice_idx]
    print(f"[extract_2d_slice] slice_idx={slice_idx} (axis length {len(slice_axis)-1})")

    if save:
        cache_fname = _build_cache_fname(sim_name, axis, slice_value, file_number, cache_dir)
        if os.path.exists(cache_fname):
            cached = _load_valid_cache(cache_fname, slice_shape, varnames)
            if cached is not None:
                print(f"[extract_2d_slice] Cache hit: {cache_fname}")
                cache_status = "cache_hit"
                return cached
            quarantine_fname = (
                f"{cache_fname}.invalid.{int(time.time())}.{os.getpid()}"
            )
            os.replace(cache_fname, quarantine_fname)
            print(
                "[extract_2d_slice] Preserved invalid cache as "
                f"{quarantine_fname}; recomputing"
            )
        lock_acquired, cache_available, lock_path = _acquire_cache_lock(
            cache_fname,
            lock_wait=lock_wait,
            lock_poll=lock_poll,
            lock_stale=lock_stale,
        )
        if cache_available:
            cached = _load_valid_cache(cache_fname, slice_shape, varnames)
            if cached is not None:
                print(f"[extract_2d_slice] Cache became available while waiting: {cache_fname}")
                return cached
            raise RuntimeError(f"Cache appeared while waiting but is invalid: {cache_fname}")

    # Prepare empty arrays for each variable
    slice_data = {var: np.full(slice_shape, np.nan) for var in varnames}

    progress_step = max(1, N_meshblocks // 20)
    hits = 0

    def _meshblock_info(gid):
        i3, i2, i1 = meshblock_locations[gid]
        i_start = i1 * nx1_meshblock
        j_start = i2 * nx2_meshblock
        k_start = i3 * nx3_meshblock
        bounds = (
            (x1f[i_start], x1f[i_start + nx1_meshblock]),
            (x2f[j_start], x2f[j_start + nx2_meshblock]),
            (x3f[k_start], x3f[k_start + nx3_meshblock]),
        )
        return i_start, j_start, k_start, bounds

    rank0_files = sorted(glob.glob(f'data/data_{sim_name}/bin/rank_00000000/Turb.full_mhd_w_bcc.*.bin'))
    if not rank0_files:
        if lock_acquired and lock_path is not None:
            os.remove(lock_path)
        raise FileNotFoundError(f"No binary snapshots found for simulation {sim_name}")
    if file_number is None or file_number == -1:
        selected_basename = os.path.basename(rank0_files[-1])
    else:
        if file_number < 0 or file_number >= len(rank0_files):
            if lock_acquired and lock_path is not None:
                os.remove(lock_path)
            raise ValueError(f"file_number {file_number} is out of range. Available files: {len(rank0_files)}")
        selected_basename = os.path.basename(rank0_files[file_number])

    try:
        rank0_filename = f"data/data_{sim_name}/bin/rank_00000000/{selected_basename}"
        manifest = build_rank_manifest(rank0_filename)
        if len(manifest.rank_files) != Nranks:
            raise FileNotFoundError(
                f"Snapshot {selected_basename} exists on {len(manifest.rank_files)} "
                f"of {Nranks} discovered rank directories"
            )
        if manifest.global_shape != (nx1, nx2, nx3):
            raise ValueError(
                f"Rank manifest global shape {manifest.global_shape} does not match "
                f"input file {(nx1, nx2, nx3)}"
            )
        if manifest.meshblock_shape != (nx1_meshblock, nx2_meshblock, nx3_meshblock):
            raise ValueError(
                f"Rank manifest meshblock shape {manifest.meshblock_shape} does not match "
                f"input file {(nx1_meshblock, nx2_meshblock, nx3_meshblock)}"
            )
        if manifest.levels != (0,):
            raise NotImplementedError(
                "2-D extraction currently requires a uniform level-0 manifest; "
                "retain the manifest interface when adding AMR-aware 3-D chunks"
            )
        manifest_by_location = {
            block.logical_location: block
            for block in manifest.blocks
        }
        expected_locations = {
            (i1, i2, i3)
            for i3, i2, i1 in meshblock_locations
        }
        if set(manifest_by_location) != expected_locations:
            raise ValueError(
                "Rank manifest logical locations do not exactly cover the expected "
                "uniform meshblock layout"
            )

        axis_edges = {1: x1f, 2: x2f, 3: x3f}[axis]
        axis_centres = 0.5 * (axis_edges[:-1] + axis_edges[1:])
        slice_idx_minus = (slice_idx - 1) % len(axis_centres)
        slice_idx_plus = (slice_idx + 1) % len(axis_centres)
        target_values = (
            slice_coord,
            axis_centres[slice_idx_minus],
            axis_centres[slice_idx_plus],
        )
        needed_blocks = {
            block
            for target_value in target_values
            for block in manifest.blocks_intersecting(axis, target_value)
        }
        grouped_blocks = read_blocks_grouped(manifest, needed_blocks, quantities=varnames)
        print(
            f"[extract_2d_slice] Manifest validated: {len(manifest.blocks)} blocks; "
            f"loaded {len(needed_blocks)} blocks from "
            f"{len({block.filename for block in needed_blocks})} rank files"
        )
    except Exception:
        if lock_acquired and lock_path is not None:
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                pass
        raise

    def _manifest_record_for_gid(gid):
        i3, i2, i1 = meshblock_locations[gid]
        return manifest_by_location[(i1, i2, i3)]

    def _meshblock_data(record):
        data = dict(grouped_blocks[record])
        x1min, x1max, x2min, x2max, x3min, x3max = record.geometry
        data["x1f"] = np.linspace(x1min, x1max, nx1_meshblock + 1)
        data["x2f"] = np.linspace(x2min, x2max, nx2_meshblock + 1)
        data["x3f"] = np.linspace(x3min, x3max, nx3_meshblock + 1)
        return data

    def _copy_meshblock_plane(out, mb_data, target_value, i_start, j_start, k_start, variables):
        if axis == 1:
            idx = _cell_index(mb_data['x1f'], target_value)
            for var in variables:
                out[var][k_start:k_start + nx3_meshblock, j_start:j_start + nx2_meshblock] = mb_data[var][:, :, idx]
        elif axis == 2:
            idx = _cell_index(mb_data['x2f'], target_value)
            for var in variables:
                out[var][k_start:k_start + nx3_meshblock, i_start:i_start + nx1_meshblock] = mb_data[var][:, idx, :]
        else:
            idx = _cell_index(mb_data['x3f'], target_value)
            for var in variables:
                out[var][j_start:j_start + nx2_meshblock, i_start:i_start + nx1_meshblock] = mb_data[var][idx, :, :]

    try:
        # Loop over all meshblocks (by GID)
        for gid in range(N_meshblocks):
            if gid % progress_step == 0:
                print(f"[extract_2d_slice] progress gid={gid}/{N_meshblocks} hits={hits}")
            i_start, j_start, k_start, bounds = _meshblock_info(gid)

            # Does this meshblock contain the slice?
            mb_min, mb_max = bounds[axis - 1]
            if not (mb_min <= slice_coord < mb_max):
                continue
            hits += 1

            mb_data = _meshblock_data(_manifest_record_for_gid(gid))
            _copy_meshblock_plane(slice_data, mb_data, slice_coord, i_start, j_start, k_start, varnames)

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
        def _fill_single_slice(target_idx):
            out = {var: np.full(slice_shape, np.nan) for var in needed_vec_vars}
            target_value = slice_centres[target_idx]
            for gid in range(N_meshblocks):
                i_start, j_start, k_start, bounds = _meshblock_info(gid)
                mb_min, mb_max = bounds[axis - 1]
                if not (mb_min <= target_value < mb_max):
                    continue
                mb_data = _meshblock_data(_manifest_record_for_gid(gid))
                _copy_meshblock_plane(out, mb_data, target_value, i_start, j_start, k_start, needed_vec_vars)
            return out
    
        minus_slice = _fill_single_slice(slice_idx_minus)
        plus_slice = _fill_single_slice(slice_idx_plus)
    
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

        # A partially assembled plane is scientifically unusable.  Failing
        # here prevents a missing rank file or incomplete meshblock mapping
        # from becoming a successful-looking cache that later drops samples.
        for name, values in slice_data.items():
            if values.shape != slice_shape or not np.isfinite(values).all():
                raise RuntimeError(
                    f"Extracted slice contains incomplete or non-finite {name}: "
                    f"shape={values.shape}, expected={slice_shape}"
                )
    
        # ------------------------------------------------------------------
        # Save to cache if requested and return -----------------------------
        # ------------------------------------------------------------------
        if save:
            # Double-check in case another process finished while we were computing
            if os.path.exists(cache_fname):
                cached = _load_valid_cache(cache_fname, slice_shape, varnames)
                if cached is not None:
                    print(f"[extract_2d_slice] Cache became available after computation: {cache_fname}")
                    cache_status = "cache_ready_after_wait"
                    return cached

            try:
                tmp_cache = f"{cache_fname}.tmp.{os.getpid()}.npz"
                np.savez(tmp_cache, **slice_data)
                os.replace(tmp_cache, cache_fname)
                cache_status = "cached_after_compute"
            except Exception as err:
                # Do not fail the main path if caching fails; just warn.
                if "tmp_cache" in locals() and os.path.exists(tmp_cache):
                    os.remove(tmp_cache)
                print(f"[extract_2d_slice] Warning: could not write cache '{cache_fname}': {err}")

            print(f"[extract_2d_slice] Completed status={cache_status} cache={cache_fname}")
            return slice_data
        else:
            print("[extract_2d_slice] Completed (no cache write requested)")
            return slice_data
    finally:
        if lock_acquired and lock_path is not None:
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                pass
        print(f"[extract_2d_slice] Finished sim={sim_name} axis={axis} slice={slice_value} hits={hits} elapsed={time.time()-start_time:.1f}s")
