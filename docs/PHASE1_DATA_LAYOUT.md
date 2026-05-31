# Phase 1 cbin Data Layout

## Scope

This report covers the final nonrelativistic MHD snapshot at simulation time
`t=6.0`, cycle `799945`. The full domain is `10240^3` fine cells with periodic
boundaries. Each full-resolution rank file owns one `160 x 320 x 320`
mesh block, and there are `64 x 32 x 32 = 65536` rank shards.

The corrected environmental scales are:

```text
L_sub / dx = 80, 160, 320, 640, 1280
```

All five scales tile the domain exactly. `L_sub=1280` has `8^3 = 512`
non-overlapping cubes.

## Authoritative Sources

- Reader: `/ccs/home/dfielding/athenak-df/vis/python/bin_convert.py`
- Writer: `/ccs/home/dfielding/athenak-df/src/outputs/coarsened_binary.cpp`
- Variable mapping: `/ccs/home/dfielding/athenak-df/src/outputs/basetype_output.cpp`
- Simulation input:
  `/lustre/orion/ast207/proj-shared/dfielding/Production_plm/inputs/Turb_10240_beta25_dedt025_plm.athinput`

## Main Products

The cbin trees are siblings of `bin`, not children of it:

```text
data_Turb_10240_beta25_dedt025_plm/
  bin/rank_00000000/Turb.full_mhd_w_bcc.00024.bin
  cbin_mhd_u_bcc_40/rank_00000000/Turb.mhd_u_bcc.00024.cbin
  cbin_mhd_u_bcc_80/rank_00000000/Turb.mhd_u_bcc.00024.cbin
  cbin_mhd_u_bcc_160/rank_00000000/Turb.mhd_u_bcc.00024.cbin
```

Join products by header time and cycle, not by output suffix.

The moment-bearing `mhd_u_bcc` files save conserved variables and
cell-centered magnetic fields:

```text
dens mom1 mom2 mom3 ener bcc1 bcc2 bcc3
```

Each scalar has `_1st`, `_2nd`, `_3rd`, and `_4th` raw-moment fields. The
payload therefore contains 32 float32 arrays. `mom1..3` are conserved
momenta, not primitive velocities. `ener` is conserved total MHD energy.

The full-resolution files save primitive quantities:

```text
dens velx vely velz eint bcc1 bcc2 bcc3
```

## Ordering And Shapes

Payload arrays use NumPy order `[k,j,i] = [x3,x2,x1]`. Logical mesh-block
coordinates use `(x1,x2,x3)`.

| cbin kernel | global grid | rank-local payload shape `[k,j,i]` |
|---:|---:|---:|
| 40 | `256^3` | `(8,8,4)` |
| 80 | `128^3` | `(4,4,2)` |
| 160 | `64^3` | `(2,2,1)` |

Each cbin record contains six int32 local bounds, four int32 logical
coordinates, six float64 mesh-block bounds, and contiguous float32 payloads.
The inspected files are little-endian.

Rank IDs are Morton ordered rather than Cartesian ordered. Production code
must cache the verified logical-coordinate-to-rank map from shard metadata.

## Production Strategy

The complete census uses one strict streamed assembly of `mhd_u_bcc_80`.
Raw moments are merged before central moments are calculated. Factor-40 and
factor-160 trees remain validation oracles. SGS products are intentionally
excluded because they are not trustworthy.

Every full scan rejects missing shards, duplicated logical owners, schema
drift, time/cycle disagreement, geometry disagreement, truncated payloads,
coverage holes, coverage overlaps, and non-finite payload values. Reused raw
caches and catalogs are accepted only when their external manifests match the
input root, snapshot time and cycle, schema, source scale, rank-map hash,
source-code hashes, the passing validation token, and a recomputed filesystem
inventory digest over all `65536` expected source shards. The inventory digest
uses rank, basename, byte size, nanosecond modification time, and nanosecond
change time so a later non-rank-zero source replacement invalidates cache
reuse without rereading every payload byte.

## Andes Queue Convention

Live scheduler inspection on May 30, 2026 found CPU partition `batch`, 32
CPUs per node, and QoS `normal`. No debug partition or debug QoS is currently
exposed. Phase 1 jobs therefore use explicit `#SBATCH -p batch` with short
wall limits for smoke and validation stages. Legacy scripts containing
`#SBATCH -q debug` are stale for the current scheduler configuration.
