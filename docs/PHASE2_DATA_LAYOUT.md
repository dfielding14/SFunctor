# Phase 2 selected-cube data layout

Phase 2 extracts only the trusted 21-region pilot census produced by Phase 1.
Each output cube is stored in KJI order:

```text
array[k, j, i] == field[x3, x2, x1]
```

The audited full primitive basename is:

```text
Turb.full_mhd_w_bcc.00024.bin
```

It records simulation time `6.0`, cycle `799945`, eight float32 primitive
fields (`dens`, `velx`, `vely`, `velz`, `eint`, `bcc1`, `bcc2`, `bcc3`), and
local Cartesian dimensions `(nx1, nx2, nx3) = (160, 320, 320)`.  The reader
removes the recorded three ghost zones and exposes active arrays in KJI order
with shape `(320, 320, 160)`.  Selected full-resolution blocks tile the domain
without active-zone overlap.  Physical bounds are checked against the expected
logical-block coordinates during preflight.

The default four-region benchmark subset is:

```text
L640_sub00370
L640_sub03942
L640_sub00579
L640_sub00738
```

Each `L_sub = 640` cube intersects exactly 16 primitive rank files.  The eight
materialized float32 arrays contain `8,388,608,000` payload bytes plus `.npy`
headers.  The 16 referenced primitive files contain `8,389,646,944` bytes
including their headers.

Each completed cube directory contains:

```text
<cube_id>/
  COMPLETE.json
  manifest.json
  cbin_comparison.csv
  catalog_comparison.csv
  fields/
    dens.npy
    velx.npy
    vely.npy
    velz.npy
    eint.npy
    bcc1.npy
    bcc2.npy
    bcc3.npy
```

`manifest.json` records the half-open global IJK bounds, KJI output shape,
source rank paths, source intersections, header identity, raw moments, Phase 1
catalog comparisons, timing, output sizes, checksums, and code provenance.
`COMPLETE.json` is written only after an atomic publish and pins the manifest
checksum.  Missing markers, stale markers, resized arrays, and checksum changes
are rejected by the verifier.

The strict verifier also recomputes the selected primitive-shard SHA-256
digests, enforces canonical paths under the declared source root, checks the
selected source-plan digest, rejects source or target slices that escape their
declared shapes, rechecks asymmetric source-to-output positions, compares every
materialized voxel directly to its primitive source, recomputes raw moments,
and verifies the exact trusted factor-40 `mhd_u_bcc` cbin basename and parsed
snapshot identity.  The official benchmark summary accepts exactly the four
approved cube IDs, semantically recomputes all four held-out stream checks, and
records hashes for every cube completion marker, stream-validation artifact,
comparison table, restart report, and inspection figure.  Consumers must rerun
strict array verification after any dependency changes; the lightweight
benchmark marker is not a substitute for rehashing the materialized arrays.

Field arrays are written as sequential KJI `.npy` streams.  The first real
smoke showed that strided output memmaps can retain excessive allocated blocks
under the project filesystem's progressive Lustre layout.  Publication now
also fails if allocated blocks exceed the larger of `1.5x` logical output or
logical output plus `64 MiB`.  Strict restart verification enforces the same
gate again at verification time.  Lustre allocation can continue to drift
after that snapshot, so later consumers should rerun strict verification.

The metadata-only preflight supports both `--all-pilot` and a deterministic
`probe` action.  The latter checks origin-edge, upper-edge, remote-Morton, and
fixed randomized-stratified regions without reading primitive payloads.
The `stream-validate` action reads four bounded held-out primitive regions
without materializing cubes and compares reconstructable quantities directly
against retained factor-40 primary `cbin` shards.

The benchmark records first-touch timing within one allocation, but the shared
filesystem cache state is uncontrolled.  It is not a controlled cold-read
benchmark.  This is a descriptive timing caveat, not a Phase 3 gate.  Use the
measured operational runtime for planning; no separate cold/warm protocol is
required before continuing.

Primitive read time is not isolated from validation, moment accumulation,
SHA-256 passes, or cache effects.  Forecasts derived from
`total_wall_seconds_before_publish` are extraction-only forecasts, not
end-to-end Slurm allocation estimates.

`--clean-stale-lock` is an operator override.  Use it only after confirming that
no active extraction owns the lock directory.  A mistaken override can permit
concurrent writers.  The Andes wrapper also acquires one action lock under the
output root so separate allocations cannot publish into the same root
concurrently.

The trusted Phase 1 artifact root intentionally retains its historical external
name:

```text
/lustre/orion/ast207/proj-shared/dfielding/Production_plm/prompt1_catalog/t6_final_primary_20260530
```

That Lustre path is not renamed or moved during the phase terminology cleanup.
