# Phase 3a: science-quality parallel finite-domain 3D estimator remediation

Read `phase0.md`, `phase2.md`, `phase3.md`, `PHASE2_STATUS_UPDATE.md`,
`PHASE3_STATUS_UPDATE.md`, and `docs/PHASE3_FINITE_DOMAIN.md` before starting.

Begin this phase from the committed Phase 3 baseline:

```text
ec883a7 Add Phase 1-3 finite-domain validation workflow
```

Phase 3 completed its intended software-validation smoke test. It established
that the non-periodic 3D pair sampler is numerically sound, source-bound,
restartable, and operationally bounded. It also documented a scientific
NO-GO for the 21-region Phase 4 pilot because the initial seven-radius
configuration was intentionally conservative and not sufficient for a
science-quality large-scale interpretation.

This phase remediates that scientific configuration on the same four retained
`L_sub = 640` cubes. Do not launch the 21-region Phase 4 pilot until the
Phase 3a gate is documented.

==================================================
OBJECTIVE
==================================================

Build and validate a science-quality, parallel, non-periodic 3D
structure-function pipeline for extracted `L_sub = 640` cubes.

The revised baseline must:
- extend the two-point analysis through `ell_max = 320 = L_sub / 2`;
- implement labeled 2-point, 3-point, and 5-point 3D increment filters;
- use at least `32` separation bins and evaluate whether `64` or `128` bins
  are preferable;
- resolve large-scale behavior without forcing a power law through
  grid-adjacent dissipative scales;
- measure uncertainty with spatial block resampling and sampling-convergence
  tests;
- parallelize displacement work across Slurm nodes and node-local CPU
  workers, following the validated 2D production architecture where
  appropriate;
- preserve explicit non-periodic endpoint handling and offset-resolved
  finite-support accounting;
- define a measured GO or NO-GO recommendation for the bounded 21-region
  Phase 4 pilot.

The goal is not to demand a perfectly straight structure-function curve.
The goal is to report large-scale behavior, fitted intervals, local slopes,
and uncertainty honestly enough that the Phase 4 environmental comparison is
defensible.

==================================================
BOUNDED SCOPE
==================================================

Reuse the four immutable Phase 2 cubes:

```text
L640_sub00370  low dBB
L640_sub03942  median dBB
L640_sub00579  high dBB
L640_sub00738  weak mean field
```

Keep the first Phase 3a matrix deliberately narrow:

```text
L_sub = 640
q = B, u
p = 2
primary baseline: 2-point increments through ell_max = 320
```

The Phase 3a API and validation suite must also implement labeled 3-point and
5-point filters. After the primary 2-point baseline passes, run bounded
four-cube comparison products with the stencil-specific scale limits defined
below.

Do not:
- rerun the Phase 1 census;
- re-extract the four retained Phase 2 cubes merely to begin this work;
- launch the 21-region Phase 4 pilot;
- launch `L_sub = 1280` work;
- use `mhd_sgs` or `mhd_dynamo_ks`;
- add the full compressible-MHD variable matrix until the baseline estimator
  and parallel reducer pass;
- add higher-order `p` production work until the `p = 2` path passes;
- wrap pairs across extracted-cube boundaries.

==================================================
INHERITED HARD STOPS
==================================================

Retain the durable Phase 0, Phase 2, and Phase 3 operational and
data-provenance hard stops:
- treat the trusted Phase 1 tree as read-only;
- reuse validated Phase 2 extraction products and manifests;
- use only Andes CPU Slurm allocations for heavy work;
- register every nontrivial allocation before submission;
- write Slurm stdout and stderr under `logs/`;
- do not add Slurm email-notification directives;
- keep at most one debug job queued or running at a time;
- use unique restartable output directories;
- publish source-bound completion markers;
- preserve KJI storage and IJK displacement conventions explicitly;
- stop after the bounded four-cube Phase 3a validation campaign.

The Phase 3 limit `ell_max <= L_sub / 4` was a smoke-test scope limit. This
Phase 3a document explicitly supersedes that local limit for `L_sub = 640`
stencil validation with the initial footprint-bounded limits:

```text
2-point: ell_max = L_sub / 2 = 320
3-point: ell_max = L_sub / 4 = 160
5-point: ell_max = L_sub / 8 = 80
```

These follow the existing 2D production convention. Measure the realized
finite support rather than treating the nominal limits as proof that every
outer-shell statistic is usable.

==================================================
TASK 1: FREEZE THE SCIENCE-QUALITY SEPARATION DESIGN
==================================================

Replace the seven-radius Phase 3 smoke census:

```text
4, 8, 16, 32, 64, 96, 128
```

with denser, source-bound separation designs through:

```text
2-point: ell_max = 320
3-point: ell_max = 160
5-point: ell_max = 80
```

Evaluate at least:

```text
32 separation bins
64 separation bins
128 separation bins
```

Use logarithmic or documented hybrid spacing. Retain small scales for
diagnostic plots and dissipation-range inspection, but do not assume that
grid-adjacent bins belong in a physical slope fit.

Generate deterministic approximately uniform 3D displacement directions for
each separation bin and retained stencil. Require:
- signed closure where required by the support mode;
- exact integer offsets after grid rounding;
- removal of zero and duplicate offsets;
- a strict post-rounding stencil-specific `ell_max` check;
- explicit direction counts per bin;
- enough angular coverage to populate `parallel`, `perpendicular`, `xi`, and
  `lambda` diagnostics;
- a source-bound displacement manifest with checksums and configuration.

Test at least two direction densities per separation bin. Do not assume that
the denser census is automatically necessary. Use convergence evidence and
accepted directional counts to select the retained design.

Report:
- stencil label;
- requested and realized separation-bin edges;
- requested and realized offsets per bin;
- directional occupancy;
- post-rounding duplicates;
- post-rounding exclusions;
- maximum realized separation;
- displacement-manifest checksum.

==================================================
TASK 2: IMPLEMENT LABELED 2-POINT, 3-POINT, AND 5-POINT FILTERS
==================================================

Port the existing 2D production increment filters into the non-periodic 3D
path. Preserve the labels and normalizations exactly:

```text
2-point:
    delta_2 q = q(x + r) - q(x)

3-point:
    delta_3 q = [q(x + r) - 2 q(x) + q(x - r)] / sqrt(3)

5-point:
    delta_5 q =
        [q(x - 2r) - 4 q(x - r) + 6 q(x)
         - 4 q(x + r) + q(x + 2r)] / sqrt(35)
```

These are distinct normalized increment filters. They are not
higher-accuracy approximations to one interchangeable statistic. Never merge,
compare, fit, or plot their results without explicit stencil labels.

For extracted non-periodic cubes, require every stencil point to remain inside
the cube:

```text
2-point: x, x + r
3-point: x - r, x, x + r
5-point: x - 2r, x - r, x, x + r, x + 2r
```

Implement stencil-aware valid-origin bounds, shell-local shared bounds,
all-valid-origin bounds, eligible counts, sampled counts, excluded counts, and
endpoint assertions. Do not use modulo indexing.

Define and document the stencil-local magnetic field used for directional
conditioning. Begin by matching the existing 2D filter conventions:

```text
2-point:
    B_loc,2 = [B(x + r) + B(x)] / 2

3-point:
    B_loc,3 = [B(x + r) + B(x) + B(x - r)] / 3

5-point:
    B_loc,5 =
        [B(x - 2r) + 4 B(x - r) + 6 B(x)
         + 4 B(x + r) + B(x + 2r)] / 16
```

Exercise these definitions against direct synthetic calculations and a slow
3D oracle. Keep the 2-point pair-local definition as the primary
Chen/Mallet-style science baseline. Treat 3-point and 5-point results as
separately labeled scale-filter comparisons unless later evidence justifies a
broader interpretation.

==================================================
TASK 3: REVISE THE NON-PERIODIC SUPPORT POLICY
==================================================

The Phase 3 `nested_core` implementation uses one shared interior origin box
for every retained offset. This is a useful conservative diagnostic, but it
collapses at `ell_max = 320` and must not be promoted automatically to the
science-quality estimator.

Compute support separately for each stencil because wider filters require
more in-domain points and lose support more rapidly.

Implement and compare at least:

## Mode A: shell-local shared support

For each separation shell, intersect the valid-origin boxes for the offsets
assigned to that shell and stencil. Every directional comparison within that
shell and stencil uses a common spatial support region.

Record:
- KJI bounds;
- retained volume;
- retained volume fraction;
- eligible origins;
- sampled origins;
- accepted and excluded pairs;
- support versus separation;
- support versus direction.

Reject a shell-local result when its shared support is too small for a
defensible measurement. Do not allow a nominally fair but nearly empty shell
to masquerade as a robust outer-scale statistic.

## Mode B: all valid origins

For each displacement and stencil, use its complete non-periodic valid-origin
box. Never wrap endpoints. Record the same offset-resolved support diagnostics.

The existing 2-point implementation calls this mode `all_valid_pairs`. Preserve
that historical label where compatibility requires it, but use a
stencil-aware schema name for new 3-point and 5-point products. Those products
contain valid stencil tuples, not merely endpoint pairs.

This mode retains more volume, especially at large `ell`, but different
offsets see different spatial regions. Treat that difference as a measurable
support effect rather than hiding it.

## Mode C: original global nested core

Retain the original Phase 3 global `nested_core` as a labeled diagnostic where
it remains non-empty. It is useful for regression checks and for quantifying
why one core shared across all scales is unsuitable at `ell_max = 320`.

Do not choose a final primary estimator in advance. Compare shell-local shared
support and all-valid-origin results with spatial block uncertainty. Select and
document a production policy only after measuring:
- stencil dependence;
- support fraction versus `ell`;
- support anisotropy versus direction;
- directional-curve differences;
- local-slope differences;
- fitted-interval differences;
- block-resampled uncertainty;
- sampling-convergence uncertainty.

If shell-local support collapses at the outermost scales, it is acceptable to:
- omit those shell-local points;
- use them only as labeled limits;
- report all-valid-origin outer-scale points with explicit support diagnostics;
- stop slope fitting below the affected scales;
- report outer-scale turnover behavior without claiming a power law.

==================================================
TASK 4: PORT THE 2D PARALLEL EXECUTION MODEL TO 3D
==================================================

Use the existing 2D production workflow as the architectural template:

```text
generate deterministic displacements once
split displacement ranges across Slurm nodes
load one data product per node
share read-only field arrays across node-local workers
process independent displacement batches
stream additive partial reductions
publish restartable partial outputs
validate and combine partial outputs
```

The 3D implementation must preserve its intentional differences from the 2D
slice path:
- 3D extracted cubes are non-periodic;
- targets use valid-origin construction, not modulo indexing;
- support diagnostics are required per offset and per shell;
- partial outputs must retain block-resolved accumulators for uncertainty;
- cube memory is large enough that worker-private field copies are
  unacceptable.

Implement deterministic work sharding using stable identifiers such as:

```text
cube ID
stencil label
support mode
separation shell
displacement-manifest checksum
displacement shard
spatial block layout
sampling seed
source hash
```

Each partial output must contain enough metadata to prove:
- which offsets it covers;
- which stencil filter it uses;
- which support mode it uses;
- which cube and source arrays it reads;
- which random schedule it uses;
- which block layout it uses;
- whether it is complete;
- whether it can be reduced exactly once.

Use additive reducers for:
- counts;
- sums;
- sums squared;
- exclusion counts;
- eligible-pair counts;
- sampled-pair counts;
- offset-resolved support;
- block-resolved accumulators.

Require reducer checks that reject:
- overlapping displacement shards;
- missing shards;
- mixed source hashes;
- mixed displacement manifests;
- mixed support policies;
- mixed stencil filters;
- mixed seeds where one deterministic result is expected;
- mixed block layouts;
- duplicate reductions;
- incomplete partial outputs.

Benchmark:
- serial execution;
- one node with multiple workers;
- more than one node with deterministic displacement sharding;
- restart after an intentionally omitted shard;
- reduction after partial-output reordering.

Do not assume that all CPU cores improve throughput. Sweep worker counts
because the 3D path has random-access and memory-bandwidth pressure.

==================================================
TASK 5: ADD SPATIAL BLOCK UNCERTAINTY
==================================================

Sampling standard errors from individual point pairs are useful diagnostics
but are not physical uncertainty bars because nearby pairs are spatially
correlated.

Add spatial block-resolved accumulation. Define a documented 3D block layout
for each cube and store additive statistics by block, shell, direction,
field, stencil, measurement, and order.

Evaluate at least two scientifically reasonable block layouts. Report:
- block side lengths;
- block counts;
- edge-block handling;
- whether a pair is assigned by origin block, midpoint block, or another
  explicit rule;
- how non-periodic support interacts with the block assignment;
- blocks contributing to every reported point;
- effective block counts;
- sensitivity to block size.

Use a block bootstrap, block jackknife, or both. Record deterministic
resampling seeds and enough metadata to reproduce the uncertainty bands.

For every reported curve or slope, distinguish:
- point-pair sampling noise;
- displacement-census sensitivity;
- spatial block uncertainty;
- support-mode sensitivity;
- stencil-filter sensitivity;
- fit-interval sensitivity.

Do not collapse these into one unexplained error bar.

==================================================
TASK 6: REPORT LARGE-SCALE BEHAVIOR HONESTLY
==================================================

Plot `S_p(ell)` across the complete measured range, including the small scales,
but emphasize well-supported large-scale behavior. Keep 2-point, 3-point, and
5-point curves visibly and programmatically distinct.

Compute local logarithmic slopes where justified:

```text
alpha(ell) = d log(S_p) / d log(ell)
```

Use a documented local regression or smoothing rule. Show uncertainty bands
and the number of contributing blocks.

For any fitted power-law summary:
- choose candidate intervals after inspecting dissipation-range curvature,
  large-scale turnover, support loss, and uncertainty;
- report the interval explicitly;
- report the number of bins and blocks used;
- report uncertainty;
- compare nearby defensible intervals;
- avoid grid-adjacent dissipative scales unless a documented scientific
  reason requires them;
- avoid outermost scales when finite-support limitations make the fit
  unreliable;
- do not require a perfect line;
- do not report a slope when the curve does not support one.

It is acceptable for Phase 3a to conclude that:
- some directions support robust slopes;
- some directions support only curve-level comparisons;
- some outer scales support turnover diagnostics rather than slope fits;
- some quantities should be omitted from Phase 4.

==================================================
TASK 7: ADD TARGETED EXACT AND CONVERGENCE CONTROLS
==================================================

Do not attempt to enumerate every pair of points in a `640^3` cube. Instead,
use exact calculations where they are affordable and informative.

At minimum:
- exhaustively enumerate all valid origins for selected displacement vectors;
- include representative small, intermediate, and large separations;
- include representative Cartesian and oblique directions;
- include shell-local shared-support and all-valid-origin controls;
- cover 2-point, 3-point, and 5-point filters;
- compare exact and sampled counts, sums, and derived structure functions;
- exercise smaller synthetic or cropped cubes with exhaustive enumeration;
- compare serial and parallel exact-control reductions.

Measure convergence with:
- multiple origin counts per displacement;
- multiple displacement densities per shell;
- multiple deterministic origin seeds;
- multiple block sizes;
- 2-point, 3-point, and 5-point filters;
- `32`, `64`, and `128` separation-bin candidates;
- 2-point `ell_max = 128`, `192`, `256`, and `320`;
- 3-point `ell_max = 80` and `160`;
- 5-point `ell_max = 40` and `80`;
- shell-local shared support versus all valid origins;
- serial versus parallel execution.

Use the retained Phase 3 `ell_max = 128` smoke as a regression anchor. The new
implementation must reproduce it within explicit tolerances before the
science-quality `ell_max = 320` conclusions are accepted.

==================================================
TASK 8: RUN A BOUNDED EXECUTION LADDER
==================================================

Run only the smallest useful job at each step. Register every nontrivial
allocation before submission and refresh the ledger afterward.

## Step A: local correctness

Run:
- unit tests;
- synthetic tests;
- 2-point, 3-point, and 5-point oracle agreement;
- stencil-aware endpoint and support-accounting tests;
- cropped-cube exact controls;
- reducer overlap and omission failures;
- deterministic seed tests;
- block-resampling tests.

## Step B: one-cube serial-versus-parallel control

On one approved cube:
- compare serial and node-local multiprocess results;
- compare selected exact-origin controls against Monte Carlo sampling;
- sweep worker count;
- record runtime, RSS, and output size.

## Step C: one-cube multi-node control

Partition displacement shards across more than one Andes CPU node:
- compare reduced output with Step B;
- test restart after an intentionally missing shard;
- test order-independent reduction;
- record node-hours and scaling.

## Step D: four-cube 2-point convergence campaign

Run the bounded `B`, `u`, `p = 2`, `ell_max = 320` campaign on the four
retained cubes:
- retain the source-bound displacement manifest;
- retain shell-local and all-valid-origin support outputs;
- retain block-resolved accumulators;
- retain uncertainty outputs;
- retain resource accounting;
- retain completion markers;
- publish a source-bound release root.

## Step E: bounded four-cube stencil comparison

After Step D passes, run labeled `B`, `u`, `p = 2` comparison products:

```text
3-point: ell_max = 160
5-point: ell_max = 80
```

For each stencil:
- retain stencil-specific displacement manifests;
- retain stencil-specific support diagnostics;
- retain block-resolved uncertainty;
- compare serial and parallel reductions;
- compare selected sampled offsets against exhaustive-origin controls;
- report runtime, RSS, output size, and node-hours;
- compare curves with the 2-point baseline without conflating the filters.

Do not expand to 21 cubes inside Phase 3a.

==================================================
TASK 9: PROFILE AND FORECAST PHASE 4
==================================================

Measure:
- runtime per cube;
- runtime per support mode;
- runtime per stencil;
- runtime per shell;
- runtime per displacement shard;
- runtime versus worker count;
- runtime versus node count;
- peak RSS;
- shared-memory or memory-map footprint;
- output bytes;
- temporary bytes;
- reduction time;
- restart overhead;
- block-resampling time;
- node-hours;
- cumulative ledger use;
- remaining budget;
- pending exposure.

Provide a linear forecast and identify any known non-linear terms for the
21-region Phase 4 Batch A baseline.

Do not claim CPU scaling unless it was measured. Do not launch Phase 4 merely
because the budget is sufficient.

==================================================
EXPECTED REPOSITORY AND OUTPUT ORGANIZATION
==================================================

Keep Phase 3a implementation, job wrappers, documentation, and report assets
clearly separated from the committed Phase 3 baseline:

```text
scripts/phase3a/
job_scripts/phase3a/
tests/test_phase3a_parallel.py
docs/PHASE3A_PARALLEL_FINITE_DOMAIN.md
PHASE3A_STATUS_UPDATE.md
figures/phase3a_status_update/
```

Reuse and extend shared `sfunctor/` modules only where the functionality is
genuinely common. Preserve the Phase 3 serial path as a regression anchor
rather than silently replacing its behavior.

Use unique restartable Lustre roots under a clearly labeled Phase 3a
directory, for example:

```text
/lustre/orion/ast207/proj-shared/dfielding/Production_plm/phase3a_sampler/
```

Keep provisional, superseded, and retained release roots distinct. Bind every
retained release to source hashes, input identities, displacement manifests,
partial-output inventories, reduction manifests, and completion markers.

==================================================
SUBAGENT REVIEWS
==================================================

Use independent subagents for:

1. Finite-support review:
   Audit shell-local shared support, all-valid-origin behavior, global-core
   regression diagnostics, stencil-specific origin bounds, offset-resolved
   counts, and boundary-bias risks.

2. Parallel-reducer review:
   Audit deterministic sharding, shared-memory or memory-map handling,
   restartability, duplicate-shard rejection, reduction invariance, RSS, and
   measured scaling.

3. Uncertainty review:
   Audit block definitions, resampling, effective block counts, seed
   handling, convergence diagnostics, and distinctions among uncertainty
   sources.

4. Scientific interpretation review:
   Challenge dissipative-scale inclusion, large-scale support, local slopes,
   fitted intervals, stencil labeling, anisotropy claims, and any pressure to
   report a slope where the curve does not support one.

5. Independent adversarial review:
   Try to break the complete implementation with missing shards, duplicate
   shards, mixed manifests, mixed hashes, changed block layouts, sparse bins,
   weak fields, invalid density, axis permutations, and boundary cases.

Reconcile all findings yourself.

==================================================
GO / NO-GO GATE
==================================================

Phase 3a is a GO for Phase 4 only if:
- the two-point `L_sub = 640` path measures through `ell_max = 320`;
- labeled 3-point and 5-point 3D paths are implemented, validated, and
  measured through their approved footprint-bounded limits;
- the retained separation design has at least `32` bins and a documented
  convergence justification;
- non-periodic endpoint handling remains explicit and tested;
- stencil-specific shell-local and all-valid-origin support accounting is exact
  and
  offset-resolved;
- the retained production support policy is selected from measured evidence;
- serial, node-local parallel, and multi-node reduced outputs agree within
  explicit tolerances;
- restart and reduction-order tests pass;
- selected sampled offsets converge against exhaustive-origin controls;
- 2-point, 3-point, and 5-point serial and parallel paths agree with their
  slow-oracle controls;
- origin-count and displacement-density convergence are measured;
- spatial block uncertainty is implemented and reported;
- large-scale curves, local slopes, and any fitted summaries include
  uncertainty and support diagnostics;
- no slope is claimed merely to populate a table;
- runtime, RSS, storage, and projected Phase 4 Batch A node-hours are measured
  and ledger-backed;
- the complete repository test suite passes;
- the four-cube Phase 3a report receives independent review.

Phase 3a is a NO-GO if:
- parallel reduction changes the scientific result;
- restartability or provenance is ambiguous;
- non-periodic support accounting is inconsistent;
- any stencil path silently wraps, conflates filter labels, or disagrees with
  its oracle without explanation;
- sampled results do not converge against targeted exact controls;
- shell-local and all-valid-origin differences remain unexplained relative to
  block uncertainty;
- large-scale claims remain dominated by unsupported outer shells;
- uncertainty is too broad to support the proposed Phase 4 interpretation;
- the measured Phase 4 forecast is unacceptable.

A valid Phase 3a GO may still remove selected slope claims from Phase 4. Curve-
level directional comparisons, ratios, or outer-scale trends may remain
scientifically useful when labeled honestly.

==================================================
DELIVERABLES
==================================================

Provide:
1. source-bound dense separation and displacement manifests for labeled
   2-point, 3-point, and 5-point filters;
2. stencil-aware non-periodic valid-origin geometry;
3. shell-local shared-support implementation;
4. retained all-valid-origin implementation, including compatibility with the
   historical 2-point `all_valid_pairs` label;
5. labeled original global-core regression diagnostic;
6. displacement-distributed multi-node runner;
7. node-local shared-memory or memory-map worker path;
8. restartable additive partial-output schema and strict reducer;
9. block-resolved accumulators and reproducible uncertainty reducer;
10. targeted exhaustive-origin controls for every stencil;
11. sampling and displacement convergence report;
12. separation-bin convergence report;
13. serial, node-local, and multi-node equivalence report;
14. runtime, RSS, storage, node-hour, and ledger forecast by stencil;
15. four-cube `B`, `u`, `p = 2`, 2-point `ell_max = 320` outputs;
16. bounded four-cube 3-point `ell_max = 160` and 5-point `ell_max = 80`
   comparison outputs;
17. large-scale curve, local-slope, support, and uncertainty figures;
18. `PHASE3A_STATUS_UPDATE.md`;
19. `docs/PHASE3A_PARALLEL_FINITE_DOMAIN.md`;
20. explicit GO or NO-GO recommendation for Phase 4;
21. list of files created or modified;
22. unresolved ambiguities.

Do not begin the 21-region Phase 4 pilot until this gate is documented.
