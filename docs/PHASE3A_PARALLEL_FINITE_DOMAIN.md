# Phase 3a parallel finite-domain 3-D estimator

Phase 3a extends the bounded Phase 3 serial smoke path into a science-quality
validation path for the four retained `L_sub = 640` cubes. It does not launch
the 21-region Phase 4 campaign.

## Scope

The retained baseline remains:

```text
q = B, u
p = 2
L_sub = 640
```

The labeled stencil limits are:

| Stencil | Increment | Maximum separation |
|---|---|---:|
| `2-point` | $\mathbf{q}(\mathbf{x}+\mathbf{r})-\mathbf{q}(\mathbf{x})$ | $320$ |
| `3-point` | $[\mathbf{q}(\mathbf{x}+\mathbf{r})-2\mathbf{q}(\mathbf{x})+\mathbf{q}(\mathbf{x}-\mathbf{r})]/\sqrt{3}$ | $160$ |
| `5-point` | $[\mathbf{q}(\mathbf{x}-2\mathbf{r})-4\mathbf{q}(\mathbf{x}-\mathbf{r})+6\mathbf{q}(\mathbf{x})-4\mathbf{q}(\mathbf{x}+\mathbf{r})+\mathbf{q}(\mathbf{x}+2\mathbf{r})]/\sqrt{35}$ | $80$ |

The filters are distinct scale filters. Their outputs must remain labeled.

## Non-Periodic Geometry

Arrays remain in AthenaK KJI storage order:

```text
array[k, j, i] == field[x3, x2, x1]
```

Displacements remain Cartesian IJK tuples:

```text
(di, dj, dk)
```

For one cube-axis size $N$, displacement component $d$, and stencil
multipliers $m$, the valid half-open origin interval is:

$$
\left[
\max_m(0,-md),
\min_m(N,N-md)
\right).
$$

No path uses modulo indexing. The implementation asserts every stencil point
before indexing.

## Local Magnetic Field

Directional conditioning follows the existing 2-D filter conventions:

$$
\mathbf{B}_{\mathrm{loc},2}
=
\frac{
\mathbf{B}(\mathbf{x})
+
\mathbf{B}(\mathbf{x}+\mathbf{r})
}{2},
$$

$$
\mathbf{B}_{\mathrm{loc},3}
=
\frac{
\mathbf{B}(\mathbf{x}-\mathbf{r})
+
\mathbf{B}(\mathbf{x})
+
\mathbf{B}(\mathbf{x}+\mathbf{r})
}{3},
$$

$$
\mathbf{B}_{\mathrm{loc},5}
=
\frac{
\mathbf{B}(\mathbf{x}-2\mathbf{r})
+4\mathbf{B}(\mathbf{x}-\mathbf{r})
+6\mathbf{B}(\mathbf{x})
+4\mathbf{B}(\mathbf{x}+\mathbf{r})
+\mathbf{B}(\mathbf{x}+2\mathbf{r})
}{16}.
$$

## Support Modes

Phase 3a retains three labeled policies:

| Mode | Meaning | Role |
|---|---|---|
| `shell_local` | Intersect valid-origin boxes for every offset in one shell and stencil. | Fair directional comparison within each shell. |
| `all_valid_origins` | Use the complete non-periodic valid-origin box for each offset. | Maximum-volume comparison with explicit support diagnostics. |
| `nested_core` | Intersect all retained offsets into one global core. | Regression diagnostic where non-empty. |

The historical label `all_valid_pairs` is accepted only for compatible
2-point products. New wider-filter products use `all_valid_origins`.

The result schema preserves Phase 3 compatibility arrays such as
`eligible_pairs`, but Phase 3a also separates:

```text
intrinsic_eligible_origins
boundary_excluded_origins
support_policy_excluded_origins
```

This distinguishes physical cube-boundary loss from a deliberate shared-box
support restriction.

## Spatial Blocks

Accepted moments, sampled origins, eligible origins, and exclusions are
stored by spatial block. Tuple assignment is:

```text
2-point: midpoint x + r / 2
3-point: center x
5-point: center x
```

The midpoint rule makes equivalent signed 2-point representations block
stable. Edge blocks are truncated naturally by the cube bounds.

`sfunctor.analysis.phase3a` implements:

- deterministic fixed-layout spatial block bootstrap, including empty blocks
  in the common layout so covariance across curves is preserved;
- delete-one spatial block jackknife;
- contributing-block counts and Kish effective accepted-block counts;
- centered local regressions for:

$$
\alpha(\ell)
=
\frac{
\mathrm{d}\log S_p
}{
\mathrm{d}\log \ell
}.
$$

The Kish effective accepted-block count is:

$$
N_{\mathrm{eff}}
=
\frac{
\left(\sum_b n_b\right)^2
}{
\sum_b n_b^2
}.
$$

Uncertainty products retain moment bands and local-slope bands. Here $n_b$ is
the accepted-sample count in block $b$. The effective count makes visible when
a nominal block layout is dominated by a smaller number of spatial regions.

Pair-sampling standard errors remain separately labeled diagnostics. They are
not physical uncertainty bars.

## Dense Separation Design

`sfunctor.core.phase3a.dense_displacement_manifest(...)` generates
deterministic signed integer 3-D offsets with:

- hybrid integer and geometric separation centers;
- strict post-rounding scale limits;
- signed closure;
- stable offset IDs;
- per-bin realized counts;
- separately reported zero-offset removals, post-rounding duplicate removals,
  and strict scale-limit exclusions;
- a compatibility aggregate for zero-or-duplicate removals;
- per-bin directional occupancy;
- checksummed NPZ and JSON manifests.

The release design uses `64` separation bins and `24` requested directions per
bin. The bounded convergence action also measures `32` and `128` bins plus a
second direction density.

## Parallel Model

Phase 3a follows the validated 2-D architecture while preserving its 3-D
differences:

1. freeze one full displacement manifest per stencil;
2. assign fixed offset intervals to restartable shards independent of Slurm
   node count;
3. fully rehash the Phase 2 array bytes before `work`, `reduce`, and `verify`
   so an in-place input change cannot inherit a planned identity;
4. load Phase 2 `.npy` arrays with `mmap_mode="r"`;
5. stack only the required `B` and `u` vectors once in the node parent;
6. fork node-local workers so read-only pages are shared copy-on-write;
7. publish pickle-free NPZ partials with atomic directory renames;
8. reduce only the exact inventory in `manifests/shards.json`;
9. reject missing, overlapping, mixed-source, mixed-support, mixed-stencil,
   mixed-seed, and mixed-block-layout products.

Shell-local geometry is always derived from the frozen full census, not from a
worker shard. Every partial stores the frozen support-census checksum.

Every publication shard also stores staging logical and allocated byte counts.
Task-local resource records retain Slurm job and task IDs, worker count, shard
wall times, estimator elapsed sums, serialization and verification times,
reuse times, mapped-input bytes, parent-stacked-vector bytes, and
parent-process RSS. These runner values are distinct from Slurm step-level
high-water RSS.

Reduction reuse is conservative: the reduction manifest is parsed and rebound
to the current campaign, current shard-marker checksums, source identity, and
uncertainty metadata before an existing reduction is accepted. Multi-node
control tasks apply the same principle to task-local partials: each task marker
binds its source hash, Phase 2 input identity, support census, sampling
configuration, and expected offset inventory before reuse or aggregation.

The default Andes setting is one worker per node. Preliminary controls found
the path to be memory-bound, so higher worker counts are explicit benchmark
overrides rather than the publication default. This is a scheduling default,
not a claim that one worker is universally optimal on other machines.

## CLI And Slurm Wrapper

The CLI is:

```bash
python scripts/phase3a/run_phase3a_sampler.py plan --output-root "$OUTPUT_ROOT"
python scripts/phase3a/run_phase3a_sampler.py work --output-root "$OUTPUT_ROOT" --workers 1
python scripts/phase3a/run_phase3a_sampler.py reduce --output-root "$OUTPUT_ROOT"
python scripts/phase3a/run_phase3a_sampler.py verify --output-root "$OUTPUT_ROOT"
python scripts/phase3a/run_phase3a_sampler.py summarize --output-root "$OUTPUT_ROOT"
```

Bounded validation actions are:

```bash
python scripts/phase3a/run_phase3a_sampler.py controls --output-root "$OUTPUT_ROOT" --workers 1
python scripts/phase3a/run_phase3a_sampler.py multinode-work --output-root "$OUTPUT_ROOT" --workers 1
python scripts/phase3a/run_phase3a_sampler.py multinode-reduce --output-root "$OUTPUT_ROOT" --workers 1 --task-count 2
python scripts/phase3a/run_phase3a_sampler.py convergence --output-root "$OUTPUT_ROOT" --workers 1
```

The controls action benchmarks explicit worker-count overrides. Publication
runs use the conservative default unless retained benchmark evidence supports
an override.

The Andes wrapper is:

```text
job_scripts/phase3a/run_phase3a_sampler_andes.sh
```

It writes Slurm stdout and stderr under `logs/`, writes detailed action logs
under the explicit allocation `RUN_DIR`, sets threaded numerical libraries to
one thread per process, contains no email directives, and does not reference
the broken SGS products.

The wrapper uses a hidden sibling action lock to prevent concurrent
publication into one output root without making a fresh `plan` root non-empty.
A stale lock is removed only after an `squeue` snapshot proves that its numeric
owner allocation is inactive. The lock must not be deleted merely because a
previous run appears interrupted. Each wrapper exit also archives a
step-resolved `sacct` snapshot under the explicit allocation `RUN_DIR`; use its
Python-step `MaxRSS` as the scheduling high-water measurement rather than the
wrapper launcher RSS.

The bounded convergence action binds its representative Phase 2 cube identity
and verifies every scenario artifact checksum. Bin-count and
direction-density comparisons use their complete generated displacement
censuses. Other bounded sensitivity scenarios use a documented
shell-stratified subset. Wider-stencil `nested_core` diagnostics remain
labeled comparisons where their retained core is non-empty.

## Publication Layout

```text
OUTPUT_ROOT/
  PLAN_COMPLETE.json
  PHASE3A_RELEASE_COMPLETE.json
  manifests/
    campaign.json
    shards.json
    displacements/
  shards/
  reductions/
  attempts/
  work_resource_records/
  phase3a_summary.json
```

`PHASE3A_RELEASE_COMPLETE.json` means that the bounded four-cube release was
aggregated and verified. It is deliberately not a Phase 4 GO marker. The
report-level science gate remains separate.

Bounded diagnostic actions follow the same distinction. Their completion
markers mean that the requested diagnostic ran and published internally
consistent evidence. Scientific acceptance is a separate, explicit evaluation
against the Phase 3a gate; operational completion alone must not be described
as a scientific pass.

Interrupted attempts remain outside the authoritative `shards/` and
`reductions/` trees. Published shard IDs remain independent of allocation
node count, so work can be resumed with a different Andes allocation size.

## Gate

Phase 3a is a GO for Phase 4 only after:

- the local, node-local, and multi-node equivalence controls pass;
- the four-cube release is complete and source-bound;
- separation, direction-density, origin-depth, block-layout, support-mode,
  and stencil sensitivities are reported;
- spatial block uncertainty is reported;
- large-scale claims are restricted to supported scales;
- independent reviews are reconciled;
- the full repository suite passes.

Budget sufficiency alone is not a GO criterion.
