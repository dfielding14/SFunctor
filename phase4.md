# Phase 4: bounded `L_sub = 640` 21-region scientific pilot

Read `phase0.md`, `phase2.md`, `phase3.md`, `phase3a.md`, the completed Phase 2,
Phase 3, and Phase 3a status reports, `PHASE4_LAUNCH_GO_DECISION.md`, and the
approved Phase 4 Batch A planning forecast before starting.

Begin this phase only after Phase 2 has a documented GO decision and
`PHASE4_LAUNCH_GO_DECISION.md` records a reviewed launch GO that supersedes the
pre-adjudication Phase 3a science-configuration NO-GO. Phase 3 remains the
validated smoke-test baseline; do not reinterpret its intentionally
conservative `ell_max = 128` configuration as the Phase 4 science
configuration.

This phase is intentionally limited to the proposed 21-region
`L_sub = 640` pilot from the trusted Phase 1 census. Do not expand to other
environmental scales, additional snapshots, shifted tilings, SGS-derived
channels, or dynamo-derivative channels.

==================================================
OBJECTIVE
==================================================

Run the validated selected-cube extractor and Phase 3a science-quality,
parallel, finite-domain 3D structure-function pipeline in staged batches on
the proposed `L_sub = 640` pilot.

Measure how structure-function statistics vary with the coarse-grained
magnetic environment, especially `dBB`, while distinguishing:
- catalog-supported environmental quantities;
- newly computed full-resolution primitive diagnostics;
- robust results;
- likely results;
- suggestive trends;
- ambiguous or underpowered comparisons.

The output of this phase is a bounded pilot result and a recommendation for
whether a broader Phase 5 campaign is justified.

==================================================
INHERITED HARD STOPS
==================================================

Retain the durable Phase 0, Phase 2, Phase 3, and Phase 3a operational and
data-provenance hard stops:
- use the trusted Phase 1 run read-only;
- reuse the approved Phase 3a sampler, parallel-reduction, and uncertainty
  settings;
- materialize all cubes under a unique Phase 4 extraction root; the current
  adapter freezes the root path into the plan, publishes plan-bound per-cube
  materialization records, and does not support historical-cube reuse;
- use the Phase 4 launch-decision support-policy roles and masking rules;
- do not rerun the Phase 1 census;
- do not use `mhd_sgs` or `mhd_dynamo_ks`;
- do not use periodic wrapping inside extracted cubes;
- use only Andes CPU Slurm allocations for heavy work;
- register every nontrivial allocation before submission;
- write Slurm stdout and stderr under `logs/`;
- do not add Slurm email-notification directives;
- keep at most one debug job queued or running at a time;
- use unique restartable output directories;
- do not launch `L_sub = 1280` work;
- stop after the bounded 21-region `L_sub = 640` pilot.

The Phase 3 and Phase 3a prohibitions on launching the 21-region pilot were
phase-local scope limits. `PHASE4_LAUNCH_GO_DECISION.md` supersedes those local
limits through a reviewed policy adjudication only for the bounded Batch A
pilot below.

==================================================
TASK 1: FREEZE THE PILOT CONFIGURATION
==================================================

Use the Phase 1 pilot table:

    $TRUSTED_RUN/analysis/pilot_sample.csv

Use `scripts/phase4/run_phase4_extraction.py` through
`job_scripts/phase4/run_phase4_extract_andes.sh` to freeze and materialize the
exact 21-cube pilot. Use `scripts/phase4/run_phase4_batch_a_sampler.py` through
`job_scripts/phase4/run_phase4_batch_a_sampler_andes.sh` for the bounded Batch
A sampler plan. Do not bypass either Phase 4-specific guard by calling the
historical four-cube Phase 2 or Phase 3a wrappers directly.

Confirm that it contains the expected 21 `L_sub = 640` regions and preserve:
- cube ID;
- half-open global bounds;
- required rank IDs;
- `dBB`;
- `B_mean`;
- `deltaB`;
- `B_rms`;
- density summaries;
- conserved-momentum summaries;
- representative, matched, or outlier role;
- Phase 1 validity flags;
- estimated extraction size.

Do not silently alter membership. If a cube must be removed for a documented
data-quality reason, report the exclusion and its effect on scientific power.

Before calculating structure functions for each newly materialized cube:
- run the supported extraction-versus-`cbin` comparisons defined in
  `phase2.md`;
- require exact shape and cell coverage;
- require a complete provenance manifest;
- reject unexplained mismatches;
- record any precision-limited standardized moments.

Freeze and record:
- source commit;
- extraction version;
- sampler version;
- random seeds;
- field variants;
- `p` values;
- approved stencil matrix and explicit stencil labels;
- separation bins;
- realized displacement counts per bin;
- angular wedges;
- stencil-specific `ell_max` values;
- retained finite-support policies and their distinct reporting roles;
- shell-local shared-support geometry where used;
- all-valid-origin sensitivity policy, including the historical 2-point
  `all_valid_pairs` label;
- chunk size;
- pair batch size;
- Slurm-node and node-local worker layout;
- deterministic displacement-shard manifest;
- partial-output and reduction schema;
- uncertainty method;
- spatial block layout;
- deterministic resampling seeds;
- fit intervals where a slope or fitted exponent is actually claimed;
- output schema.

Require the approved Phase 3a baseline:

```text
L_sub = 640
at least 32 separation bins
2-point: ell_max = 320
3-point: ell_max <= 160
5-point: ell_max <= 80
explicit stencil labels
explicit non-periodic support accounting
block-resampled uncertainty
restartable displacement-distributed reduction
```

The approved Phase 4 launch policy is:

```text
primary curve-level product:
    all_valid_origins

required directional robustness overlay:
    shell_local

shell_local science-facing curve overlay:
    require eligible-origin fraction >= 5%

directional slope-table candidate:
    require shell_local eligible-origin fraction >= 10%
    require the existing contributing-block, Kish-effective-block, and
    bootstrap-validity gates
    require a separately reviewed fit interval

nested_core:
    labeled regression diagnostic only

reporting mode:
    curve-first
    no mandatory directional slope table
```

Preserve raw weak-support values as visibly flagged diagnostics. Do not treat
the `5%` curve-overlay threshold or `10%` slope-candidate threshold as a
correction factor. A slope-table candidate remains a candidate until its
support-policy sensitivity, local curvature, nearby-window stability, and
uncertainty have been reviewed.

==================================================
TASK 2: RUN STAGED BATCHES
==================================================

Do not launch the entire variable-by-order matrix immediately.

## Batch A: baseline integrity and cost

Run all 21 cubes with:

```text
q = B, u
p = 2
stencil = 2-point
primary curve-level policy = all_valid_origins
directional robustness overlay = shell_local
```

The measured planning inputs are:

```text
21-cube extraction conservative wrapper proxy:
    1.7558 node-hours

21-cube 2-point all_valid_origins estimator allocation-share proxy:
    0.684 node-hours

21-cube 2-point shell_local estimator allocation-share proxy:
    0.727 node-hours

21-cube two-policy 2-point estimator allocation-share proxy:
    1.411 node-hours
```

These are bounded planning proxies, not guaranteed end-to-end costs. Extraction
restart verification, planning, reduction, bootstrap uncertainty, final
verification, scheduler behavior, and cache state add overhead.

Require:
- output-integrity checks;
- completion-marker checks;
- count and exclusion summaries;
- per-cube runtime;
- peak RSS;
- output size;
- ledger refresh;
- comparison against the Phase 3a forecast.

Stop and investigate if the measured cost or scientific coverage differs
materially from the forecast.

## Batch A2: bounded stencil comparison

Stop after Batch A reporting. Only after a separate post-Batch-A review and
explicit human approval, run the planned labeled comparison matrix:

```text
q = B, u
p = 2
3-point: ell_max <= 160
5-point: ell_max <= 80
```

Begin with the retained representative low-, intermediate-, high-`dBB`, and
weak-mean-field cubes. Expand the 3-point product to all 21 cubes only if the
representative-cube review finds an informative and stable labeled comparison.
Keep the 5-point product bounded to representative cubes during the initial
Phase 4 launch. Any 5-point expansion requires a separate reviewed decision.

Keep the filters distinct. Do not present 3-point or 5-point values as
higher-accuracy replacements for the 2-point statistic.

## Batch B: order dependence

Only after Batch A, the bounded Batch A2 review, and the all-21-cube labeled
3-point extension review pass, begin a representative-cube baseline 2-point
probe with:

    q = B, u
    p = 1, 2, 3, 4, 5, 6

Require an explicit human-reviewed Batch B launch decision before starting
this probe. Inspect accepted counts, effective block counts, bootstrap
validity, support-policy sensitivity, and especially the $p = 6$ tails. Expand
the same labeled 2-point matrix to all 21 cubes only after a second explicit
reviewed decision.

Batch B retains the equal-origin, volume-weighted `B,u` estimators. Do not
introduce density weighting in this batch; `rho_0` is not applicable until the
separately reviewed Batch C variable variants.

Measure higher-order exponents only where the accepted counts and fit
intervals are defensible.

### Batch B representative checkpoint: HOLD before all-21 expansion

The approved eight-cube representative probe completed on 2026-06-01. Its
validated 2-point products cover `B,u`, both support policies, and
`p = 1,2,3,4,5,6`. The retained review is documented in
`PHASE4_BATCH_B_REPRESENTATIVE_STATUS_UPDATE.md`.

Do not expand Batch B to all 21 cubes yet. The representative review found
supported large-scale high-order tail sensitivity between
`all_valid_origins` and `shell_local`. Before reconsidering expansion, run a
bounded matched-origin comparison and a bounded origin-seed sensitivity sweep
on the retained priority cases. Decompose common interior origins,
intrinsic-valid exterior origins, and their population-weighted recomposition.
Repeat selected $p = 2,4,6$ cases across deterministic seeds at increased
sample depth, and inspect cumulative $p = 6$ contributions or top
contributors. Review the complete $p = 6$ tails and sixth-root amplitude
factors. Keep the 5-point product bounded.

## Batch C: compressible-MHD variable comparisons

Only after Batches A and B pass and after reviewing incremental cost and
scientific value, consider:

    v_A
    v_A_ref
    z_plus
    z_minus
    z_plus_ref
    z_minus_ref

Keep pointwise-density and reference-density variants labeled separately.
Record the exact `rho_0` convention for each reference-density calculation.

Use intermediate go/no-go reviews between batches. It is acceptable to defer
some Batch C variants if they do not justify their cost or if density
pathologies make interpretation unreliable.

==================================================
TASK 3: REPORT ENVIRONMENTAL QUANTITIES HONESTLY
==================================================

For each selected cube, report catalog-supported Phase 1 quantities:

    L_sub
    global coordinate bounds
    dBB
    B_mean
    deltaB
    B_rms
    bounded magnetic complements
    density summaries
    conserved-momentum summaries
    valid standardized moments
    representative, matched, or outlier role
    extraction size
    runtime
    accepted and excluded pair counts

Also compute useful full-resolution primitive diagnostics where definitions are
explicit, including where justified:

    volume-weighted velocity statistics
    delta u
    sonic Mach number M_s
    Alfvén Mach number variants M_A
    pressure summaries
    kinetic-energy summaries
    magnetic-to-kinetic energy ratios

Label these as newly computed full-resolution diagnostics. Do not claim that
the Phase 1 `cbin` census independently validates them. Document definitions,
units, density weighting, sound-speed convention, and any approximation.

==================================================
TASK 4: ANALYZE DBB DEPENDENCE
==================================================

The primary environmental question is how structure-function outputs depend on
`dBB` at:

    L_sub = 640

Use the Phase 1 scale-specific `dBB` selector quantiles as the primary regime
definition. Fixed language such as:

    dBB << 1
    dBB ~ 1
    dBB >> 1

may be retained only as a secondary descriptive view.

Always report `B_mean`, `deltaB`, and the bounded magnetic complements beside
`dBB`, because extreme `dBB` can be denominator-driven.

At minimum, examine:

    S_p(ell) curves with block-resampled uncertainty
    local logarithmic slopes alpha(ell) where diagnostically supported
    structure-function slopes where an approved interval exists
    higher-order exponents zeta_p where an approved interval exists
    ell_parallel
    xi
    lambda
    ell_parallel / lambda
    xi / lambda
    pair-scale versus subvolume-scale magnetic-field conditioning
    accepted-pair and exclusion-count dependence on environment

Use:
- matched comparisons;
- conditional medians;
- quantile bands;
- support-versus-separation diagnostics;
- uncertainty-versus-separation diagnostics;
- rank correlations;
- simple regression or partial-correlation tools where sample size permits;
- careful visual inspection.

==================================================
TASK 5: DISTINGUISH CONTROLS FROM EXPLORATORY COVARIATES
==================================================

The original Phase 1 pilot was not matched on `M_s`, `M_A`, `delta u`,
pressure, or kinetic-energy ratios.

For matched comparisons and pre-extraction interpretation, use only supported
Phase 1 catalog properties:
- density summaries;
- `B_mean`;
- `deltaB`;
- `B_rms`;
- conserved-momentum widths;
- valid higher moments where numerically resolved.

Treat the following as post-extraction exploratory covariates:
- `M_s`;
- `M_A` variants;
- `delta u`;
- pressure;
- kinetic-energy summaries;
- magnetic-to-kinetic energy ratios;
- additional primitive-only diagnostics.

Do not claim that retrospective correlations in a 21-cube pilot eliminate
confounding. Report residual confounding and limited sample size explicitly.

==================================================
TASK 6: CHECK ROBUSTNESS
==================================================

For representative low-, intermediate-, high-`dBB`, and weak-mean-field cubes:
- compare the primary `all_valid_origins` curve products and `shell_local`
  directional robustness overlays;
- retain the original global nested-core result only as a labeled regression
  diagnostic where it remains meaningful;
- vary `ell_max`;
- compare approved 2-point, 3-point, and 5-point stencil products;
- vary angular wedge widths;
- vary separation-bin widths;
- vary sampled-pair count;
- compare matched-origin policy diagnostics or vary deterministic origin seeds
  before interpreting policy differences physically;
- vary displacement density;
- vary spatial block layout;
- inspect accepted and excluded counts;
- inspect fit-interval sensitivity;
- compare pair-scale and subvolume-scale field conditioning;
- inspect representative extracted slices and structure-function curves.

Require enough accepted pairs and contributing blocks to support every
reported directional claim. Do not fit through dissipative, sparse,
support-limited, or unstable bins merely to populate a table. It is acceptable
to report curve-level or outer-scale comparisons without fitted exponents.

==================================================
SUBAGENT REVIEWS
==================================================

Use independent subagents for:

1. Batch-integrity review:
   Check restartability, manifests, completion markers, output schema, and
   count diagnostics.

2. Scientific interpretation review:
   Look for overclaimed `dBB` trends, residual confounding, unstable slopes,
   denominator-driven outliers, and misleading aggregate plots.

3. Variable-convention review:
   Verify density weighting, `rho_0`, code-unit conventions, Mach-number
   definitions, and labels for pointwise versus reference-density variables.

4. Performance review:
   Compare measured node-hours, RSS, storage, worker scaling, and reduction
   cost against the Phase 3a forecast before each batch expansion.

5. Independent adversarial review:
   Challenge the final report, figures, and campaign recommendation.

Reconcile all findings yourself.

==================================================
GO / NO-GO GATE
==================================================

Phase 4 is complete only after the bounded `L_sub = 640` pilot has been
reported. It does not automatically authorize Phase 5.

Recommend Phase 5 only if:
- all 21 retained cubes have supported extraction-versus-`cbin` comparison
  results and validated manifests or documented exclusions;
- output integrity checks pass;
- stencil products remain explicitly labeled and use approved
  stencil-specific support limits;
- accepted and excluded pair counts support the reported comparisons;
- block-resampled uncertainty and effective block counts support the reported
  comparisons;
- the retained support policy and all-valid-origin sensitivity results do not
  reveal an unexplained scientific inconsistency;
- displacement-shard reduction remains restartable and partition invariant;
- runtime, RSS, storage, and ledger totals are documented;
- any fitted slopes use justified large-scale intervals and report
  uncertainty; it is acceptable to withhold fitted slopes;
- robustness comparisons do not reveal unexplained edge bias;
- catalog-supported and primitive-only covariates are clearly separated;
- residual confounding and sample-size limits are stated honestly;
- the scientific value of expansion is explicit;
- a staged Phase 5 cost forecast is provided.

Stop after Phase 4 and request human approval before expanding.

==================================================
DELIVERABLES
==================================================

Provide:
1. frozen 21-region configuration;
2. validated extraction manifests for retained cubes;
3. staged Batch A, A2, B, and any approved Batch C outputs;
4. per-cube environmental and runtime table;
5. pair-count and exclusion diagnostics;
6. support-versus-scale and contributing-block diagnostics;
7. structure-function curves and supported local-slope diagnostic figures with
   uncertainty;
8. conditional slope and aspect-ratio tables only where uncertainty, support,
   and reviewed fit intervals justify them;
9. `dBB` trend figures with magnetic complements;
10. matched-comparison figures;
11. robustness figures;
12. labeled 2-point, 3-point, and 5-point stencil-comparison figures;
13. runtime, memory, storage, parallel-scaling, and ledger report;
14. Phase 4 status report;
15. explicit recommendation for or against Phase 5;
16. proposed Phase 5 scope and cost if expansion is justified;
17. list of files created or modified;
18. unresolved ambiguities.
