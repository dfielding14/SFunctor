# Phase 4: bounded `L_sub = 640` 21-region scientific pilot

Read `phase0.md`, `phase2.md`, `phase3.md`, `phase3a.md`, the completed Phase 2,
Phase 3, and Phase 3a status reports, and the approved Phase 3a benchmark
forecast before starting.

Begin this phase only after Phase 2 has a documented GO decision and Phase 3a
has documented a GO decision that resolves the Phase 3 science-configuration
NO-GO. Phase 3 remains the validated smoke-test baseline; do not reinterpret
its intentionally conservative `ell_max = 128` configuration as the Phase 4
science configuration.

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
- reuse validated Phase 2 extraction outputs and the approved Phase 3a
  sampler, support, parallel-reduction, and uncertainty settings;
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
phase-local scope limits. This Phase 4 document explicitly supersedes those
local limits only after a documented Phase 3a GO and only for the bounded
pilot below.

==================================================
TASK 1: FREEZE THE PILOT CONFIGURATION
==================================================

Use the Phase 1 pilot table:

    $TRUSTED_RUN/analysis/pilot_sample.csv

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
- retained finite-support policy;
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
- fit intervals;
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

==================================================
TASK 2: RUN STAGED BATCHES
==================================================

Do not launch the entire variable-by-order matrix immediately.

## Batch A: baseline integrity and cost

Run all 21 cubes with:

    q = B, u
    p = 2
    stencil = 2-point

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

Only after Batch A passes, run the approved labeled comparison matrix:

```text
q = B, u
p = 2
3-point: ell_max <= 160
5-point: ell_max <= 80
```

Begin with representative low-, intermediate-, and high-`dBB` cubes. Expand
to all 21 cubes only if the Phase 3a evidence, measured cost, support
diagnostics, and scientific value justify that expansion.

Keep the filters distinct. Do not present 3-point or 5-point values as
higher-accuracy replacements for the 2-point statistic.

## Batch B: order dependence

Only after Batch A and the bounded Batch A2 review pass, expand the baseline
2-point variables to:

    q = B, u
    p = 1, 2, 3, 4

Measure higher-order exponents only where the accepted counts and fit intervals
are defensible.

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
    local logarithmic slopes alpha(ell)
    structure-function slopes
    higher-order exponents zeta_p
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

For representative low-, intermediate-, and high-`dBB` cubes:
- compare the retained Phase 3a primary support policy and all-valid-origin
  results;
- retain the original global nested-core result only as a labeled regression
  diagnostic where it remains meaningful;
- vary `ell_max`;
- compare approved 2-point, 3-point, and 5-point stencil products;
- vary angular wedge widths;
- vary separation-bin widths;
- vary sampled-pair count;
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
- fitted slopes use justified large-scale intervals and report uncertainty;
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
7. structure-function curves and local-slope figures with uncertainty;
8. slope and aspect-ratio tables with uncertainty and fit intervals;
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
