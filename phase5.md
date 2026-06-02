# Phase 5: human-approved cross-scale and expanded production campaign

Read `phase0.md`, `phase2.md`, `phase3.md`, `phase3a.md`, `phase4.md`, all
completed phase status reports, and the Phase 4 campaign recommendation before
starting.

The final Phase 4 closeout report is:

    PHASE4_STATUS_UPDATE.md

Treat `figures/phase4_completion_supplement_v2/` as the final Phase 4
completion-supplement package. Preserve
`figures/phase4_completion_supplement/` as a superseded visual artifact and
audit checkpoint; do not overwrite it.

Do not execute this phase without explicit human approval of a bounded campaign
plan and its maximum node-hour exposure.

This phase is a decision-driven expansion stage. It is not permission to run
every scale, snapshot, field convention, diagnostic, or subvolume
automatically.

==================================================
OBJECTIVE
==================================================

Use the validated extractor and finite-domain 3D structure-function pipeline
to determine whether the Phase 4 `L_sub = 640` findings persist across selected
environmental averaging scales, broader samples, shifted tilings, or additional
snapshots.

The conceptual target is:

    S_p(ell | dBB(L_sub), other environmental properties at L_sub)

Keep:

    L_sub
    ell
    p

distinct everywhere.

The output is a production-grade cross-scale assessment or a documented
decision that further expansion is not justified.

==================================================
INHERITED HARD STOPS
==================================================

Retain all durable prior operational and data-provenance hard stops:
- use trusted Phase 1 products read-only;
- reuse validated extractor and Phase 3a sampler, finite-support,
  parallel-reduction, and uncertainty versions;
- do not use periodic wrapping inside extracted cubes;
- do not use `mhd_sgs` or `mhd_dynamo_ks`;
- do not restore excluded channels without a separate direct-validation
  project and explicit approval;
- use only Andes CPU Slurm allocations for heavy work;
- register every nontrivial allocation before submission;
- write Slurm stdout and stderr under `logs/`;
- do not add Slurm email-notification directives;
- require every new wrapper to reject an existing `RUN_DIR` before writing
  allocation logs or artifacts;
- keep at most one debug job queued or running at a time;
- use unique restartable output directories;
- stop before any unapproved campaign expansion;
- obtain separate explicit human approval before any expensive
  `L_sub = 1280` extraction.

Prior prohibitions on cross-scale extraction, additional snapshots, and
shifted tilings were phase-local scope limits. This Phase 5 document supersedes
those local limits only for extensions that receive explicit human approval
under a bounded plan. The durable prohibition on unvalidated or in-place
census regeneration remains in force.

==================================================
TASK 1: PRESENT A BOUNDED CAMPAIGN PLAN
==================================================

Before submitting production jobs, propose a staged plan that states:
- scientific question;
- environmental scales;
- number of cubes per scale and regime;
- selection strategy;
- matched-comparison strategy;
- strategy for testing whether `dBB` trends survive conditioning on supported
  catalog controls and separately labeled primitive-only exploratory
  covariates;
- field variants;
- `p` values;
- sampler configuration;
- approved stencil matrix and explicit stencil labels;
- finite-support policy;
- separation-bin design;
- `ell_max` policy;
- spatial block layout and resampling method;
- displacement-shard and partial-reduction design;
- random seeds;
- estimated bytes read;
- estimated output bytes;
- estimated temporary storage;
- expected node-hours;
- maximum node-hours;
- cumulative ledger use;
- remaining budget;
- pending exposure;
- restart strategy;
- stopping rules.

Wait for explicit human approval of that plan.

Before recommending or launching that bounded plan, require the retained
Phase 4 all-21 Batch B 2-point `p = 2` slice to reproduce the earlier Batch A
baseline exactly for both support policies. Treat an unexplained mismatch as
a stop condition. Start any approved cross-scale campaign with a 2-point
`p = 2` smoke product, then add higher orders only after reviewing support,
block uncertainty, and high-order tails. Keep 3-point products as separately
labeled comparisons and require a separate reviewed decision before any
5-point expansion.

==================================================
TASK 2: PRIORITIZE SCALES CONSERVATIVELY
==================================================

The Phase 1 environmental catalogs exist at:

    L_sub / Delta x = 80, 160, 320, 640, 1280

Do not assume that full-resolution extraction is required at every scale.

Use Phase 4 results and measured scaling to prioritize an informative subset.
A reasonable staged order is:

    320
    160
    80
    1280 only after separate approval

Retain `640` as the validated baseline.

For each proposed scale:
- define scale-specific `dBB` quantile regimes;
- select representative, matched, and outlier cubes explicitly;
- record physical overlap with existing cubes where relevant;
- define scale-appropriate stencil-specific `ell_max` values, normally
  testing the 2-point filter through `L_sub / 2`, the 3-point filter through
  `L_sub / 4`, and the 5-point filter through `L_sub / 8`;
- retain at least `32` separation bins unless a measured resolution study
  justifies another choice;
- verify non-periodic finite-support and block-uncertainty behavior at that
  scale;
- estimate incremental scientific value;
- estimate incremental cost;
- obtain approval before launch.

Fixed `dBB` thresholds may be reported as a secondary view but must not replace
scale-specific selector definitions.

==================================================
TASK 3: ASSESS SCALE DEPENDENCE
==================================================

Assess:
- how environmental classification changes with `L_sub`;
- whether the same physical region changes `dBB` regime across scales;
- whether structure-function correlations strengthen or weaken with `L_sub`;
- whether a particular environmental scale is especially informative;
- whether apparent trends are numerator-driven, denominator-driven, or both;
- whether matched comparisons reduce apparent confounding;
- whether directional coverage or finite-domain bias changes with cube size;
- whether block-resampled uncertainty changes with cube size;
- whether large-scale local slopes or outer-scale turnover behavior change
  with cube size.

Always report:

    dBB
    B_mean
    deltaB
    B_rms
    bounded magnetic complements

where:

    bounded magnetic complements =
        B_mean^2 / <B^2>_V
        deltaB^2 / <B^2>_V

beside one another.

Do not collapse results from different `L_sub` values into one environmental
label.

==================================================
TASK 4: OPTIONAL EXTENSIONS REQUIRE SEPARATE DECISIONS
==================================================

Treat each of the following as an optional extension with its own scientific
justification, cost forecast, and explicit approval:

1. Shifted tilings:
   Compare origin-aligned and shifted environmental catalogs to assess tiling
   sensitivity. Generate any new primary-only catalogs in a fresh output tree.
   Validate them with the Phase 1 primary-only reconstruction and
   second-pass-verification gates before using them for extraction selection.

2. Additional snapshots:
   Evaluate time variability only if snapshot-specific Phase 1 conclusions are
   insufficient. Generate any new primary-only catalogs in a fresh output
   tree. Require snapshot-identity checks, direct primitive reconstruction
   checks, second-pass verification, and an approved selection table before
   extracting full-resolution cubes.

3. Expanded sample sizes:
   Add cubes only where statistical power or regime coverage requires them.

4. Broader field or order matrices:
   Expand `q` variants or `p` values only where Phase 4 results justify the
   additional cost.

5. Deferred diagnostics:
   Consider increment-distribution flatness, intermittency, current-sheet
   statistics, dissipation proxies, or specialized alignment diagnostics only
   after the primary campaign is stable.

6. Excluded channels:
   Restoration of SGS-derived or dynamo-derivative quantities is a separate
   validation project. Do not include it in routine expansion.

==================================================
TASK 5: VALIDATE EVERY EXPANSION STEP
==================================================

For each approved batch:
- run the smallest useful smoke case first;
- compare measured runtime and RSS against forecast;
- refresh the ledger;
- inspect output integrity;
- inspect accepted and excluded pair counts;
- inspect contributing-block counts and block-resampled uncertainty;
- inspect scale-range, local-slope, and fit-interval sensitivity;
- inspect explicitly labeled 2-point, 3-point, and 5-point comparisons where
  included in the approved batch;
- compare the primary `all_valid_origins` curves and the retained
  `shell_local` robustness overlays;
- verify deterministic displacement-shard reduction and restartability;
- inspect representative cubes and plots;
- stop if scientific value or numerical quality degrades.

Do not launch large arrays of jobs merely because the remaining node-hour
budget is substantial.

==================================================
SUBAGENT REVIEWS
==================================================

Use independent subagents for:

1. Campaign-design review:
   Challenge scale priorities, sample sizes, stopping rules, and expected
   scientific value.

2. Compute-budget review:
   Audit ledger totals, pending exposure, storage forecasts, and maximum
   node-hours before each production batch.

3. Cross-scale interpretation review:
   Look for tiling artifacts, denominator-driven `dBB`, sample-selection bias,
   residual confounding, and inconsistent regime definitions.

4. Independent final review:
   Inspect final code, validation evidence, plots, tables, and continuation
   recommendation without relying on development notes.

Reconcile all findings yourself.

==================================================
COMPLETION GATE
==================================================

Phase 5 is complete only when the approved bounded campaign has been reported.

Report:
- which scales were executed;
- which proposed scales were deferred;
- which optional extensions were executed;
- which optional extensions were deferred;
- measured versus forecast node-hours;
- cumulative ledger total;
- remaining budget;
- output inventory;
- robust conclusions;
- likely conclusions;
- suggestive trends;
- unresolved ambiguities;
- whether any further campaign is scientifically justified.

Do not imply that deferred scales, snapshots, tilings, or diagnostics were
validated.

==================================================
DELIVERABLES
==================================================

Provide:
1. approved bounded campaign plan;
2. scale-specific selection tables;
3. per-batch ledger and resource reports;
4. validated cross-scale outputs;
5. pair-count and exclusion diagnostics;
6. finite-support and contributing-block diagnostics;
7. uncertainty, local-slope, and scale-dependent trend figures;
8. matched-comparison figures;
9. robustness figures;
10. shifted-tiling or time-variability results only if separately approved;
11. output inventory;
12. Phase 5 status report;
13. recommendation for any further work;
14. list of files created or modified;
15. unresolved ambiguities.
