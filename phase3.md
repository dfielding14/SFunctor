# Phase 3: finite-domain 3D sampler validation and smallest structure-function smoke test

Read `phase0.md`, `phase2.md`, the completed Phase 2 status report, and the
Phase 2 extractor manifests before starting.

Begin this phase only after Phase 2 has a documented GO decision.
Phase 2 is documented as GO in `PHASE2_STATUS_UPDATE.md`. Its
shared-filesystem cache caveat is planning context, not a Phase 3 blocker.

This phase is intentionally limited to implementing and validating the
non-periodic 3D pair sampler, generalizing the strict scientific accumulator,
and running the smallest scientifically useful smoke test on the four
benchmarked `L_sub = 640` cubes. Do not launch the 21-region pilot.

==================================================
OBJECTIVE
==================================================

Build a trusted CPU-only finite-domain 3D structure-function path for selected
non-periodic cubes.

The largest numerical risk is boundary handling. Existing strict directional
structure functions in `sfunctor/core/directional.py` operate on periodic 2D
slices. Reuse validated basis definitions and output concepts where
appropriate, but replace the pair-generation and periodic-lookup path with a
non-periodic 3D design.

The output of this phase is a measured scientific and numerical go/no-go
recommendation for the bounded 21-region `L_sub = 640` Phase 4 pilot.

==================================================
INHERITED HARD STOPS
==================================================

Retain the durable Phase 0 and Phase 2 operational and data-provenance hard
stops:
- treat the trusted Phase 1 tree as read-only;
- reuse the approved Phase 2 extracted cubes and manifests;
- do not rerun the census;
- do not use `mhd_sgs` or `mhd_dynamo_ks`;
- do not use periodic wrapping inside extracted cubes;
- run heavy work only through Andes CPU Slurm allocations;
- register every nontrivial allocation before submission;
- write Slurm stdout and stderr under `logs/`;
- do not add Slurm email-notification directives;
- keep at most one debug job queued or running at a time;
- use unique output directories and restartable outputs;
- do not launch `L_sub = 1280` work.

The Phase 2 prohibition on production structure-function implementation was a
Phase 2 scope limit. This Phase 3 document explicitly supersedes that local
limit only for the bounded sampler-validation and smoke-test work below.

==================================================
TASK 1: DEFINE THE FINITE-DOMAIN PAIR MODEL
==================================================

For every displacement vector `r`, sample point pairs:

    x
    x + r

Require both endpoints to remain inside the extracted cube. Never wrap from
one extracted-cube boundary to the opposite boundary.

Implement two explicit sampling modes.

## Primary mode: nested core

Define:
- an outer extracted cube of side `L_sub`;
- a configured displacement set;
- a maximum separation `ell_max`;
- an inner set of allowed base points such that every configured displacement
  from every allowed base point remains inside the outer cube.

Use the same geometric sampling rules for every directional bin whenever
possible.

## Robustness mode: all valid pairs

For a limited number of small validation cases:
- allow any pair whose two endpoints remain inside the cube;
- record valid-pair counts as a function of separation and angle;
- compare against nested-core results;
- quantify whether finite-domain geometry changes slopes, anisotropy, or
  directional comparisons.

Treat nested core as the primary production mode unless validation demonstrates
that another choice is scientifically preferable.

Keep `L_sub`, pair separation `ell`, and structure-function order `p`
conceptually and programmatically distinct.

==================================================
TASK 2: BUILD A SLOW TRUSTED 3D ORACLE
==================================================

Before optimizing, implement a slow explicit 3D reference calculation for
small synthetic cubes.

The oracle must:
- enumerate or explicitly sample valid non-periodic pairs;
- reject out-of-domain endpoints;
- support both nested-core and all-valid-pairs rules;
- record attempts, accepted pairs, and exclusions;
- support deterministic random seeds;
- support arbitrary positive real `p`;
- make axis conventions explicit;
- remain simple enough to inspect line by line.

Use the oracle to validate the optimized path. Performance is secondary for the
oracle.

==================================================
TASK 3: GENERALIZE THE STRICT 3D ACCUMULATOR
==================================================

For a vector field `q`, support:

    S_p^q(ell) = < |delta q|^p >

and, where scientifically appropriate:

    S_p,perp^q(ell) = < |delta q_perp|^p >

with:

    delta q = q(x + r) - q(x)

Document:
- whether total-vector or perpendicular-vector increments are used;
- whether longitudinal or transverse scalar increments are supported, omitted,
  or deferred;
- whether absolute increments are used;
- normalization;
- binning;
- sampled-pair counts;
- excluded-pair counts;
- uncertainty estimates;
- fit intervals;
- random seeds.

The API must support configurable positive real `p`. Keep the Phase 3 smoke
matrix deliberately small:

    q = B, u
    p = 2

After the baseline passes, exercise a small API-level test with:

    p = 1, 2, 3, 4

Do not execute the full variable-by-order matrix in this phase.

==================================================
TASK 4: IMPLEMENT THREE-DIRECTION CONDITIONING
==================================================

For every pair, define:

    B_loc,pair = [B(x) + B(x + r)] / 2

    e_parallel,pair = B_loc,pair / |B_loc,pair|

For a chosen vector field `q`, define:

    delta q_perp =
        delta q - (delta q . e_parallel,pair) e_parallel,pair

    e_xi = delta q_perp / |delta q_perp|

    e_lambda = e_parallel,pair x e_xi

Classify separation directions using explicit angular wedges around:

    e_parallel,pair
    e_xi
    e_lambda

Treat `e_xi` and `e_lambda` as unoriented statistical axes. Require sign
invariance under:

    e_xi -> -e_xi
    e_lambda -> -e_lambda

Use folded angles or an equivalent documented rule so sign choices cannot
change directional classification.

Record and exclude degenerate pairs when:
- `|B_loc,pair|` is too small;
- `|delta q_perp|` is too small;
- the perpendicular separation needed for azimuthal classification is too
  small;
- density is invalid for density-dependent variables;
- any required field is non-finite.

Never force an undefined direction into a valid angular bin.

Also implement a clearly labeled subvolume-mean-field comparison:

    B_mean,sub = <B>_sub

Use pair-scale local-field conditioning for primary science and the
subvolume-scale field only as a diagnostic comparison.

Implement reducers for the statistical eddy dimensions:

    ell_parallel
    xi
    lambda

and the scale-dependent aspect ratios:

    ell_parallel / lambda
    xi / lambda

Document the interpolation, matching, uncertainty, and invalid-bin rules used
to derive these quantities from directional structure functions. Infer aspect
ratios by comparing directional curves at fixed `S_p`, not merely at equal
`ell`.

==================================================
TASK 5: SUPPORT COMPRESSIBLE-MHD FIELD VARIANTS
==================================================

Support the following configurable `q` variants:

    B
    u
    v_A
    v_A_ref
    z_plus
    z_minus
    z_plus_ref
    z_minus_ref

where:

    v_A = B / sqrt(rho)

    z_plus  = u + v_A
    z_minus = u - v_A

    v_A_ref = B / sqrt(rho_0)

    z_plus_ref  = u + v_A_ref
    z_minus_ref = u - v_A_ref

Use the AthenaK code-unit convention with `4 pi` absorbed where appropriate.

Make `rho_0` an explicit configuration value and record its provenance. State
whether it is:
- a global simulation reference density; or
- a subvolume mean used for a labeled comparison.

Treat `v_A_ref` and `z_*_ref` as reference-density comparison diagnostics. Do
not conflate them with pointwise-density quantities or with uniquely preferred
compressible-MHD variables.

The smoke test remains restricted to `B` and `u`. The additional variants must
be validated on small synthetic or reduced cases before Phase 4 expansion.

==================================================
TASK 6: VALIDATE AGGRESSIVELY
==================================================

Test at minimum:

1. Oracle agreement:
   - optimized versus slow 3D reference;
   - exact counts and numerical agreement on small cubes;
   - nested-core and all-valid-pairs modes.

2. Finite-domain behavior:
   - confirm no periodic wrapping;
   - inspect pair counts versus `ell` and angle;
   - compare multiple `ell_max` values;
   - compare nested-core sizes.

3. Synthetic isotropic fields:
   - verify no spurious anisotropy;
   - rotate and translate fields;
   - permute axes.
   - reverse `e_xi` and `e_lambda` signs and confirm invariant classification.

4. Synthetic guide-field anisotropy:
   - impose a known preferred direction;
   - recover parallel versus perpendicular behavior.

5. Synthetic ribbon-like structures:
   - construct controlled `ell_parallel`, `xi`, and `lambda`;
   - recover the expected ordering and aspect ratios.

6. Degenerate cases:
   - `B_loc,pair` close to zero;
   - `B_mean,sub` close to zero;
   - `delta q_perp` close to zero;
   - low or invalid density;
   - NaNs and infinities;
   - empty or sparse angular bins.

7. Reproducibility:
   - fixed random seeds;
   - repeated-run agreement;
   - stable results under increasing sample count.

8. Sensitivity:
   - angular wedge widths;
   - separation-bin widths;
   - `ell_max`;
   - nested-core versus all-valid-pairs differences.

Do not claim fitted slopes or anisotropy when accepted counts, scale range, or
fit stability are inadequate. A documented no-go result is valid.

==================================================
TASK 7: PROFILE AND RUN THE SMALLEST SMOKE TEST
==================================================

Profile:
- chunk size;
- pair batch size;
- sampled-pair count;
- number of separation bins;
- number of angular bins;
- number of `p` values;
- runtime per `q`;
- peak RSS;
- output size;
- restart behavior.

After synthetic validation passes, run the smallest scientifically useful
smoke test on the four approved Phase 2 cubes with:

    q = B, u
    p = 2

Use an empirically justified conservative `ell_max`, initially testing values
of order:

    ell_max <= L_sub / 4

Do not assume the final cutoff in advance.

==================================================
SUBAGENT REVIEWS
==================================================

Use independent subagents for:

1. Finite-domain audit:
   Review nested-core geometry, all-valid-pairs behavior, excluded-pair
   accounting, and boundary-bias risks.

2. Scientific-definition audit:
   Compare the 3D definitions against the validated 2D strict implementation
   and document any intentional changes.

3. Adversarial validation:
   Try to break the sampler with rotations, translations, axis permutations,
   sparse bins, weak fields, invalid densities, and boundary edge cases.

4. Performance review:
   Inspect chunking, peak memory, CPU scaling, and the projected Phase 4 cost.

5. Independent final review:
   Inspect code, tests, and representative results without relying on
   development notes.

Reconcile all findings yourself.

==================================================
GO / NO-GO GATE
==================================================

Phase 3 is a GO only if:
- no periodic wrapping is observed;
- optimized results agree with the slow 3D oracle;
- accepted and excluded pair counts are explicit;
- rotation, translation, and axis-permutation tests pass;
- degenerate directions are excluded rather than misclassified;
- nested-core versus all-valid-pairs differences are characterized;
- directional bins retain scientifically usable counts;
- any slope fit uses an explicitly justified stable interval;
- the four-cube smoke test passes;
- runtime, RSS, output size, and restart behavior are measured;
- the full repository test suite passes;
- the projected Phase 4 cost is ledger-backed and acceptable.

Phase 3 is a NO-GO if sparse bins, edge bias, unstable fits, or unexplained
oracle differences prevent defensible interpretation.

==================================================
DELIVERABLES
==================================================

Provide:
1. slow finite-domain 3D oracle;
2. optimized nested-core 3D sampler;
3. limited all-valid-pairs robustness implementation;
4. arbitrary-positive-real-`p` accumulator API;
5. pair-scale and subvolume-scale magnetic-field conditioning;
6. statistical eddy-dimension and aspect-ratio reducers;
7. configurable compressible-MHD field variants;
8. synthetic-validation suite;
9. pair-count and exclusion diagnostics;
10. four-cube `B`, `u`, `p = 2` smoke-test results;
11. CPU and memory benchmark report;
12. Phase 3 status report;
13. explicit GO or NO-GO recommendation for Phase 4;
14. list of files created or modified;
15. unresolved ambiguities.

Do not begin the 21-region pilot until this gate is documented.
