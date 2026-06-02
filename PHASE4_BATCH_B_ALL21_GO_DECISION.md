# Phase 4 Batch B all-21 extension GO decision

| Item | Value |
| --- | --- |
| Date | 2026-06-02 |
| Repository | `SFunctor` |
| Decision scope | Exact all-21 `L_sub = 640` Batch B 2-point extension only |
| Authorization source | Explicit user request |
| Status | **GO for the exact guarded all-21 extension below. No broader Phase 4 expansion is authorized.** |

## Evidence Acknowledgement

The representative Batch B review dated 2026-06-01 recorded a **HOLD** for
automatic all-21 expansion because large-scale high-order tails were sensitive
to policy-specific origin sampling. The first bounded follow-up was retained as
discarded exploratory evidence because an independent audit found that its
population weighting changed the production estimand. The replacement
matched-origin diagnostic partitions one production-equivalent intrinsic
schedule exactly, separately measures shell-schedule sensitivity, adds a
separately labeled exterior overlay, and retains raw block accumulators plus
bounded rare-event records.

This post-diagnostic explicit user-authorized GO decision acknowledges that the
high-order tails remain imperfect and authorizes one bounded all-21 expansion
for review. It does not reinterpret the representative tail sensitivity, claim
that the two origin policies are interchangeable, call $p = 6$ converged, or
authorize fitted directional exponents.

The adapter must bind:

- the immutable Batch A release;
- the immutable representative Batch A2 release;
- the immutable all-21 Batch A2 3-point extension;
- the immutable representative Batch B release;
- `config/phase4_batch_b_representative_review_decision.json`;
- `PHASE4_BATCH_B_REPRESENTATIVE_STATUS_UPDATE.md`;
- the representative review summary and figure manifest;
- the completed corrected matched-origin tail-diagnostic summary and marker;
- `config/phase4_batch_b_all21_go_decision.json` and this report.

## Approved Extension

Run the retained `L_sub = 640` extraction with:

```text
q = B, u
p = 1, 2, 3, 4, 5, 6
stencil = labeled 2-point only
ell_max = 320 cells
separation bins = 64
directions per bin = 24
support policies = all_valid_origins, shell_local
density weighting = none
cube set = exact extraction.PHASE4_PILOT_CUBE_IDS all-21 tuple
```

The exact cube IDs, in frozen extraction order, are:

```text
L640_sub00370
L640_sub02822
L640_sub03026
L640_sub02615
L640_sub03942
L640_sub00957
L640_sub03356
L640_sub01582
L640_sub00579
L640_sub00886
L640_sub00032
L640_sub02279
L640_sub01088
L640_sub02297
L640_sub00732
L640_sub02000
L640_sub02602
L640_sub02249
L640_sub00738
L640_sub01591
L640_sub01651
```

## Boundaries

This decision does not authorize:

- SGS-derived channels;
- a 5-point expansion;
- fitted directional exponents;
- Batch C variables;
- `L_sub = 1280`;
- density weighting;
- an automatic interpretation of origin-policy differences.

The all-21 output remains a bounded review product. Any broader follow-up
requires a separate decision artifact.
