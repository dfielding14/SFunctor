# Phase 1 Execution Report

## Status

**Completed primary-only census.**

The trusted final artifact directory is:

```text
/lustre/orion/ast207/proj-shared/dfielding/Production_plm/prompt1_catalog/t6_final_primary_20260530
```

The user identified the `mhd_sgs` products as broken while an earlier hardened
census was running on May 30, 2026. That job was cancelled immediately. The
final workflow excludes SGS products from validation, cache assembly, catalog
construction, structured second-pass verification, analysis, and pilot
matching.

Do not treat either earlier artifact directory as a final result:

```text
/lustre/orion/ast207/proj-shared/dfielding/Production_plm/prompt1_catalog/t6_final_20260530
/lustre/orion/ast207/proj-shared/dfielding/Production_plm/prompt1_catalog/t6_final_hardened_20260530
```

The first directory contained obsolete SGS-backed catalogs. The second was an
aborted intermediate attempt. Both were deleted after the final primary-only
audit at the user's request. Historical job accounting remains in the
persistent ledger.

## Staged Execution

All final jobs used one Andes CPU node, allocation `AST207`, partition `batch`,
QoS `normal`, and 32 CPUs. Each job was registered in the persistent compute
ledger before submission.

| Slurm job | Stage | State | Runtime | Node-hours | Notes |
|---:|---|---|---:|---:|---|
| `3314767` | validation iteration 1 | failed as intended | `00:00:51` | `0.014167` | Exposed finite float32 standardized-moment discrepancies. |
| `3314768` | validation iteration 2 | failed as intended | `00:00:49` | `0.013611` | Proved that numerator-only standardized bounds were insufficient. |
| `3314769` | final validation | completed | `00:00:49` | `0.013611` | Passed interval-normalized writer-uncertainty validation. |
| `3314770` | primary cache and catalogs | completed | `00:18:06` | `0.301667` | Fresh streamed `mhd_u_bcc_80` assembly and five catalogs. |
| `3314771` | structured second-pass verification | completed | `00:01:49` | `0.030278` | Rebuilt exact grids, raw-derived statistics, flags, and provenance graph. |
| `3314772` | analysis and pilot proposal | completed | `00:06:19` | `0.105278` | Published hashed analysis outputs. |

The primary-only rerun consumed `0.478612` node-hours including the two
rejected validation iterations. The persistent workflow ledger includes prior
superseded work and reports:

```text
Consumed node-hours: 1.783056
Remaining budget: 4998.216944
Pending maximum additional exposure: 0
Accounting-flagged records: 0
```

## Validation

The final validation token binds the trusted data root, time `t=6.0`, cycle
`799945`, simulation input hash, primary snapshot identities, full primitive
snapshot identity, and source hashes.

Direct primitive-to-cbin validation passed `603 / 603` comparisons across:

- one `40^3`, one `80^3`, and one `160^3` cbin voxel;
- merged regions crossing rank boundaries in each coordinate direction;
- one merged region crossing rank boundaries in all three directions;
- factor-40 versus factor-80 and factor-160 hierarchy comparisons.

The validator compares raw first-through-fourth moments before derived
quantities. It then checks retained variances, standard deviations, resolved
skewnesses and kurtoses, density contrast, exact coarse mass-weighted mean
velocities, magnetic energy, and labeled Alfvén-speed proxies. Three
cancellation-dominated standardized moments were explicitly unavailable.
Another 109 finite standardized comparisons were evaluated with field-specific
interval-normalized float32 writer uncertainty bounds, including numerator
and variance-denominator uncertainty. Three required expanded allowance
beyond the baseline retained-diagnostic tolerance.

## Catalog Graph

The completed build, structured second-pass verifier, and analysis marker all
bind the same artifact graph:

```text
0504464a4fa967bb5fb037c4283fa38dde533638bdd80c2e0e79182c2a7f0742
```

The separately audited verifier source SHA256 is:

```text
2f23a787c2fc4b6a10c758772e2a73456a3a373834b75d5a5525225bd82892d0
```

The strict primary cache contains the expected 32 `mhd_u_bcc` raw-moment
fields over `65536` rank shards with zero coverage holes, zero overlaps, and
zero missing rank locations.

| `L_sub / dx` | Rows | Flagged rows | Required primitive ranks per row |
|---:|---:|---:|---:|
| `80` | `2097152` | `227706` | `1` |
| `160` | `262144` | `7435` | `1` |
| `320` | `32768` | `116` | `2` |
| `640` | `4096` | `2` | `16` |
| `1280` | `512` | `0` | `128` |

Structured second-pass verification passed for all scales. It checks the
all-shard filesystem metadata fingerprint, raw payload schema, embedded and
external manifests, exact grid coordinates and parent mappings, source-rank
mappings, foundational raw-statistic reconstruction, derived formulas,
finite availability, ratio flags, aggregate validity flags, and graph hashes.
It shares low-level helpers with the builder and is not a fully independent
source-to-cache reconstruction. All `dBB` values are finite and unflagged.
The table's flagged-row counts are aggregate validity flags, primarily for
less stable higher-order moments.

## Census Summary

The analysis completion marker hashes 30 generated files and binds them to the
verified artifact graph.

| `L_sub / dx` | `dBB` q16 | `dBB` median | `dBB` q84 | `dBB` q99 |
|---:|---:|---:|---:|---:|
| `80` | `0.136693` | `0.253268` | `0.527346` | `1.97883` |
| `160` | `0.199861` | `0.356112` | `0.733763` | `2.79686` |
| `320` | `0.289501` | `0.498766` | `1.01018` | `3.68184` |
| `640` | `0.423410` | `0.695996` | `1.33717` | `4.82099` |
| `1280` | `0.638682` | `0.991299` | `1.81578` | `5.35717` |

Median `dBB` rises with averaging scale. At `L_sub=640`, the pilot scale,
domain-wide Spearman correlations are `-0.85655` for `dBB` versus `B_mean`
and `0.70355` for `dBB` versus `deltaB`. These are descriptive ratio
associations, not an independent causal separation of weak mean field and
elevated fluctuation amplitude.

## Pilot Proposal

The proposal contains 21 verified `L_sub=640` regions:

- 12 representative regions across low, near-median, and high `dBB`;
- 3 accepted low/high matched pairs, represented by 6 rows;
- 3 targeted outliers.

Each row records bounds, centers, environmental properties, the verified
primitive-rank list, and extraction estimates. An `L_sub=640` extraction
touches 16 primitive rank files and has an estimated eight-field payload of
`7.8125 GiB`. No full-resolution pilot extraction or structure-function
calculation was launched.

## Core Artifacts

```text
validation/VALIDATION_COMPLETE.json
validation/validation_results.json
validation/validation_report.md
cache/raw_mhd_u_bcc_80.npz
cache/raw_mhd_u_bcc_80_manifest.json
cache/rank_map.npy
catalogs/catalog_L{80,160,320,640,1280}.npz
verification/VERIFY_COMPLETE.json
verification/verification_results.json
verification/verification_report.md
analysis/ANALYSIS_COMPLETE.json
analysis/analysis_manifest.json
analysis/scientific_summary.md
analysis/distribution_quantiles.csv
analysis/spearman_correlations.csv
analysis/cross_scale_dBB.csv
analysis/cross_scale_dBB_transitions.csv
analysis/pilot_sample.csv
analysis/pilot_sample_metadata.json
analysis/plots/*.png
```

The trusted primary schema cannot reconstruct primitive velocity dispersions,
Mach numbers, kinetic partitions, pressure, mixed velocity-magnetic
statistics, or off-diagonal covariance tensors. Those omissions are explicit
and must remain so until a separately validated source product exists.

## Post-Run Presentation Correction

After the final artifact audit, the validation Markdown table heading was
clarified from `Max relative difference` to `Max raw-moment relative
difference`. The same heading-only change was applied to the future validator
template. This did not alter `validation_results.json`, the validation token,
the catalog graph, verification results, or analysis outputs.

The archived validation token therefore correctly retains the compute-stage
validator SHA256:

```text
b8ee2cd3cfe1cd47477c1158779ceb85359c442387dd9a726cf3c8689b605e15
```

The local validator template SHA256 after the presentation-only correction is:

```text
dbfd4ef92197d890f5621dac0d227512c60c86acb477900b2dfbe830b0eeee70
```
