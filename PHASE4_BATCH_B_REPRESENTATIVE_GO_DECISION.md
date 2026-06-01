# Phase 4 Batch B representative-probe GO decision

| Item | Value |
| --- | --- |
| Date | 2026-06-01 |
| Repository | `SFunctor` |
| Branch | `cleanup/cpu-production` |
| Evidence checkpoint | `PHASE4_BATCH_A2_3POINT_EXTENSION_STATUS_UPDATE.md` |
| Decision scope | Bounded representative Batch B probe only |
| Status | **GO for the exact representative probe below. No all-21 Batch B expansion is authorized.** |

## Approved Probe

Run the retained `L_sub = 640` extraction with:

```text
q = B, u
p = 1, 2, 3, 4, 5, 6
stencil = labeled 2-point
ell_max = 320 cells
separation bins = 64
directions per bin = 24
primary curve-level policy = all_valid_origins
directional robustness overlay = shell_local
```

Use the exact eight-cube representative set:

| Cube | Role |
| --- | --- |
| `L640_sub00370` | Low-`dBB` ordinary control |
| `L640_sub03942` | Median-`dBB` ordinary control |
| `L640_sub00579` | High-`dBB` ordinary control |
| `L640_sub00738` | Weak-mean-field ordinary control |
| `L640_sub03026` | Strong supported 3-point policy-sensitivity case |
| `L640_sub00732` | Supported magnetic-parallel policy-sensitivity case |
| `L640_sub02822` | Supported magnetic-lambda policy-sensitivity case |
| `L640_sub02602` | Supported velocity policy-sensitivity case |

## Boundaries

This decision does not authorize:

- an all-21 Batch B launch;
- a 5-point expansion;
- fitted directional exponents;
- a physical interpretation of policy differences without matched-origin or
  origin-seed diagnostics;
- Batch C variables;
- SGS-derived channels;
- `L_sub = 1280`.

Review the representative $p = 6$ tails, accepted counts, effective blocks,
bootstrap validity, signed primary-over-shell ratios, and the complete
`L640_sub03026` tail before considering any all-21 Batch B expansion.
