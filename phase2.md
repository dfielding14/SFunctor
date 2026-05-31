# Phase 2: selected-cube 3D extraction and bounded performance benchmarking

Read `phase0.md`, `PHASE1_STATUS_UPDATE.md`,
`docs/PHASE1_EXECUTION_REPORT.md`, and
`docs/PHASE1_RECONSTRUCTABILITY.md` before starting.

This phase is intentionally limited to implementing, validating, and
benchmarking a selected-cube full-resolution 3D primitive extractor. Do not
implement production structure functions in this phase. Do not extract the
full 21-region pilot in this phase.

==================================================
OBJECTIVE
==================================================

Build a reliable CPU-only extractor that materializes selected
full-resolution 3D primitive cubes from the `10240^3` nonrelativistic MHD
snapshot without scanning or loading the entire full-resolution domain.

The extractor must consume the Phase 1 pilot table directly, preserve the
AthenaK array ordering, attach provenance, and validate materialized cubes
against the supported `cbin` census quantities.

The immediate benchmark is restricted to four proposed `L_sub = 640` cubes:

    L640_sub00370
    L640_sub03942
    L640_sub00579
    L640_sub00738

The output of this phase is a measured go/no-go recommendation for Phase 3.

==================================================
INHERITED HARD STOPS
==================================================

Treat the following as mandatory:

1. Use only the read-only trusted Phase 1 run:

       /lustre/orion/ast207/proj-shared/dfielding/Production_plm/prompt1_catalog/t6_final_primary_20260530

   Define:

       TRUSTED_RUN=/lustre/orion/ast207/proj-shared/dfielding/Production_plm/prompt1_catalog/t6_final_primary_20260530

2. Reuse its verified cache, catalogs, rank map, and pilot table:

       $TRUSTED_RUN/analysis/pilot_sample.csv
       $TRUSTED_RUN/analysis/pilot_sample_metadata.json

3. Do not rerun the Phase 1 census merely to begin extraction.

4. Do not use `mhd_sgs` or `mhd_dynamo_ks`. They remain excluded
   reconstruction inputs.

5. Use the existing AthenaK reader scripts where appropriate, including:

       /ccs/home/dfielding/athenak-df/vis/python/bin_convert.py

   Audit the reader behavior rather than replacing it casually.

6. Run heavy work only through Andes CPU Slurm allocations. Login-node work
   must remain lightweight.

7. Register every nontrivial Slurm allocation in the retained compute ledger
   before submission:

       /lustre/orion/ast207/proj-shared/dfielding/Production_plm/prompt1_catalog

8. Use unique output directories. Do not mutate the trusted Phase 1 tree.

9. Write Slurm stdout and stderr under `logs/`. Do not add Slurm
   email-notification directives.

10. Keep at most one debug job queued or running at a time.

==================================================
KNOWN DATA LAYOUT
==================================================

The full-resolution domain is:

    10240^3 cells

The rank-local files are located under:

    /lustre/orion/ast207/proj-shared/dfielding/Production_plm/data/data_Turb_10240_beta25_dedt025_plm/bin

Each rank-local high-resolution block has dimensions:

    nx1 = 160
    nx2 = 320
    nx3 = 320

The Phase 1 catalog already provides:
- half-open selected-cube coordinate bounds;
- required rank-local file IDs;
- environmental properties;
- `dBB` selector information;
- estimated memory footprint;
- representative, matched, or outlier status.

Use those products directly.

==================================================
TASK 1: AUDIT EXISTING EXTRACTION FOUNDATIONS
==================================================

Inspect and reuse the repository foundations before adding new code:
- `scripts/phase1/cbin_tools.py`;
- `scripts/phase1/validate_reconstruction.py`;
- `sfunctor/io/rank_manifest.py`;
- `sfunctor/io/extract.py`;
- the external `bin_convert.py` reader listed above.

Document:
- file headers and snapshot identity;
- field labels;
- dtypes;
- units or code-unit conventions;
- KJI versus Cartesian ordering;
- logical block coordinates;
- physical bounds;
- ghost-zone behavior;
- whether rank-local blocks overlap or tile exactly;
- required read volume for each selected cube.

Add an automated primitive-bin preflight that records:
- file existence;
- header identity;
- snapshot time and cycle;
- expected field inventory;
- unique referenced rank IDs;
- expected bytes read;
- expected cube shape;
- expected materialized output size;
- any rejected or inconsistent file.

==================================================
TASK 2: IMPLEMENT THE SELECTED-CUBE EXTRACTOR
==================================================

Implement a packaged CLI and reusable API that:
- reads one pilot-table row;
- identifies the minimum rank-local files;
- loads only required fields and intersecting data;
- crops block intersections using half-open bounds;
- assembles one selected cube in a documented global order;
- preserves the native KJI ordering unless an explicit conversion is recorded;
- detects duplicate cells;
- detects missing cells;
- reports exact coverage;
- records source files and source headers;
- records code commit and configuration;
- writes restartable outputs with completion markers;
- avoids unnecessary copies;
- reports wall time, read time, output size, and peak child-process RSS.

The output format must be explicit and self-describing. Include:
- cube ID;
- global half-open bounds;
- shape;
- axis convention;
- field names;
- field dtypes;
- units or code-unit convention;
- source snapshot;
- required ranks;
- checksum or equivalent integrity metadata;
- extraction configuration;
- completion state.

Do not silently transpose axes. Make any conversion explicit in metadata and
tests.

==================================================
TASK 3: VALIDATE EXTRACTION AGAINST CBIN
==================================================

The extraction-versus-`cbin` gate must use only quantities that Phase 1 proved
reconstructable.

For each validation region:

1. Convert extracted primitive fields to the conserved fields used by the
   Phase 1 census.

2. Compare first through fourth raw moments of:

       dens
       mom1
       mom2
       mom3
       ener
       bcc1
       bcc2
       bcc3

3. Compare supported derived quantities:

       B_mean
       B_rms
       deltaB
       dBB
       B_mean^2 / <B^2>
       deltaB^2 / <B^2>
       density summaries
       magnetic energy
       coarse mass-weighted velocity means

4. Reuse the documented Phase 1 tolerances and propagated precision handling.

5. Merge raw moments before deriving central moments. Never average child
   variances, skewnesses, kurtoses, or `dBB`.

6. Report standardized moments only when their precision flags show that they
   are numerically resolved.

Do not use the following as extraction-versus-`cbin` validation targets:

    volume-weighted delta u
    sonic Mach number M_s
    Alfvén Mach number M_A
    pressure
    kinetic-energy partitions
    magnetic-to-kinetic energy ratios

Those may be newly computed from extracted primitive cubes later, with
explicit definitions, but Phase 1 did not provide an independent `cbin`
oracle for them.

Test at minimum:
- a region contained within one rank-local file;
- regions crossing rank boundaries in each coordinate direction;
- one merged cross-rank region;
- all four benchmark pilot IDs;
- held-out probes at remote Morton ranks;
- domain-edge probes;
- extreme-outlier probes;
- randomized stratified held-out probes;
- exact shape and cell coverage;
- deliberate duplicate-cell or missing-cell corruption rejection;
- representative visual slices for axis-order inspection.

==================================================
TASK 4: BENCHMARK ONLY THE FOUR-ID SUBSET
==================================================

A `640^3` cube containing eight float32 fields has an estimated payload of:

    7.8125 GiB

This is a planning estimate, not an operational benchmark.

Extract only the four named `L_sub = 640` cubes. Measure:
- primitive-bin preflight time;
- observed operational wall time;
- extraction assembly time;
- bytes read;
- output bytes written;
- peak child-process RSS;
- restart behavior;
- checksum or integrity-check time;
- failure cleanup behavior.

Use these measurements to forecast:
- the 21-region `L_sub = 640` extraction campaign;
- temporary-storage exposure;
- output-storage exposure;
- expected node-hours;
- maximum node-hours;
- whether chunking or a different output layout is required.

Do not extract `L_sub = 1280` cubes in this phase.

==================================================
SUBAGENT REVIEWS
==================================================

Use independent subagents for:

1. Extraction audit:
   Review rank mapping, bounds, orientation, overlap handling, ghost treatment,
   coverage accounting, and provenance.

2. Adversarial test review:
   Try to construct axis transpositions, off-by-one bounds, missing-file
   behavior, duplicate-cell behavior, interrupted outputs, and stale completion
   markers.

3. Performance review:
   Inspect RSS measurement, I/O accounting, avoidable copies, output layout,
   and the 21-region forecast.

Reconcile the reviews yourself. Do not accept suggestions blindly.

==================================================
GO / NO-GO GATE
==================================================

Phase 2 is a GO only if:
- all primitive-file preflight checks pass;
- source headers match the expected snapshot identity;
- every extracted cube has exact expected shape and cell count;
- no unexplained overlaps, dropped cells, or axis ambiguities remain;
- supported extraction-versus-`cbin` comparisons pass documented tolerances;
- deliberate corruption is rejected;
- observed operational runtime, peak RSS, output size, and restart behavior are
  measured;
- all repository tests pass;
- the four-ID benchmark is registered in the compute ledger;
- a ledger-backed 21-region forecast is documented.

Phase 2 is a NO-GO if any mismatch is unexplained or if resource behavior has
not been measured.

Uncontrolled shared-filesystem cache state is not, by itself, a NO-GO
condition. Record it as a timing caveat and proceed using measured operational
runtime. A separate cold/warm cache experiment is optional and is not a
prerequisite for Phase 3.

==================================================
DELIVERABLES
==================================================

Provide:
1. selected-cube extractor API and CLI;
2. primitive-bin preflight tool;
3. focused extraction tests;
4. per-cube provenance manifests;
5. supported extraction-versus-`cbin` comparison tables;
6. visual axis-order inspection figures;
7. four-ID bounded performance benchmark tables;
8. restart and corruption-test results;
9. ledger entries and resource forecast;
10. Phase 2 status report;
11. explicit GO or NO-GO recommendation for Phase 3;
12. list of files created or modified;
13. unresolved ambiguities.

Do not begin Phase 3 implementation until this gate is documented.
