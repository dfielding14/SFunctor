# Prompt 2: extract selected full-resolution 3D subvolumes and calculate finite-domain conditional structure functions

Use this only after reviewing the cbin census and selecting the pilot subvolumes.

Your next task is to extend the validated structure-function workflow to selected full-resolution 3D subvolumes of the 10240^3 nonrelativistic MHD domain.

This task should use the pilot subvolumes selected from the cbin census. Do not scan or load the entire full-resolution domain.

The scientific goal is to determine how local turbulent statistics depend on the coarse-grained environment, especially dBB, and how those relationships depend on the environmental averaging scale L_sub.

The primary quantities of interest include:
- arbitrary-order structure functions;
- scale-dependent slopes;
- anisotropy relative to the magnetic field;
- three-direction conditional structure functions;
- the statistical eddy dimensions ℓ_parallel, ξ, and λ;
- scale-dependent aspect ratios;
- differences between local environments with dBB << 1, dBB ~ 1, and dBB >> 1;
- whether apparent dBB trends survive matching or conditioning on M_s, M_A, density statistics, and magnetic-field strength.

The selected subvolumes are not periodic. The finite-domain pair-sampling logic must therefore be treated as a major scientific and numerical component of the implementation.

==================================================
USE SUBAGENTS ACTIVELY
==================================================

Use subagents extensively and independently.

At minimum, delegate:

1. Full-resolution extraction audit:
   Verify mapping from selected cbin catalog entries to rank-local single_file_per_rank files, coordinate assembly, orientation, overlap handling, and boundary correctness.

2. Finite-domain pair-sampling audit:
   Review the nested-core design, all-valid-pairs robustness calculation, angular binning, sample-count behavior, and boundary-bias risks.

3. Scientific-definition audit:
   Check all isotropic, anisotropic, and three-direction structure-function definitions against the validated 2D implementation and the relevant literature conventions.

4. Performance review:
   Profile CPU runtime, memory use, I/O behavior, chunking strategy, and scaling with L_sub, p, number of ℓ bins, number of angle bins, and number of sampled pairs.

5. Stress-test and adversarial review:
   Try to break the extraction and analysis pipeline with synthetic fields, coordinate rotations, translations, finite-domain edge cases, sparse angular bins, degenerate magnetic fields, and density pathologies.

6. Independent final review:
   Have a separate subagent inspect the final code, tests, and representative results without relying on the original development notes.

You remain responsible for reconciling the findings, verifying the final implementation yourself, and documenting unresolved ambiguities.

==================================================
KNOWN DATA LAYOUT
==================================================

The full-resolution domain is:

    10240^3 cells.

The rank-local files are located under:

    /lustre/orion/ast207/proj-shared/dfielding/Production_plm/data/data_Turb_10240_beta25_dedt025_plm/bin

Each rank-local high-resolution block has dimensions:

    nx1 = 160
    nx2 = 320
    nx3 = 320.

The cbin census used environmental subvolume sizes:

    L_sub / Δx = 80, 160, 320, 640, 1240.

The cbin catalog should already provide:
- selected pilot-subvolume coordinate bounds;
- mapping to required rank-local files;
- environmental properties;
- dBB regime;
- estimated memory footprint;
- representative, matched, or outlier status.

Use those catalog products directly.

==================================================
PHASE 1: AUDIT THE FULL-RESOLUTION EXTRACTION PIPELINE
==================================================

Before running structure functions, implement or audit a reliable extractor that assembles only the selected full-resolution subvolumes.

For each selected region:
- identify the minimum required rank-local files;
- load only those files;
- extract only the needed cells;
- assemble the subvolume in a consistent global coordinate order;
- verify coordinate bounds;
- verify axis ordering;
- verify field labels;
- verify dtypes;
- verify units;
- verify ghost-zone treatment if relevant;
- verify whether neighboring rank-local files overlap or tile exactly;
- avoid duplicating cells;
- avoid dropping cells;
- report memory use.

Validate the extraction by comparing full-resolution means and moments against the corresponding cbin-derived values for the same region:
- first moments;
- second moments;
- third moments;
- fourth moments;
- B_mean;
- δB;
- dBB;
- δu;
- density moments;
- any Mach numbers and energy statistics used in the census.

Do not proceed until extracted full-resolution volumes reproduce the cbin environmental properties within expected numerical tolerance.

==================================================
PHASE 2: DESIGN THE FINITE-DOMAIN PAIR SAMPLER
==================================================

The extracted subvolumes are not periodic.

Never wrap a pair from one boundary of the extracted subvolume to the opposite boundary.

For each structure-function pair:

    x
    x + ℓ

require both endpoints to lie inside the extracted full-resolution region.

Use a nested-core design for the primary calculation.

That means:
- define an outer extracted cube of side L_sub;
- define an inner set of allowed base points;
- choose a maximum separation ℓ_max;
- require that every sampled displacement from an allowed base point remains inside the outer cube;
- use the same geometric sampling rules for all directional bins whenever possible.

The nested-core design should be the primary production calculation because it provides cleaner control of directional sampling and boundary effects.

For a small number of early test cases, also implement an all-valid-pairs calculation:
- allow any pair with both endpoints inside the cube;
- report the number of valid pairs as a function of ℓ and angle;
- compare against the nested-core result;
- determine whether edge geometry biases slopes, anisotropy, or directional comparisons.

Use the all-valid-pairs calculation as a robustness check, not as the default production path unless validation demonstrates that it is clearly preferable.

Determine an empirically justified maximum separation, likely of order:

    ℓ_max ≲ L_sub / 4

or another conservative fraction of L_sub.

Do not assume the exact cutoff in advance. Validate it using:
- pair counts;
- directional coverage;
- slope stability;
- nested-core versus all-valid-pairs comparisons;
- synthetic fields.

Keep L_sub and ℓ conceptually and programmatically distinct everywhere.

==================================================
PHASE 3: ARBITRARY-ORDER STRUCTURE FUNCTIONS
==================================================

The implementation must support arbitrary structure-function order p, not only p = 2.

Use a configurable list of p values.

For a chosen field q, calculate quantities of the general form:

    S_p^q(ℓ) = < |δq|^p >

and where scientifically appropriate:

    S_p,perp^q(ℓ) = < |δq_perp|^p >.

The initial pilot analysis should include a practical set of orders such as:

    p = 1, 2, 3, 4

and should be written so that additional positive real or integer orders can be added easily.

Document:
- whether absolute increments are used;
- whether longitudinal, transverse, total-vector, or perpendicular-vector increments are used;
- normalization conventions;
- binning;
- sample counts;
- uncertainty estimation;
- fit intervals.

Do not conflate:
- the order p of the structure function;
- the environmental scale L_sub;
- the pair separation ℓ.

==================================================
PHASE 4: THREE-DIRECTION CONDITIONAL STRUCTURE FUNCTIONS
==================================================

Audit and implement the three-direction conditional structure functions carefully.

For every point pair x and x+ℓ, define the pair-scale local mean magnetic field:

    B_loc,pair = [B(x) + B(x+ℓ)] / 2

    e_parallel,pair = B_loc,pair / |B_loc,pair|.

For a chosen vector field q, define:

    δq = q(x+ℓ) - q(x)

    δq_perp = δq - (δq · e_parallel,pair) e_parallel,pair

    e_xi = δq_perp / |δq_perp|

    e_lambda = e_parallel,pair × e_xi.

The three conditional directions are:

1. Parallel direction:
   ℓ is aligned with e_parallel,pair.
   Associated scale: ℓ_parallel.

2. Fluctuation direction:
   ℓ is perpendicular to e_parallel,pair and aligned with e_xi.
   Associated scale: ξ.

3. Perpendicular or sheet-thickness direction:
   ℓ is perpendicular to both e_parallel,pair and e_xi.
   Equivalently, ℓ is aligned with e_lambda.
   Associated scale: λ.

For arbitrary order p, calculate:

    S_p^parallel(ℓ_parallel)
    S_p^fluc(ξ)
    S_p^perp(λ).

Use explicit angular bins.
Record:
- angular-bin definitions;
- angular-bin widths;
- sample counts;
- excluded-pair counts;
- sensitivity to narrower and wider bins.

Handle:
- |B_loc,pair| close to zero;
- |δq_perp| close to zero;
- invalid density;
- NaNs and infinities;
- sparse bins;
- sign invariance of e_xi and e_lambda.

==================================================
PHASE 5: COMPARE PAIR-SCALE AND SUBVOLUME-SCALE MEAN FIELDS
==================================================

There are two distinct magnetic-field directions in this problem.

1. Pair-scale local mean field:

    B_loc,pair(x, ℓ) = [B(x) + B(x+ℓ)] / 2.

2. Subvolume-scale mean field:

    B_mean,sub = <B>_V.

The pair-scale local mean field is the primary definition for the three-direction conditional structure functions.

However, implement a controlled comparison using the subvolume-scale mean field:

    e_parallel,sub = B_mean,sub / |B_mean,sub|.

For representative pilot subvolumes, compare anisotropic structure functions measured relative to:
- e_parallel,pair;
- e_parallel,sub.

The purpose is to determine:
- how much anisotropy is visible relative to the coarse-grained field;
- how coherent the field direction remains within the subvolume;
- whether the result depends strongly on dBB;
- whether high-dBB subvolumes lose a meaningful subvolume-scale preferred direction even when pair-scale local anisotropy remains measurable.

Do not substitute e_parallel,sub for the pair-scale local definition in the primary analysis.

==================================================
PHASE 6: VARIABLE CHOICES
==================================================

The full-MHD simulations are compressible. The clean RMHD literature does not uniquely dictate a single compressible generalization.

Implement and compare:

A. Pointwise-density Alfvén velocity:

    v_A = B / sqrt(rho)

    z_plus  = u + v_A
    z_minus = u - v_A.

B. Fixed-reference-density Alfvén velocity:

    v_A_ref = B / sqrt(rho_0)

    z_plus_ref  = u + v_A_ref
    z_minus_ref = u - v_A_ref.

C. Magnetic and velocity diagnostics:

    B
    v_A
    v_A_ref
    u.

For each q:
- use q to define δq and e_xi;
- use |δq|^p or |δq_perp|^p as appropriate for the measured structure function;
- label the convention explicitly;
- do not silently substitute one variable for another.

Keep the Elsasser-like variables prominent, but treat B, v_A, v_A_ref, and u as important physically informative comparisons.

Document the choice of rho_0.

==================================================
PHASE 7: VALIDATION AND STRESS TESTS
==================================================

Before production calculations, build or reuse slow trusted reference implementations for small 3D finite-domain problems.

Test at least:

1. Extraction correctness:
   - compare extracted moments against cbin;
   - test regions contained inside one rank file;
   - test regions crossing rank-file boundaries;
   - test each coordinate direction;
   - test L_sub values that align differently with rank blocks.

2. Finite-domain behavior:
   - confirm no periodic wrapping;
   - compare nested-core and all-valid-pairs calculations;
   - inspect pair counts as a function of ℓ and angle;
   - test multiple ℓ_max choices.

3. Synthetic isotropic fields:
   - verify no spurious anisotropy;
   - rotate the field and confirm invariance.

4. Synthetic guide-field anisotropy:
   - impose a known preferred direction;
   - confirm recovery of parallel versus perpendicular behavior.

5. Synthetic ribbon-like structures:
   - construct controlled ℓ_parallel, ξ, and λ;
   - confirm recovery of the expected ordering and aspect ratios.

6. Coordinate transformations:
   - translate fields;
   - rotate fields;
   - permute axes;
   - confirm consistent outputs.

7. Degenerate cases:
   - B_loc,pair close to zero;
   - B_mean,sub close to zero;
   - δq_perp close to zero;
   - low density;
   - NaNs and infinities;
   - empty angular bins.

8. Reproducibility:
   - fixed random seeds;
   - repeated-run agreement;
   - stable results under changes in sample size.

9. Resolution and sampling convergence:
   - compare different numbers of sampled pairs;
   - compare angular-bin widths;
   - compare separation-bin widths;
   - compare nested-core sizes.

Use independent subagents to design failure cases and to review whether the tests are genuinely capable of detecting errors.

==================================================
PHASE 8: CPU PERFORMANCE AND MEMORY MANAGEMENT
==================================================

These subvolumes can be very large.

A 640^3 full-resolution cube is definitely feasible.
A 1280^3-scale cube may be feasible but should not be assumed to fit comfortably without measurement.
The requested environmental sizes include:

    L_sub / Δx = 80, 160, 320, 640, 1240.

Profile memory use before launching large cases.

Design the implementation for CPU-only execution.

Measure:
- I/O time;
- extraction time;
- peak memory;
- pair-sampling time;
- binning time;
- runtime per q;
- runtime per p;
- runtime per L_sub;
- scaling with number of sampled pairs;
- scaling with number of angular bins;
- scaling with number of ℓ bins.

Prefer:
- chunked processing;
- streaming;
- precomputed reusable quantities;
- avoiding unnecessary copies;
- avoiding avoidable temporary arrays;
- efficient histogram accumulation;
- CPU parallelism where justified;
- reproducible sampling.

Do not launch expensive L_sub = 1240 calculations until smaller pilot cases have established:
- memory requirements;
- runtime scaling;
- scientific value;
- stable behavior.

==================================================
PHASE 9: PILOT SCIENTIFIC ANALYSIS
==================================================

Run the validated 3D structure-function pipeline on the pilot sample selected from the cbin census.

For each selected subvolume, report:
- L_sub;
- global coordinate bounds;
- dBB;
- B_mean;
- δB;
- B_rms;
- M_s;
- reliable M_A variants;
- density statistics;
- higher moments;
- energy ratios if available;
- representative, matched, or outlier status;
- memory footprint;
- runtime;
- number of sampled pairs.

For each q and each p, calculate:
- isotropic or angle-averaged structure functions where relevant;
- parallel and perpendicular anisotropic structure functions;
- three-direction conditional structure functions;
- fitted inertial-range slopes with explicit fit ranges;
- sample counts;
- uncertainty estimates.

For the three-direction analysis, infer statistical eddy shapes at fixed S_p and calculate scale-dependent aspect ratios such as:

    ξ / λ
    ℓ_parallel / λ.

Keep all conventions explicit.

==================================================
PHASE 10: CORRELATE STRUCTURE-FUNCTION PROPERTIES WITH ENVIRONMENT
==================================================

Begin with the relationship between structure-function outputs and dBB.

At minimum, examine how the following depend on dBB:

    structure-function slopes
    higher-order exponents ζ_p
    ℓ_parallel / λ
    ξ / λ
    differences between pair-scale and subvolume-scale field conditioning.

Then test whether apparent dBB trends survive controlling for:
- M_s;
- M_A;
- mean density;
- density variance;
- B_mean;
- δB;
- B_rms;
- magnetic-to-kinetic energy ratios;
- higher moments of key fields;
- any other strongly correlated environmental properties identified in the cbin census.

Use:
- matched comparisons;
- conditional medians;
- quantile bands;
- rank correlations;
- simple regression or partial-correlation tools where useful;
- careful visual inspection.

Do not overinterpret a small pilot sample.
Distinguish:
- robust trends;
- suggestive trends;
- ambiguous results;
- insufficient statistics.

==================================================
L_sub DEPENDENCE IS A CENTRAL QUESTION
==================================================

Assess how the conclusions depend on the environmental averaging scale:

    L_sub / Δx = 80, 160, 320, 640, 1240.

The key conditional statistic is conceptually:

    S_p(ℓ | dBB(L_sub), other environmental properties at L_sub).

Do not collapse results from different L_sub values into a single label.

Internally assess:
- how environmental classification changes with L_sub;
- whether the same physical region changes dBB regime across scales;
- whether structure-function correlations strengthen or weaken with L_sub;
- whether there is a particularly informative environmental scale.

The paper may ultimately present only a simplified subset, but the internal analysis should understand the scale dependence.

==================================================
DEFERRED DIAGNOSTICS
==================================================

Do not implement the following in the first production pass unless they are already supported cleanly and add little complexity:

- kurtosis or flatness versus pair separation ℓ;
- more elaborate intermittency diagnostics;
- local current-sheet statistics;
- local dissipation proxies;
- additional specialized alignment measures.

However, design the outputs and interfaces so these can be added later without a major rewrite.

Higher-order structure-function exponents ζ_p are part of the present task.
Higher-order increment-distribution diagnostics beyond the structure functions may be deferred.

==================================================
FINAL DELIVERABLES
==================================================

Provide:

1. A validated full-resolution extraction pipeline.

2. A validated nested-core finite-domain pair sampler.

3. A limited all-valid-pairs robustness implementation and comparison.

4. Arbitrary-order 3D structure-function calculations.

5. Three-direction conditional structure functions for:
   - ℓ_parallel;
   - ξ;
   - λ.

6. Comparisons using:
   - pair-scale local mean field;
   - subvolume-scale mean field.

7. Comparisons across:
   - B;
   - v_A;
   - v_A_ref;
   - u;
   - z_plus;
   - z_minus;
   - z_plus_ref;
   - z_minus_ref.

8. Synthetic validation tests.

9. Extraction-versus-cbin consistency checks.

10. CPU runtime and memory benchmarks.

11. Pilot scientific plots and tables.

12. A careful assessment of:
    - which trends appear to depend on dBB;
    - which may instead be explained by confounding environmental properties;
    - how results depend on L_sub;
    - which conclusions are robust enough to motivate an expanded sample.

13. A recommendation for the next production campaign:
    - which L_sub values to prioritize;
    - how many subvolumes per regime;
    - whether matched samples should be expanded;
    - whether any deferred diagnostics should be added.

14. A list of files created or modified.

15. A list of unresolved ambiguities.

==================================================
WORKING STYLE
==================================================

Work carefully and iteratively.

Before each substantial change:
- explain the intended method;
- identify assumptions;
- have a subagent stress-test the plan;
- validate on a small case.

After each substantial change:
- run targeted tests;
- compare against a slow trusted reference;
- inspect representative outputs;
- have an independent subagent look for mistakes;
- profile cost where relevant;
- document the result.

Do not launch large calculations prematurely.
Do not silently change scientific definitions.
Do not use periodic wrapping inside extracted subvolumes.
Do not conflate L_sub with ℓ.
Do not generalize the implementation for unrelated datasets.
Optimize for this fixed data structure and this family of simulations while keeping simulation parameters configurable.