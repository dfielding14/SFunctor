# Phase 1:

audit the cbin outputs, build the domain-wide subvolume catalog, and propose a pilot sample

Your next task is to build a rigorous, efficient, and extensively validated census of the local coarse-grained environments in the full 10240^3 nonrelativistic MHD simulation domain using the cbin outputs.

Do not begin the expensive full-resolution 3D structure-function calculations yet. This task should stop after:
1. inspecting and validating the cbin data model;
2. determining exactly which coarse-grained quantities can be reconstructed;
3. building a comprehensive subvolume catalog;
4. exploring the one-dimensional and multi-dimensional distributions of the subvolume properties;
5. proposing a carefully designed pilot sample of subvolumes for the later full-resolution 3D analysis.

The scientific goal is to understand the full range of local environments sampled by the domain and to design a statistically and physically meaningful targeted sample for expensive follow-up calculations.

The first and most important organizing variable is

    dBB = δB / B_mean

with

    B_mean = |<B>_V|

    δB = sqrt( <|B - <B>_V|^2>_V )

where <...>_V denotes an average over a coarse-grained subvolume V.

Use the name `dBB` consistently in code, output tables, and documentation.

The broader goal is not limited to dBB. Build a comprehensive catalog of the local subvolume properties that can be reconstructed reliably from the available cbin data, then study how those properties correlate with one another and how they vary with the environmental averaging scale L_sub.

==================================================
USE SUBAGENTS ACTIVELY
==================================================

Use subagents extensively, but do not accept their conclusions without verification.

At minimum, delegate the following independent tasks:

1. Data-format audit:
   Inspect the cbin reader, file naming conventions, metadata, variable labels, coarsening conventions, and spatial indexing.

2. Mathematical reconstruction audit:
   Derive exactly which scalar, vector, variance, skewness, kurtosis, Mach-number, energy, and alignment-related quantities can and cannot be reconstructed from the stored moments, given that cross moments were not saved.

3. Performance review:
   Design the most computationally efficient CPU workflow for scanning the cbin products, merging neighboring cbin voxels, and producing the catalogs without unnecessary I/O or memory use.

4. Stress-test and adversarial review:
   Try to find mistakes in the proposed definitions, merge logic, indexing, edge handling, statistical formulas, and assumptions. Construct explicit failure cases.

5. Independent verification pass:
   After the implementation is complete, have a separate subagent review the resulting code and outputs without relying on the original implementation notes.

You remain responsible for reconciling disagreements, checking all conclusions yourself, and documenting any unresolved ambiguities.

==================================================
KNOWN DATA LAYOUT
==================================================

The full-resolution simulation domain is:

    10240^3 cells.

The full-resolution output was written using `single_file_per_rank`.

The rank-local high-resolution files are located under:

    /lustre/orion/ast207/proj-shared/dfielding/Production_plm/data/data_Turb_10240_beta25_dedt025_plm/bin

Each full-resolution rank-local block has dimensions:

    nx1 = 160
    nx2 = 320
    nx3 = 320

The cbin outputs contain coarsened cells with kernel sizes:

    40^3
    80^3
    160^3

For each stored scalar field q and each cbin voxel, the cbin outputs save:

    q_1st = mean(q)
    q_2nd = mean(q^2)
    q_3rd = mean(q^3)
    q_4th = mean(q^4)

No cross moments were saved.

A single full-resolution rank-local file corresponds to four 160^3 cbin voxels because:

    160 x 320 x 320 = 4 x 160^3.

The forcing scale is approximately:

    L_forcing = 5120 Δx.

The candidate environmental subvolume sizes to study are:

    L_sub / Δx = 80, 160, 320, 640, 1280.

Use exactly these values unless you identify a concrete technical reason that one cannot be assembled correctly. If a technical issue arises, report it explicitly rather than silently substituting a nearby value.

==================================================
IMPORTANT CONCEPTUAL DISTINCTION
==================================================

Keep the following scales explicit and separate throughout the analysis:

1. Environmental averaging scale:

    L_sub

   This is the size of the coarse-grained subvolume over which quantities such as dBB, sonic Mach number, density variance, and higher moments are measured.

2. Structure-function separation:

    ℓ

   This will be used only in the later full-resolution structure-function analysis.

The eventual scientific question will involve conditional statistics of the form:

    S_p(ℓ | environment measured at L_sub).

A central goal of the present task is to determine how strongly the environmental classification itself depends on L_sub.

==================================================
PHASE 1: INSPECT THE cbin FORMAT AND EXISTING CODE
==================================================

Before building catalogs, read the existing cbin-related code carefully and inspect representative cbin files.

Document:

- file naming conventions;
- directory layout;
- variable names;
- whether the stored quantities are primitive variables, conserved variables, or a mixture;
- array ordering;
- coordinate ordering;
- shape of each saved array;
- endianness and dtype;
- kernel-size metadata;
- spatial indexing conventions;
- whether the cbin voxels tile the global domain exactly;
- periodicity assumptions;
- whether all expected cbin outputs are present;
- whether there are missing, duplicated, or corrupt files;
- how cbin voxels map onto rank-local full-resolution files;
- how neighboring cbin voxels should be merged into larger subvolumes.

Do not rely on assumptions from comments alone. Check representative files directly.

Produce a concise data-layout report before proceeding.

==================================================
PHASE 2: DETERMINE WHAT CAN BE RECONSTRUCTED EXACTLY
==================================================

Because the cbin files store first through fourth raw moments of each saved scalar field but do not store cross moments, determine carefully which derived quantities can be reconstructed exactly, which can be reconstructed only under assumptions, and which cannot be reconstructed from cbin alone.

For a scalar q, use raw moments to calculate:

    μ_1 = <q>
    μ_2 = <q^2>
    μ_3 = <q^3>
    μ_4 = <q^4>

    variance(q) = μ_2 - μ_1^2

    σ_q = sqrt(variance(q))

    central_moment_3(q) = μ_3 - 3 μ_1 μ_2 + 2 μ_1^3

    central_moment_4(q) = μ_4 - 4 μ_1 μ_3 + 6 μ_1^2 μ_2 - 3 μ_1^4

    skewness(q) = central_moment_3(q) / σ_q^3

    kurtosis(q) = central_moment_4(q) / σ_q^4

    excess_kurtosis(q) = kurtosis(q) - 3.

Handle nearly zero variance robustly and explicitly.

For merged subvolumes, reconstruct the raw moments by volume-weighted averaging of the constituent cbin voxels before converting to central moments.

For vector quantities such as B and u, inspect which components are stored. If componentwise first and second moments are available, calculate quantities such as:

    <B>_V = ( <B_x>, <B_y>, <B_z> )

    B_mean = |<B>_V|

    <B^2>_V = <B_x^2> + <B_y^2> + <B_z^2>

    δB^2 = <B^2>_V - |<B>_V|^2

    dBB = δB / B_mean.

Similarly, if the velocity components are stored, calculate:

    <u>_V

    δu^2 = <u^2>_V - |<u>_V|^2.

Be explicit about the limitations caused by the absence of cross moments.

For example:
- componentwise variances can be reconstructed if the individual component moments exist;
- total vector fluctuation variance can be reconstructed by summing componentwise contributions;
- covariance tensors cannot be reconstructed without cross moments;
- many alignment statistics cannot be reconstructed without cross moments;
- moments of |B| are not generally equivalent to combinations of componentwise moments beyond limited cases;
- quantities involving products such as u_i B_i, rho u_i, or mixed density-field moments cannot be reconstructed unless those products were saved explicitly as their own scalar fields.

Inspect the saved labels carefully. If useful product fields were saved directly, use them. Do not assume they exist.

Produce a table with columns:

    quantity
    exact from cbin?
    required stored labels
    formula
    caveats

==================================================
PHASE 3: VALIDATE THE cbin RECONSTRUCTION
==================================================

Before scanning the full cbin dataset, validate the cbin-derived quantities against direct calculations from a small number of full-resolution rank-local files.

Choose representative regions and compare:

- first moments;
- second moments;
- third moments;
- fourth moments;
- variances;
- standard deviations;
- skewness;
- kurtosis;
- B_mean;
- B_rms;
- δB;
- dBB;
- mean velocity;
- δu;
- density statistics;
- any Mach numbers or energy-related quantities that can be reconstructed.

Validate:
1. a single 40^3 cbin voxel;
2. a single 80^3 cbin voxel;
3. a single 160^3 cbin voxel;
4. a merged region made from neighboring cbin voxels;
5. at least one merged region spanning boundaries between rank-local files.

Check indexing carefully in all three coordinate directions.

Report:
- absolute differences;
- relative differences;
- expected floating-point tolerance;
- any discrepancies;
- whether discrepancies arise from precision, weighting, indexing, or an incorrect assumption.

Do not proceed to the full census until the reconstruction is validated.

==================================================
PHASE 4: BUILD THE COMPREHENSIVE SUBVOLUME CATALOG
==================================================

Build catalogs for:

    L_sub / Δx = 80, 160, 320, 640, 1280.

Use the cbin products only. Do not load the full-resolution 3D rank-local files during the domain-wide census except for the limited validation tests described above.

Design the implementation for CPU efficiency.

Prefer:
- minimal I/O;
- streaming or chunked reading where useful;
- reuse of the smallest sufficient stored cbin scale;
- efficient aggregation of neighboring voxels;
- prefix-sum, summed-volume-table, hierarchical merge, or equivalent strategies where appropriate;
- explicit profiling;
- compact catalog outputs.

Do not optimize blindly. Compare candidate strategies and justify the chosen approach.

Determine whether the catalog should use:
- non-overlapping subvolumes;
- overlapping sliding windows;
- both.

A reasonable default is:
- use non-overlapping subvolumes for the first complete census and for interpretable approximately independent statistics;
- optionally add an overlapping-window or offset-grid robustness study if computationally cheap and scientifically useful.

Do not allow the overlapping windows to explode the cost without clear value.

For every catalog entry, store:
- global subvolume identifier;
- L_sub;
- global coordinate bounds;
- center coordinate;
- list or compact mapping of constituent cbin voxels;
- mapping to the required full-resolution rank-local files for later extraction;
- all reconstructed environmental properties;
- any validity flags;
- any caveats.

==================================================
COMPREHENSIVE ENVIRONMENTAL PROPERTIES
==================================================

Calculate every useful quantity that can be reconstructed reliably from the available saved fields and moments.

At minimum, attempt to calculate the following.

A. Magnetic-field statistics

    <B>_V
    B_mean = |<B>_V|
    B_rms = sqrt(<B^2>_V)
    δB = sqrt(<|B - <B>_V|^2>_V)
    dBB = δB / B_mean

Also store bounded or complementary measures such as:

    B_mean^2 / <B^2>_V
    δB^2 / <B^2>_V.

These are important because dBB can become extremely large when B_mean is small. Distinguish:
- genuinely large fluctuations;
- small local mean field caused by cancellation;
- both effects together.

If the component moments are available, store componentwise means, variances, skewnesses, and kurtoses.

B. Velocity statistics

    <u>_V
    u_rms
    δu = sqrt(<|u - <u>_V|^2>_V)

If the component moments are available, store componentwise means, variances, skewnesses, and kurtoses.

C. Density statistics

    <rho>_V
    variance(rho)
    σ_rho / <rho>_V
    skewness(rho)
    kurtosis(rho)

If reconstructable from saved fields, also calculate useful statistics of:

    ln(rho).

Do not infer ln(rho) moments from rho moments unless mathematically justified. Use them only if ln(rho) itself was saved or can be calculated from an appropriately available field.

D. Sonic Mach number

Calculate a local sonic Mach number if the required sound-speed information is available:

    M_s = δu / c_s

For an isothermal simulation, use the known constant c_s and document it.

If the sound speed varies, inspect the available fields and document the precise definition used. Do not silently choose between volume-weighted, mass-weighted, or other conventions.

E. Alfvénic quantities

In the code magnetic units:

    v_A = B / sqrt(rho).

Attempt to calculate useful local Alfvén-speed and Alfvén-Mach-number summaries, but distinguish exact quantities from approximations.

Potential definitions include:

    v_A,mean = B_mean / sqrt(<rho>_V)

    v_A,rms-like = sqrt(<B^2>_V / <rho>_V)

    M_A,mean = δu / v_A,mean.

Because cross moments and mixed moments were not saved, quantities such as:

    <B^2 / rho>

may not be reconstructable exactly unless an appropriate field was saved directly.

Implement only quantities that are exact or clearly labeled approximations. Document the distinction.

F. Energetic quantities

Attempt to reconstruct:
- magnetic energy density;
- kinetic energy density;
- magnetic-to-kinetic energy ratio;
- turbulent kinetic energy relative to mean-flow kinetic energy;
- other physically useful energy ratios.

Be careful: exact kinetic energy may require saved conserved variables or mixed moments. Determine what is possible from the stored labels.

G. Higher moments

Use the saved third- and fourth-order moments comprehensively.

For every useful scalar field and vector component with valid stored moments, calculate:
- variance;
- standard deviation;
- skewness;
- kurtosis;
- excess kurtosis.

These higher moments are a major part of the scientific census, not an optional afterthought.

H. Cross helicity, Elsasser imbalance, and residual energy

Investigate whether the saved labels are sufficient to reconstruct:
- cross helicity;
- Elsasser imbalance;
- residual energy;
- any related normalized quantities.

Do not assume they are available.

Because no generic cross moments were saved, they may be impossible to reconstruct exactly unless the relevant products or energies were saved explicitly.

If they are reconstructable, include them.
If they are not reconstructable, state exactly what is missing.

I. Additional useful properties

Inspect the stored labels and propose any additional scientifically useful subvolume properties that can be reconstructed reliably.

Do not generate a large set of opaque features merely because they are available. Favor interpretable physical quantities, but be comprehensive.

==================================================
PHASE 5: EXPLORE THE DOMAIN-WIDE DISTRIBUTIONS
==================================================

For each L_sub, analyze the distribution of the catalog properties.

A. One-dimensional distributions

At minimum, plot and summarize:
- dBB;
- B_mean;
- B_rms;
- δB;
- B_mean^2 / <B^2>;
- δB^2 / <B^2>;
- δu;
- M_s;
- any reliable M_A variants;
- <rho>;
- σ_rho / <rho>;
- skewness and kurtosis of key fields;
- magnetic-to-kinetic energy ratios if reconstructable;
- cross-helicity, Elsasser-imbalance, or residual-energy quantities if reconstructable.

Use logarithmic axes where appropriate.
Report robust quantiles, not only means and standard deviations.

B. Two-dimensional correlations

At minimum, examine:

    dBB vs B_mean
    dBB vs δB
    dBB vs B_rms
    dBB vs M_s
    dBB vs M_A variants
    dBB vs <rho>
    dBB vs σ_rho / <rho>
    dBB vs key skewnesses
    dBB vs key kurtoses
    dBB vs magnetic-to-kinetic energy ratio

Also inspect whether large dBB is driven primarily by:
- unusually large δB;
- unusually small B_mean;
- both.

C. Scale dependence

This is essential.

Compare the distributions and correlations across:

    L_sub / Δx = 80, 160, 320, 640, 1280.

Quantify how:
- the dBB distribution changes with L_sub;
- the median and quantiles shift;
- the correlations with M_s, M_A, density variance, and higher moments evolve;
- environmental classifications change as the averaging scale changes;
- the same spatial region moves between dBB regimes as L_sub changes.

Construct cross-scale comparisons where possible.

Do not treat dBB as a single scale-independent label.

D. Higher-dimensional analysis

After the interpretable one- and two-dimensional plots are understood, use:
- correlation matrices;
- conditional medians and quantile bands;
- rank correlations;
- matched comparisons;
- PCA or clustering only if they add clear value.

Do not use sophisticated dimensionality reduction as a substitute for physical interpretation.

==================================================
PILOT-SAMPLE DESIGN FOR THE LATER 3D ANALYSIS
==================================================

Use the cbin census to propose a pilot set of full-resolution 3D subvolumes for later structure-function calculations.

The initial pilot should be modest and computationally realistic.

Start by proposing approximately four representative subvolumes per scientifically useful dBB regime. Do not hard-code the regime boundaries before inspecting the actual dBB distributions.

Consider:
- physically motivated thresholds such as dBB << 1, dBB ~ 1, and dBB >> 1;
- quantile-based bins if extreme regimes are rare;
- multiple L_sub values;
- computational cost of extracting and loading each full-resolution region;
- statistical independence and spatial separation;
- avoiding a sample dominated by neighboring or nearly redundant regions.

The pilot sample should include both:

1. Representative examples:
   Typical subvolumes from each dBB regime.

2. Controlled matched comparisons:
   Pairs or groups of subvolumes with different dBB but similar values of possible confounding variables such as:
   - M_s;
   - M_A;
   - mean density;
   - density variance;
   - B_rms;
   - δB;
   - magnetic-to-kinetic energy ratio.

This is important because an apparent dBB trend may actually be caused by another correlated property.

Also identify:
- extreme outliers;
- regions with anomalously small B_mean;
- regions with genuinely large δB;
- regions that are unusual in multiple properties.

For each proposed pilot subvolume, report:
- L_sub;
- global coordinate bounds;
- center coordinate;
- dBB;
- B_mean;
- δB;
- B_rms;
- M_s;
- reliable M_A variants;
- density statistics;
- higher moments;
- energy ratios if available;
- whether it is representative, matched, or an outlier;
- required full-resolution rank-local files;
- estimated memory footprint for full-resolution extraction;
- any technical extraction caveats.

==================================================
DO NOT YET IMPLEMENT THE EXPENSIVE 3D ANALYSIS
==================================================

Stop after:
- the data-layout report;
- the reconstructability table;
- cbin validation;
- catalog construction;
- distribution and correlation analysis;
- scale-dependence analysis;
- pilot-sample proposal;
- estimated extraction costs;
- recommendations for the next stage.

Do not yet:
- assemble large full-resolution 3D subvolumes;
- run the full-resolution 3D structure-function calculator;
- optimize the finite-domain pair sampler;
- launch a large production campaign.

==================================================
DELIVERABLES
==================================================

Provide:

1. A concise architecture and data-layout report.

2. A table of quantities that are:
   - exactly reconstructable;
   - approximately reconstructable;
   - not reconstructable from cbin.

3. Validation results comparing cbin reconstruction against direct high-resolution calculations.

4. A CPU-efficient catalog-building implementation.

5. Catalog files for each L_sub.

6. One-dimensional distribution plots.

7. Two-dimensional correlation plots.

8. Cross-scale comparisons.

9. A concise scientific summary of what controls dBB across the domain.

10. A proposed pilot sample for the full-resolution 3D analysis.

11. Memory and runtime estimates for the later extraction stage.

12. A list of unresolved questions or ambiguities.

13. A list of all files created or modified.

==================================================
WORKING STYLE
==================================================

Work carefully and iteratively.

Before each substantial implementation decision:
- state the intended method;
- identify assumptions;
- ask a subagent to stress-test the reasoning;
- validate on a small example.

After each substantial change:
- run targeted tests;
- compare against direct calculations where possible;
- inspect representative outputs;
- have an independent subagent look for mistakes;
- document the result.

Do not silently repair discrepancies by changing definitions.
Do not hide unavailable quantities behind approximations.
Do not process more data than necessary.
Do not generalize the code for unrelated datasets.
Optimize for this family of simulations and this fixed data structure while keeping simulation parameters configurable.
