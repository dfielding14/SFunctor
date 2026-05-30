# Prompt 1 Unresolved Questions

## Higher-Order Precision

AthenaK serializes cbin raw moments as float32 values. The catalog propagates
serialization, writer-summation, and merge-rounding bounds and stores
standardized skewness or kurtosis only when the corresponding central moment
is resolved above that bound. Rows with cancellation-dominated higher moments
remain useful for better-conditioned quantities but carry explicit flag bits.

## Quarantined Dynamo Channels

The `mhd_dynamo_ks` derivative channels remain excluded because the available
source contains a suspicious derivative-index expression and the channels are
not needed for the environmental census. Use them only after a dedicated
primitive-data validation.

## Broken SGS Products

The `mhd_sgs` products are excluded from validation, catalog construction,
verification, analysis, and pilot matching because they are not trustworthy.
Consequently, velocity moments, Favre statistics, Mach numbers, kinetic
partitions, pressure, mixed velocity-magnetic statistics, and off-diagonal
covariance tensors are unavailable in Prompt 1. Do not restore those columns
without a separate source fix and direct primitive-data validation.

The retained `u_mass_weighted_mean_* = <rho u_i>/<rho>` columns are exact
coarse means from primary conserved momentum. They are not velocity moments
and do not make a velocity dispersion reconstructable.

## Later Extraction Benchmark

Pilot rows include rank-file read volume and idealized pure-read times at 250,
500, and 1000 MiB/s. Those are planning estimates, not measured end-to-end
runtime. The later extraction stage should benchmark a cold-read pilot before
launching a production campaign.
