# Phase 1 Reconstructability Audit

For merged subvolumes, average constituent raw moments first. Do not average
child variances, skewnesses, kurtoses, or `dBB`.

| Quantity | Status | Required labels and definition | Caveat |
|---|---|---|---|
| Scalar moments of `dens`, `mom1..3`, `ener`, `bcc1..3` | exact | each scalar's `_1st.._4th` fields | Momentum statistics are not velocity statistics. |
| Scalar variance and sigma | exact subject to precision flags | standard raw-to-central formulas | AthenaK serializes float32 raw moments. Bounds include serialization, writer-summation, and merge-rounding uncertainty. |
| Scalar skewness, kurtosis, excess kurtosis | available only when numerically resolved | standard raw-to-central formulas with interval-normalized numerator and variance-denominator bounds | Cancellation-dominated standardized moments are stored as `NaN` with explicit flag bits rather than reported as trustworthy values. |
| `<B>`, `B_mean`, `B_rms`, `deltaB`, `dBB` | exact subject to denominator flags | component first and second moments | `dBB` is undefined when `B_mean` is numerically zero. |
| `B_mean^2 / <B^2>` and `deltaB^2 / <B^2>` | exact | magnetic component moments | These bounded complements diagnose large `dBB`. |
| `<rho>`, density variance, `sigma_rho/<rho>`, density skewness and kurtosis | exact | `dens_1st..4th` | Near-zero denominators are flagged. |
| `ln(rho)` moments | unavailable | no saved `logrho` field | Do not infer them from density moments. |
| Volume-weighted `<u>` and `delta u` | unavailable | would require primitive velocity moments | Main cbin files contain momentum. |
| Mass-weighted mean velocity `<rho u_i>/<rho>` | exact coarse statistic | `mom1..3_1st / dens_1st` | Stored as `u_mass_weighted_mean_*`; this does not provide a velocity dispersion. |
| Magnetic energy | exact | `0.5 <B^2>` | Mean and fluctuating partitions are also exact. |
| Kinetic energy | unavailable | would require reliable `<rho u^2>` or equivalent mixed moments | Do not infer it from separate momentum and density moments. |
| Mean-flow and turbulent kinetic partitions | unavailable | would require reliable velocity or Favre mixed moments | SGS products are excluded. |
| Mean pressure and coarse sound speed | unavailable | pressure requires a kinetic-energy subtraction from total energy | `ener` alone is insufficient. |
| Sonic Mach number | unavailable | would require velocity dispersion and sound speed | SGS products are excluded. |
| `B_mean/sqrt(<rho>)`, `sqrt(<B^2>/<rho>)` | approximate physical summaries | exact evaluations of coarse means | Stored as `vA_mean_proxy` and `vA_rms_like_proxy`. |
| Alfvén Mach numbers | unavailable | would require a reliable velocity dispersion | Do not divide momentum dispersion by an Alfvén-speed proxy. |
| Diagonal magnetic variances | exact subject to precision flags | `bcc1..3_1st..2nd` | Their sum gives `deltaB^2`. |
| Off-diagonal magnetic covariance tensor | unavailable | would require reliable `<B_i B_j>` cross moments | Separate component moments are insufficient. |
| Reynolds tensor | unavailable | would require reliable velocity cross moments | SGS products are excluded. |
| Volume-averaged `<u dot B>` | unavailable | would require reliable mixed velocity-magnetic moments | SGS products are excluded. |
| Standard cross helicity, Elsasser imbalance, residual energy | unavailable exactly | would require pointwise Alfvén-unit products such as `<B^2/rho>` | Do not hide missing terms behind unlabeled approximations. |
| Generic alignment statistics | unavailable exactly | missing nonlinear cross moments | SGS products are excluded. |

The `mhd_sgs` and `mhd_dynamo_ks` products are excluded. They are not needed
for the primary-moment census and must not be used as reconstruction inputs.

## Precision Flags

Each scalar `*_moment_flags` column uses a bit mask:

| Bit | Meaning |
|---:|---|
| `0x01` | variance was slightly negative within the propagated error bound and was clamped to zero |
| `0x02` | variance was materially inconsistent and was set to `NaN` |
| `0x04` | variance is unresolved above its propagated error bound |
| `0x08` | fourth central moment was materially inconsistent and was set to `NaN` |
| `0x10` | third central moment is cancellation-dominated; skewness is unavailable |
| `0x20` | fourth central moment was slightly negative within its propagated error bound and was clamped |
| `0x40` | fourth central moment is cancellation-dominated; kurtosis is unavailable |
| `0x80` | raw input or propagated bound was non-finite |

The catalog also stores an aggregate `catalog_validity_flags` bit mask so
downstream selections can audit exclusions without hiding the underlying
per-quantity reason.
