# CPU Benchmarking

Run benchmarks after correctness tests and after Numba warm-up:

```bash
venv_frontier/bin/python -m pytest -q
/usr/bin/time -v venv_frontier/bin/python scripts/benchmarks/benchmark_cpu.py \
  --slice slice_data/Turb_320_beta100_dedt025_plm_axis3_slice0_file0000.npz \
  --output ../SFunctor_artifacts_20260530_followup/cpu_benchmark.json
```

The benchmark reports:

- isolated origin-sampler timing for population-sized no-replacement sampling
  versus `O(k)` replacement sampling;
- dense full-histogram-per-displacement timing versus one-`ell` slab timing;
- slow loop versus vectorized strict-directional timing and numerical
  agreement;
- `/usr/bin/time -v` peak RSS for the complete driver.

## Performance Conventions

- Warm JIT kernels before timing.
- Record Python and NumPy versions, host, CPU allocation, slice shape, stride,
  process count, histogram shape, displacement count, and samples per offset.
- Use a fixed random seed.
- Compare scientific outputs before accepting a speedup.
- For shared-memory runs, sweep process counts instead of assuming all cores
  help. The kernel has random-access and memory-bandwidth pressure.
- Keep benchmark outputs in a sibling results directory outside the
  repository unless intentionally adding a recorded report.

## Scaling Notes

The legacy path now samples origins with replacement. This is an unbiased
Monte Carlo estimator and makes sampling cost proportional to requested draws
instead of full slice area. At production draw fractions, duplicate draws are
small; convergence tests should still vary seed and sample count.

The shared legacy path requests a one-radial-bin slab from each per-offset
kernel and streams worker-batch reduction. This removes dense temporary
histograms per displacement and bounds parent reduction memory. Each worker
still owns one dense batch accumulator; measure worker count against available
memory.

The extraction reader now builds a file-derived rank manifest with a
payload-skipping metadata scan, validates the exact level-0 layout used by the
2-D extractor, and reads the union of central and derivative-neighbor blocks
grouped by rank file. Profile manifest scanning and grouped reads separately on
the target filesystem before choosing 3-D chunk sizes.

## Recorded Local Result

Run on May 30, 2026 with the project `venv_frontier` Python and the checked-in
driver, using the axis-3 `320 x 320` extracted slice loaded at stride `8`
(`40 x 40` benchmark plane):

| benchmark | baseline | corrected path | speedup |
| --- | ---: | ---: | ---: |
| legacy dense temporary vs one-`ell` slab, 8 offsets | `0.15784 s` | `0.00849 s` | `18.60x` |
| strict loop oracle vs vectorized directional path, 4 offsets | `3.15177 s` | `0.03725 s` | `84.60x` |
| Python no-replacement vs replacement origin draw, `2560^2` population and `2000` draws | `1.3031e-4 s` | `3.1029e-5 s` | `4.20x` |

The legacy temporary shrank from `67.72 MiB` to `1.41 MiB` in the benchmark
shape. The directional comparison had exact counts and agreed sums within the
driver tolerance `rtol=atol=1e-13`. Complete-driver peak RSS from
`/usr/bin/time -v` was `375680 KiB`.

An independent Numba sampler review measured the more important large-grid
behavior of the old internal `choice(..., replace=False)` call: at `5120^2`
cells and `2000` draws it took about `2.07 s` per displacement and peaked near
`356 MiB` RSS, while replacement draws were about `6e-5 s`. The corrected
Numba kernels use replacement draws.
