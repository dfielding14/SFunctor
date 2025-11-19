# Job Script Layout

The job scripts were previously in one flat directory. They are now grouped
by purpose to keep things manageable:

- `production/` – SLURM submission scripts that launch structure–function
  campaigns (`job_*.sh`, `run_distributed_analysis_frontier_*.sh`, etc.).
- `utilities/` – helper workflows for monitoring, resubmitting or
  aggregating runs (`resume_*`, `submit_all_jobs*.sh`, `collect_*`, …).
- `tests/` – sanity-check runners and the older `test_scripts/` helpers.

Update any personal notes or automation to point at these new paths. The
scripts themselves are unchanged aside from their locations.
