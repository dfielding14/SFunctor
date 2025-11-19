# Legacy Top-Level Scripts

These SLURM submission helpers used to live in the repository root. They’re
still referenced in some documentation/logs, so we’ve parked them in
`scripts/root/` rather than deleting them outright:

- `run_sf_analysis_frontier*.sh` – Frontier job templates for structure-function
  analysis.
- `run_distributed_analysis*.sh` – Older multi-node submission scripts.
- `run_extraction.sh`, `setup_frontier_env.sh` – environment/bootstrap helpers.

New work uses the scripts under `job_scripts/` and `scripts/production/`.
