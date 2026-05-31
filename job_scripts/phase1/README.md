# Andes Compute Ledger

Use `scripts/phase1/update_compute_ledger.py` to keep the phase1 workflow's
allocated node-hours under the 5000 node-hour budget. The helper writes these
files inside the selected project results directory:

- `compute_ledger.csv`: persistent machine-readable ledger.
- `compute_budget_summary.md`: concise budget, active-job, exposure, and
  accounting-warning summary.

The helper never submits, cancels, or modifies Slurm jobs. It queries `sacct`,
`squeue`, and `scontrol show job` with argument arrays and records only
top-level allocations. Child steps such as `.batch`, `.extern`, and `srun`
steps are excluded.

## Andes Queue Convention

The existing Andes production scripts use allocation `AST207`, explicit
partition `#SBATCH -p batch`, and 32 CPUs per task. A live scheduler check on
May 30, 2026 exposed only QOS `normal`; no debug partition or debug QOS was
available. Older scripts containing `#SBATCH -q debug` are stale for the
current configuration. Confirm the current cluster configuration with
`sinfo`, `scontrol show partition`, `sacctmgr show qos`, and
`squeue -u "$USER"` before each submission.

## Register Before Submission

Register each planned job before running `sbatch`. Use a unique output
directory and record the smallest useful run for the current question:

```bash
python scripts/phase1/update_compute_ledger.py \
  --results-dir Results/phase1 \
  --no-refresh \
  --register-planned \
  --job-name sf_smoke_640 \
  --purpose "Smoke-test restartable analysis on one representative input" \
  --script-path job_scripts/production/run_distributed_analysis_andes_restart.sh \
  --partition batch \
  --qos normal \
  --nodes 1 \
  --cpus 32 \
  --wall-time 00:15:00 \
  --expected-node-hours 0.1 \
  --output-dir Results/phase1/sf_smoke_640
```

The command prints a planned record ID. After manually submitting the job,
attach the Slurm allocation ID:

```bash
python scripts/phase1/update_compute_ledger.py \
  --results-dir Results/phase1 \
  --link-planned PLANNED_RECORD_ID \
  --job-id SLURM_JOB_ID
```

Refresh known allocations at any time:

```bash
python scripts/phase1/update_compute_ledger.py --results-dir Results/phase1
```

Import a top-level allocation that was not registered in advance:

```bash
python scripts/phase1/update_compute_ledger.py \
  --results-dir Results/phase1 \
  --job-id SLURM_JOB_ID
```

Use `--discover-active` only when active jobs owned by the selected user should
be imported automatically. Without it, untracked `squeue` jobs are reported as
warnings but are not added to the workflow ledger.

## Accounting Semantics

Completed, failed, cancelled, and running allocations consume:

```text
allocated nodes * actual elapsed wall-clock hours
```

Queued and locally planned jobs consume zero node-hours. The summary reports
pending maximum *additional* exposure: the full requested maximum for planned
or queued jobs, and the unconsumed part of the requested maximum for running
jobs. This avoids double-counting runtime already charged to a running job.

If Slurm accounting is unavailable or a scheduler record is incomplete, the
helper preserves prior ledger data and adds explicit accounting flags. Review
those flags before submitting any nontrivial job. If unfinished work could
exceed the remaining 5000 node-hour budget, the summary and terminal output
show an explicit budget warning; registering a proposal does not authorize its
submission.

## Phase 1 Wrappers

- `run_phase1_andes.sh`: direct reconstruction validation, streamed catalog
  build, and analysis actions selected with `ACTION`.
- `run_phase1_verify_andes.sh`: read-only independent verification of
  completed catalogs before analysis is authorized.
