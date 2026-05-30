==================================================
ANDES EXECUTION POLICY AND COMPUTE BUDGET
==================================================

You will run this work on the Andes supercomputer using Slurm.

You have a total compute budget of:

    5000 allocated node-hours

for the full workflow.

This budget applies cumulatively across:
- exploratory runs;
- validation tests;
- profiling jobs;
- failed or cancelled jobs that consumed allocated runtime;
- catalog-generation jobs;
- later full-resolution 3D extraction jobs;
- production structure-function calculations.

Queued time does not count toward the budget. Actual allocated runtime does count, including runtime consumed by jobs that fail.

You must track compute use carefully as you go and avoid unnecessary expensive runs.

==================================================
USE THE EXISTING ANDES JOB SCRIPT AS A TEMPLATE
==================================================

Use the existing Andes Slurm scripts as templates for:
- allocation settings;
- environment setup;
- module loading;
- virtual-environment activation;
- logging;
- result-directory organization;
- job naming;
- stdout and stderr capture;
- timing summaries;
- email notification settings.

A prior heavy-analysis script is available as an example. Inspect it before writing new job scripts.

The existing workflow uses the AST207 allocation and CPU jobs on Andes. Preserve the existing environment setup unless there is a concrete reason to change it.

Do not run computationally heavy analysis directly on a login node.

Use small login-node commands only for:
- inspecting files;
- reading metadata;
- checking directory structure;
- writing scripts;
- submitting jobs;
- checking job status;
- lightweight validation that is clearly safe.

==================================================
DEBUG QUEUE AND NORMAL QUEUE
==================================================

You may use either:

1. The debug queue:
   - intended for short validation runs, smoke tests, profiling tests, and rapid iteration;
   - usually has shorter wait times;
   - only one debug job should be queued or running at a time.

2. The normal queue:
   - intended for longer or heavier analyses;
   - use it for catalog scans, scaling tests, and production work once the workflow is validated.

Before the first submission, inspect the current Andes Slurm partition configuration and the existing job scripts to confirm the exact queue syntax.

Use commands such as:

    sinfo
    scontrol show partition
    squeue -u "$USER"

Do not assume silently that the cluster default partition is correct.

If the local convention is that the debug queue is selected by adding a debug-partition directive and the normal queue is selected by removing that override, follow that convention.

If the existing normal-production scripts explicitly use:

    #SBATCH -p batch

retain that explicit setting rather than relying on an undocumented default.

Document the exact convention you determine.

Never queue multiple debug jobs simultaneously.

For normal-queue production work, submit jobs in controlled stages rather than flooding the scheduler with a large unvalidated campaign.

==================================================
NODE-HOUR ACCOUNTING
==================================================

Track allocated node-hours, not CPU-core hours.

For a completed job:

    node_hours = allocated_nodes x elapsed_wall_clock_hours.

Examples:

    1 node x 2 hours      = 2 node-hours
    16 nodes x 30 minutes = 8 node-hours
    128 nodes x 4 hours   = 512 node-hours.

For running jobs, estimate consumed node-hours using elapsed runtime so far.

For failed or cancelled jobs, count the actual allocated runtime consumed before termination.

Do not count queued-but-not-started jobs.

Be careful not to double-count Slurm job steps. When using `sacct`, distinguish:
- the top-level allocation;
- child steps such as `.batch`, `.extern`, or `srun` steps.

Count the allocation once.

==================================================
COMPUTE LEDGER
==================================================

Create and maintain a persistent compute ledger in the project results area.

Use a machine-readable file such as:

    compute_ledger.csv

and a concise human-readable summary such as:

    compute_budget_summary.md

At minimum, record:

    timestamp
    Slurm job ID
    job name
    purpose
    script path
    git commit or code-version identifier
    partition
    node count
    allocated CPUs
    requested wall time
    actual elapsed time
    job state
    exit code
    node-hours consumed
    cumulative node-hours consumed
    remaining node-hour budget
    output directory
    notes

Also maintain a record of planned jobs that have not yet completed, including their maximum possible node-hour cost based on requested nodes and wall time.

Use Slurm accounting tools such as:

    sacct
    squeue
    scontrol show job

to update the ledger from actual scheduler records.

Write or reuse a small helper script to:
- query completed and running jobs;
- update the ledger;
- avoid duplicate entries;
- calculate cumulative use;
- calculate remaining budget;
- summarize pending exposure;
- flag jobs with missing accounting data.

==================================================
PRE-SUBMISSION CHECK
==================================================

Before submitting any nontrivial job:

1. State the scientific or technical purpose of the job.

2. Estimate:

       requested_nodes
       requested_wall_time
       maximum_node_hours
       expected_node_hours

3. Report:

       cumulative_node_hours_used
       remaining_budget
       pending_maximum_node_hours
       projected_remaining_budget_after_completion.

4. Confirm that the run is the smallest useful test for the question being asked.

5. Confirm that the code path has passed smaller validation tests where appropriate.

6. Confirm that outputs will be written to a unique, clearly named directory.

7. Confirm that the job is restartable or that duplicated work will be minimized if it fails.

Do not submit a job if its maximum possible cost would cause the cumulative total to exceed 5000 node-hours.

If a scientifically justified run would exceed the remaining budget, stop and report:
- the proposed run;
- its estimated cost;
- the expected scientific value;
- cheaper alternatives;
- the additional budget that would be required.

Wait for explicit approval before exceeding the budget.

==================================================
STAGED EXECUTION STRATEGY
==================================================

Use the following progression.

Stage 1: local inspection
- inspect formats;
- inspect representative files;
- validate indexing;
- perform only lightweight operations.

Stage 2: debug-queue smoke tests
- run the smallest useful cases;
- test file readers;
- test reconstruction;
- test merging;
- test output paths;
- test failure handling;
- validate timing and memory instrumentation.

Stage 3: debug-queue profiling tests
- measure CPU time;
- measure memory use;
- identify bottlenecks;
- test scaling on a small number of representative inputs.

Stage 4: limited normal-queue pilot
- run a small but scientifically meaningful sample;
- check correctness;
- check runtime estimates;
- check output integrity;
- update the compute ledger.

Stage 5: production campaign
- submit only after the pilot is validated;
- process in batches;
- review outputs and budget use between batches;
- stop early if the scientific return is poor or a cheaper strategy is available.

Do not jump directly to a large production campaign.

==================================================
EFFICIENCY REQUIREMENTS
==================================================

Because the simulation domain is extremely large, minimize unnecessary I/O and repeated work.

Prefer:
- cbin-derived quantities where full-resolution data are unnecessary;
- small representative tests before broad scans;
- caching reusable metadata;
- resumable processing;
- unique output directories;
- checksums or completion markers;
- avoiding repeated reads of the same large files;
- extracting only required fields;
- extracting only required spatial regions;
- profiling before optimization;
- profiling again after substantial changes.

When a job fails:
- inspect the failure;
- record consumed node-hours;
- determine the cause;
- make the smallest corrective change;
- rerun a reduced test before resubmitting a large job.

==================================================
ONGOING REPORTING
==================================================

At each major milestone, provide a brief compute-budget update with:

    total node-hours used
    node-hours remaining
    node-hours consumed since the last update
    active jobs
    queued jobs
    completed jobs
    failed jobs
    estimated cost of the next stage
    any recommended change in execution strategy.

At the end of each prompt, include:
- the final compute ledger;
- cumulative node-hours used;
- remaining budget;
- a breakdown by task category;
- a list of expensive runs and what each accomplished;
- recommended compute allocation for the next prompt.

==================================================
SUBAGENT REVIEW OF COMPUTE USE
==================================================

Assign at least one subagent to act as an independent compute-budget reviewer.

That subagent should periodically inspect:
- the compute ledger;
- submitted job scripts;
- queue choices;
- scaling estimates;
- duplicated work;
- output reuse;
- opportunities to reduce cost;
- whether a large planned run is justified by the existing evidence.

Before launching any especially expensive batch, ask the compute-budget reviewer to challenge the plan and propose cheaper alternatives.

You remain responsible for the final decision.

==================================================
IMPORTANT CONSTRAINTS
==================================================

- Total budget: 5000 allocated node-hours.
- CPU execution only unless explicitly instructed otherwise.
- Andes Slurm jobs only for heavy work.
- No heavy analysis on login nodes.
- Only one debug job queued or running at a time.
- Normal-queue jobs should be submitted in staged batches.
- Track actual allocated node-hours continuously.
- Never exceed the budget without explicit approval.
- Prefer correctness, validation, and efficient experimental design over brute-force computation.