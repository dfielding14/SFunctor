#!/usr/bin/env python3
"""Maintain a persistent Slurm node-hour ledger for the Andes workflow.

The script intentionally uses only the Python standard library.  It does not
submit or cancel jobs.  Scheduler queries are made without a shell so that job
IDs and user names are never interpolated into command strings.
"""

import argparse
import collections
import csv
import datetime
import fcntl
import getpass
import os
import re
import subprocess
import sys
import tempfile
import uuid


DEFAULT_BUDGET_NODE_HOURS = 5000.0
LEDGER_FILENAME = "compute_ledger.csv"
SUMMARY_FILENAME = "compute_budget_summary.md"
LOCK_FILENAME = ".compute_ledger.lock"

CSV_FIELDS = [
    "timestamp",
    "last_updated",
    "record_id",
    "record_type",
    "slurm_job_id",
    "job_name",
    "purpose",
    "script_path",
    "code_version",
    "partition",
    "qos",
    "requested_nodes",
    "node_count",
    "allocated_cpus",
    "requested_wall_time",
    "requested_wall_time_seconds",
    "actual_elapsed_time",
    "elapsed_seconds",
    "job_state",
    "exit_code",
    "node_hours_consumed",
    "cumulative_node_hours_consumed",
    "remaining_node_hour_budget",
    "planned_max_node_hours",
    "expected_node_hours",
    "pending_max_additional_node_hours",
    "output_directory",
    "submit_time",
    "start_time",
    "end_time",
    "scheduler_sources",
    "accounting_flags",
    "notes",
]

TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}
NOT_STARTED_STATES = {
    "PENDING",
    "PLANNED",
}
UNKNOWN_DURATION_VALUES = {
    "",
    "N/A",
    "NONE",
    "NOT_SET",
    "PARTITION_LIMIT",
    "UNLIMITED",
    "UNKNOWN",
}
JOB_ID_RE = re.compile(r"^[0-9][0-9A-Za-z_+\[\],%:-]*$")
SCONTROL_FIELD_RE = re.compile(
    r"(?:^|\s)([A-Za-z][A-Za-z0-9_:]*)=(.*?)(?=\s+[A-Za-z][A-Za-z0-9_:]*=|$)"
)

CommandResult = collections.namedtuple("CommandResult", "ok stdout stderr")


def utc_timestamp():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def parse_int(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in UNKNOWN_DURATION_VALUES:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def parse_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_duration_seconds(value):
    """Parse Slurm duration syntax: [days-]hours:minutes:seconds or minutes:seconds."""
    if value is None:
        return None
    text = str(value).strip()
    if text.upper() in UNKNOWN_DURATION_VALUES:
        return None
    if text.isdigit():
        return int(text)

    days = 0
    if "-" in text:
        day_text, text = text.split("-", 1)
        if not day_text.isdigit():
            return None
        days = int(day_text)

    parts = text.split(":")
    if len(parts) == 3:
        hours_text, minutes_text, seconds_text = parts
    elif len(parts) == 2:
        hours_text = "0"
        minutes_text, seconds_text = parts
    else:
        return None

    if not (hours_text.isdigit() and minutes_text.isdigit() and seconds_text.isdigit()):
        return None
    minutes = int(minutes_text)
    seconds = int(seconds_text)
    if minutes >= 60 or seconds >= 60:
        return None
    return days * 86400 + int(hours_text) * 3600 + minutes * 60 + seconds


def format_duration(seconds):
    value = parse_int(seconds)
    if value is None:
        return ""
    days, remainder = divmod(max(0, value), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    body = "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)
    if days:
        return "{}-{}".format(days, body)
    return body


def format_hours(value):
    number = parse_float(value)
    if number is None:
        return ""
    text = "{:.6f}".format(number).rstrip("0").rstrip(".")
    return text if text else "0"


def normalize_state(value):
    if not value:
        return "UNKNOWN"
    return str(value).strip().split()[0].rstrip("+").upper()


def is_terminal(state):
    return normalize_state(state) in TERMINAL_STATES


def is_top_level_job_id(job_id):
    return bool(job_id and "." not in job_id and JOB_ID_RE.match(job_id))


def require_job_id(job_id):
    if not is_top_level_job_id(job_id):
        raise ValueError(
            "invalid top-level Slurm job ID {!r}; child steps such as '.batch' are not accepted".format(
                job_id
            )
        )
    return job_id


def split_flags(value):
    return {item for item in str(value or "").split(";") if item}


def join_flags(flags):
    return ";".join(sorted(flags))


def new_record(record_type, now=None):
    record = {field: "" for field in CSV_FIELDS}
    record["timestamp"] = now or utc_timestamp()
    record["last_updated"] = record["timestamp"]
    record["record_id"] = str(uuid.uuid4())
    record["record_type"] = record_type
    record["job_state"] = "PLANNED" if record_type == "planned" else "UNKNOWN"
    return record


def run_command(args, timeout):
    try:
        completed = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CommandResult(False, "", "timed out after {} seconds".format(timeout))
    except OSError as exc:
        return CommandResult(False, "", str(exc))
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "exit status {}".format(completed.returncode)
        return CommandResult(False, completed.stdout, detail)
    return CommandResult(True, completed.stdout, completed.stderr)


def concise_error(text):
    return " ".join(str(text).strip().split())[:300]


def merge_info(target, incoming):
    if not target:
        target = {}
    for key, value in incoming.items():
        if key == "scheduler_sources" and value:
            sources = split_flags(target.get(key))
            sources.update(split_flags(value))
            target[key] = join_flags(sources)
            continue
        if value is not None and value != "":
            target[key] = value
    return target


class SchedulerClient(object):
    def __init__(self, user, timeout, command_runner=run_command):
        self.user = user
        self.timeout = timeout
        self.command_runner = command_runner
        self.warnings = []

    def _run(self, label, args):
        result = self.command_runner(args, self.timeout)
        if not result.ok:
            self.warnings.append("{} query failed: {}".format(label, concise_error(result.stderr)))
            return None
        return result.stdout

    def query_squeue(self):
        args = [
            "squeue",
            "--noheader",
            "--user",
            self.user,
            "--format=%i|%j|%P|%D|%C|%M|%l|%T",
        ]
        output = self._run("squeue", args)
        if output is None:
            return {}, False

        jobs = {}
        for line in output.splitlines():
            if not line.strip():
                continue
            parts = line.split("|", 7)
            if len(parts) != 8:
                self.warnings.append("ignored malformed squeue line: {}".format(concise_error(line)))
                continue
            job_id, name, partition, nodes, cpus, elapsed, limit, state = parts
            job_id = job_id.strip()
            if not is_top_level_job_id(job_id):
                continue
            jobs[job_id] = {
                "slurm_job_id": job_id,
                "job_name": name.strip(),
                "partition": partition.strip(),
                "node_count": parse_int(nodes),
                "allocated_cpus": parse_int(cpus),
                "elapsed_seconds": parse_duration_seconds(elapsed),
                "requested_wall_time_seconds": parse_duration_seconds(limit),
                "job_state": normalize_state(state),
                "scheduler_sources": "squeue",
            }
        return jobs, True

    def query_sacct(self, job_ids):
        jobs = {}
        if not job_ids:
            return jobs, True

        all_ok = True
        sorted_ids = sorted(set(job_ids))
        for offset in range(0, len(sorted_ids), 100):
            chunk = sorted_ids[offset : offset + 100]
            args = [
                "sacct",
                "--noheader",
                "--parsable2",
                "--jobs",
                ",".join(chunk),
                "--format=JobIDRaw,JobName,Partition,AllocNodes,AllocCPUS,ElapsedRaw,State,ExitCode,Timelimit,Submit,Start,End",
            ]
            output = self._run("sacct", args)
            if output is None:
                all_ok = False
                continue
            for line in output.splitlines():
                if not line.strip():
                    continue
                parts = line.split("|")
                if len(parts) < 12:
                    self.warnings.append(
                        "ignored malformed sacct line: {}".format(concise_error(line))
                    )
                    continue
                job_id = parts[0].strip()
                if not is_top_level_job_id(job_id):
                    continue
                jobs[job_id] = {
                    "slurm_job_id": job_id,
                    "job_name": parts[1].strip(),
                    "partition": parts[2].strip(),
                    "node_count": parse_int(parts[3]),
                    "allocated_cpus": parse_int(parts[4]),
                    "elapsed_seconds": parse_int(parts[5]),
                    "job_state": normalize_state(parts[6]),
                    "exit_code": parts[7].strip(),
                    "requested_wall_time_seconds": parse_duration_seconds(parts[8]),
                    "submit_time": parts[9].strip(),
                    "start_time": parts[10].strip(),
                    "end_time": parts[11].strip(),
                    "scheduler_sources": "sacct",
                }
        return jobs, all_ok

    def query_scontrol(self, job_id):
        output = self._run("scontrol job {}".format(job_id), ["scontrol", "show", "job", "--oneliner", job_id])
        if output is None:
            return {}, False

        values = {}
        for key, value in SCONTROL_FIELD_RE.findall(output.strip()):
            values[key] = value.strip()
        if not values:
            self.warnings.append("ignored malformed scontrol output for job {}".format(job_id))
            return {}, False

        return {
            "slurm_job_id": values.get("JobId", job_id),
            "job_name": values.get("JobName", ""),
            "partition": values.get("Partition", ""),
            "qos": values.get("QOS", ""),
            "node_count": parse_int(values.get("NumNodes")),
            "allocated_cpus": parse_int(values.get("NumCPUs")),
            "elapsed_seconds": parse_duration_seconds(values.get("RunTime")),
            "requested_wall_time_seconds": parse_duration_seconds(values.get("TimeLimit")),
            "job_state": normalize_state(values.get("JobState")),
            "exit_code": values.get("ExitCode", ""),
            "submit_time": values.get("SubmitTime", ""),
            "start_time": values.get("StartTime", ""),
            "end_time": values.get("EndTime", ""),
            "scheduler_sources": "scontrol",
        }, True


def load_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []
        rows = []
        for index, original in enumerate(reader):
            row = {field: original.get(field, "") for field in CSV_FIELDS}
            if not row["record_id"]:
                row["record_id"] = "legacy-{:06d}".format(index + 1)
            if not row["record_type"]:
                row["record_type"] = "allocation" if row["slurm_job_id"] else "planned"
            rows.append(row)
    validate_rows(rows)
    return rows


def validate_rows(rows):
    record_ids = set()
    job_ids = {}
    for row in rows:
        record_id = row["record_id"]
        if not record_id:
            raise ValueError("ledger row has no record_id")
        if record_id in record_ids:
            raise ValueError("duplicate ledger record_id {}".format(record_id))
        record_ids.add(record_id)

        job_id = row["slurm_job_id"]
        if not job_id:
            continue
        require_job_id(job_id)
        if job_id in job_ids:
            raise ValueError(
                "duplicate Slurm job ID {} in records {} and {}".format(
                    job_id, job_ids[job_id], record_id
                )
            )
        job_ids[job_id] = record_id


def atomic_write_text(path, text):
    directory = os.path.dirname(path)
    fd, temporary_path = tempfile.mkstemp(prefix=".tmp-ledger-", dir=directory)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def write_rows(path, rows):
    directory = os.path.dirname(path)
    fd, temporary_path = tempfile.mkstemp(prefix=".tmp-ledger-", dir=directory)
    try:
        with os.fdopen(fd, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def get_git_version():
    result = run_command(["git", "rev-parse", "HEAD"], timeout=5)
    if result.ok:
        return result.stdout.strip()
    return "unknown"


def row_by_record_id(rows, record_id):
    for row in rows:
        if row["record_id"] == record_id:
            return row
    return None


def row_by_job_id(rows, job_id):
    for row in rows:
        if row["slurm_job_id"] == job_id:
            return row
    return None


def register_planned_row(rows, args, now):
    required = [
        ("--job-name", args.job_name),
        ("--purpose", args.purpose),
        ("--script-path", args.script_path),
        ("--nodes", args.nodes),
        ("--wall-time", args.wall_time),
        ("--output-dir", args.output_dir),
    ]
    missing = [option for option, value in required if value in (None, "")]
    if missing:
        raise ValueError("--register-planned requires {}".format(", ".join(missing)))
    wall_seconds = parse_duration_seconds(args.wall_time)
    if wall_seconds is None:
        raise ValueError("--wall-time must use Slurm syntax such as 02:00:00 or 1-00:00:00")
    if args.nodes <= 0:
        raise ValueError("--nodes must be positive")
    if args.cpus is not None and args.cpus < 0:
        raise ValueError("--cpus cannot be negative")
    max_node_hours = args.nodes * wall_seconds / 3600.0
    if args.expected_node_hours is not None:
        if args.expected_node_hours < 0:
            raise ValueError("--expected-node-hours cannot be negative")
        if args.expected_node_hours > max_node_hours:
            raise ValueError("--expected-node-hours cannot exceed the requested maximum")

    row = new_record("planned", now)
    row.update(
        {
            "job_name": args.job_name,
            "purpose": args.purpose,
            "script_path": args.script_path,
            "code_version": args.code_version or get_git_version(),
            "partition": args.partition or "batch",
            "qos": args.qos or "",
            "requested_nodes": str(args.nodes),
            "node_count": str(args.nodes),
            "allocated_cpus": "" if args.cpus is None else str(args.cpus),
            "requested_wall_time_seconds": str(wall_seconds),
            "requested_wall_time": format_duration(wall_seconds),
            "expected_node_hours": format_hours(args.expected_node_hours),
            "output_directory": args.output_dir,
            "notes": args.notes or "",
        }
    )
    rows.append(row)
    return row


def merge_linked_rows(planned_row, allocation_row):
    """Move scheduler data into a planned record while retaining registered metadata."""
    registered_fields = {
        "job_name",
        "purpose",
        "script_path",
        "code_version",
        "partition",
        "qos",
        "requested_nodes",
        "requested_wall_time",
        "requested_wall_time_seconds",
        "expected_node_hours",
        "output_directory",
        "notes",
    }
    for field in CSV_FIELDS:
        if field in {"record_id", "timestamp"}:
            continue
        if field in registered_fields and planned_row.get(field):
            continue
        if allocation_row.get(field):
            planned_row[field] = allocation_row[field]


def link_planned_row(rows, planned_row, job_id):
    require_job_id(job_id)
    existing = row_by_job_id(rows, job_id)
    if existing is not None and existing is not planned_row:
        merge_linked_rows(planned_row, existing)
        rows.remove(existing)
    planned_row["slurm_job_id"] = job_id
    planned_row["record_type"] = "allocation"
    if planned_row["job_state"] == "PLANNED":
        planned_row["job_state"] = "UNKNOWN"


def import_job(rows, job_id, now, note=None):
    require_job_id(job_id)
    row = row_by_job_id(rows, job_id)
    if row is not None:
        return row
    row = new_record("allocation", now)
    row["slurm_job_id"] = job_id
    row["notes"] = note or "Imported from Slurm; purpose, script path, and code version were not registered."
    rows.append(row)
    return row


def apply_scheduler_info(row, info, now):
    for field in [
        "slurm_job_id",
        "job_name",
        "partition",
        "qos",
        "node_count",
        "allocated_cpus",
        "elapsed_seconds",
        "job_state",
        "exit_code",
        "requested_wall_time_seconds",
        "submit_time",
        "start_time",
        "end_time",
        "scheduler_sources",
    ]:
        value = info.get(field)
        if value is not None and value != "":
            row[field] = str(value)
    if not row["requested_nodes"] and row["node_count"]:
        row["requested_nodes"] = row["node_count"]
    row["record_type"] = "allocation"
    row["last_updated"] = now
    row["_scheduler_found"] = True


def refresh_rows(rows, scheduler, discover_active, explicit_job_ids, now):
    queue_jobs, squeue_ok = scheduler.query_squeue()
    tracked_before_discovery = {row["slurm_job_id"] for row in rows if row["slurm_job_id"]}
    untracked_queue_ids = sorted(set(queue_jobs) - tracked_before_discovery)
    if discover_active:
        for job_id in untracked_queue_ids:
            import_job(rows, job_id, now, note="Auto-discovered from squeue; workflow metadata was not registered.")
    elif untracked_queue_ids:
        scheduler.warnings.append(
            "squeue returned untracked active job(s): {}. Use --discover-active or --job-id to import them.".format(
                ", ".join(untracked_queue_ids)
            )
        )

    tracked_job_ids = {row["slurm_job_id"] for row in rows if row["slurm_job_id"]}
    tracked_job_ids.update(explicit_job_ids)
    account_jobs, sacct_ok = scheduler.query_sacct(tracked_job_ids)

    # sacct may expand a requested array job into top-level array allocations.
    for job_id in sorted(account_jobs):
        import_job(rows, job_id, now)

    combined = {}
    for job_id, info in account_jobs.items():
        combined[job_id] = merge_info(combined.get(job_id), info)
    for job_id, info in queue_jobs.items():
        if job_id in tracked_job_ids or discover_active:
            combined[job_id] = merge_info(combined.get(job_id), info)

    for job_id, info in list(combined.items()):
        if not is_terminal(info.get("job_state")):
            control_info, _ = scheduler.query_scontrol(job_id)
            combined[job_id] = merge_info(info, control_info)

    for row in rows:
        job_id = row["slurm_job_id"]
        if job_id and job_id in combined:
            apply_scheduler_info(row, combined[job_id], now)
        elif job_id:
            row["_scheduler_found"] = False
            row["last_updated"] = now
    return squeue_ok, sacct_ok


def calculate_rows(rows, budget, refreshed, refresh_complete, now):
    # Python's sort is stable, so records registered in the same second retain
    # ledger insertion order instead of being reordered by random UUID values.
    rows.sort(key=lambda row: row["timestamp"])
    cumulative = 0.0

    for row in rows:
        row["last_updated"] = now
        state = normalize_state(row["job_state"])
        row["job_state"] = state

        requested_nodes = parse_int(row["requested_nodes"])
        node_count = parse_int(row["node_count"])
        wall_seconds = parse_int(row["requested_wall_time_seconds"])
        elapsed_seconds = parse_int(row["elapsed_seconds"])

        if wall_seconds is None:
            wall_seconds = parse_duration_seconds(row["requested_wall_time"])
        if elapsed_seconds is None:
            elapsed_seconds = parse_duration_seconds(row["actual_elapsed_time"])
        if requested_nodes is None:
            requested_nodes = node_count

        row["requested_nodes"] = "" if requested_nodes is None else str(requested_nodes)
        row["node_count"] = "" if node_count is None else str(node_count)
        row["requested_wall_time_seconds"] = "" if wall_seconds is None else str(wall_seconds)
        row["requested_wall_time"] = format_duration(wall_seconds)
        row["elapsed_seconds"] = "" if elapsed_seconds is None else str(elapsed_seconds)
        row["actual_elapsed_time"] = format_duration(elapsed_seconds)

        max_node_hours = None
        if requested_nodes is not None and wall_seconds is not None:
            max_node_hours = requested_nodes * wall_seconds / 3600.0
        row["planned_max_node_hours"] = format_hours(max_node_hours)

        consumed = None
        if state in NOT_STARTED_STATES:
            consumed = 0.0
        elif elapsed_seconds is not None and node_count is not None:
            consumed = node_count * elapsed_seconds / 3600.0
        row["node_hours_consumed"] = format_hours(consumed)

        additional = None
        if not is_terminal(state) and max_node_hours is not None:
            if state in NOT_STARTED_STATES or consumed is None:
                additional = max_node_hours
            else:
                additional = max(0.0, max_node_hours - consumed)
        row["pending_max_additional_node_hours"] = format_hours(additional)

        flags = set()
        if not refreshed:
            flags.update(
                flag for flag in split_flags(row["accounting_flags"]) if flag.startswith("scheduler_")
            )
        if row["slurm_job_id"] and refreshed:
            if row.get("_scheduler_found") is False:
                if refresh_complete:
                    flags.add("scheduler_record_not_found")
                else:
                    flags.add("scheduler_refresh_incomplete")
        if row["record_type"] == "allocation" and not row["purpose"]:
            flags.add("unregistered_job_metadata")
        if not is_terminal(state) and max_node_hours is None:
            flags.add("missing_pending_exposure")
        if state not in NOT_STARTED_STATES and elapsed_seconds is None:
            flags.add("missing_elapsed_time")
        if state not in NOT_STARTED_STATES and node_count is None:
            flags.add("missing_node_count")
        row["accounting_flags"] = join_flags(flags)

        if consumed is not None:
            cumulative += consumed
        row["cumulative_node_hours_consumed"] = format_hours(cumulative)
        row["remaining_node_hour_budget"] = format_hours(budget - cumulative)

        row.pop("_scheduler_found", None)


def summary_metrics(rows, budget):
    consumed = sum(parse_float(row["node_hours_consumed"]) or 0.0 for row in rows)
    pending = sum(parse_float(row["pending_max_additional_node_hours"]) or 0.0 for row in rows)
    flagged = [row for row in rows if row["accounting_flags"]]
    active = [
        row
        for row in rows
        if not is_terminal(row["job_state"]) and row["job_state"] != "UNKNOWN"
    ]
    unresolved = [row for row in rows if row["job_state"] == "UNKNOWN"]
    return {
        "consumed": consumed,
        "remaining": budget - consumed,
        "pending": pending,
        "projected_remaining": budget - consumed - pending,
        "flagged": flagged,
        "active": active,
        "unresolved": unresolved,
    }


def markdown_cell(value):
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def render_summary(rows, budget, now, warnings):
    metrics = summary_metrics(rows, budget)
    visible_rows = []
    for row in rows:
        if not is_terminal(row["job_state"]):
            visible_rows.append(row)

    lines = [
        "# Compute Budget Summary",
        "",
        "Updated: `{}`".format(now),
        "",
        "| Metric | Node-hours |",
        "| --- | ---: |",
        "| Workflow budget | {} |".format(format_hours(budget)),
        "| Consumed allocated runtime | {} |".format(format_hours(metrics["consumed"])),
        "| Remaining budget | {} |".format(format_hours(metrics["remaining"])),
        "| Pending maximum additional exposure | {} |".format(format_hours(metrics["pending"])),
        "| Projected remaining after pending maximum | {} |".format(
            format_hours(metrics["projected_remaining"])
        ),
        "",
        "Tracked records: {}. Accounting-flagged records: {}.".format(
            len(rows), len(metrics["flagged"])
        ),
    ]

    if metrics["projected_remaining"] < 0:
        lines.extend(
            [
                "",
                "**Budget warning:** pending maximum exposure exceeds the remaining workflow budget by {} node-hours.".format(
                    format_hours(-metrics["projected_remaining"])
                ),
            ]
        )

    if visible_rows:
        lines.extend(
            [
                "",
                "## Planned And Active",
                "",
                "| Record ID | Slurm job ID | Name | State | Nodes | Wall time | Consumed | Max additional exposure | Flags |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in visible_rows:
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    markdown_cell(row["record_id"]),
                    markdown_cell(row["slurm_job_id"]),
                    markdown_cell(row["job_name"]),
                    markdown_cell(row["job_state"]),
                    markdown_cell(row["requested_nodes"] or row["node_count"]),
                    markdown_cell(row["requested_wall_time"]),
                    markdown_cell(row["node_hours_consumed"]),
                    markdown_cell(row["pending_max_additional_node_hours"]),
                    markdown_cell(row["accounting_flags"]),
                )
            )

    if metrics["flagged"]:
        lines.extend(["", "## Accounting Flags", ""])
        for row in metrics["flagged"]:
            identifier = row["slurm_job_id"] or row["record_id"]
            lines.append("- `{}`: {}".format(identifier, row["accounting_flags"]))

    if warnings:
        lines.extend(["", "## Refresh Warnings", ""])
        for warning in warnings:
            lines.append("- {}".format(warning))

    lines.append("")
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, help="Directory that stores the CSV ledger and Markdown summary.")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET_NODE_HOURS, help="Total workflow node-hour budget (default: 5000).")
    parser.add_argument("--user", default=os.environ.get("USER") or getpass.getuser(), help="Slurm user for squeue queries.")
    parser.add_argument("--command-timeout", type=float, default=15.0, help="Scheduler query timeout in seconds.")
    parser.add_argument("--no-refresh", action="store_true", help="Write local ledger changes without querying Slurm.")
    parser.add_argument("--discover-active", action="store_true", help="Import active jobs returned by squeue for --user.")
    parser.add_argument("--job-id", action="append", default=[], help="Import or refresh a top-level Slurm allocation ID. Repeat as needed.")

    parser.add_argument("--register-planned", action="store_true", help="Register a planned job before submission.")
    parser.add_argument("--link-planned", metavar="RECORD_ID", help="Attach one --job-id to an existing planned record after submission.")
    parser.add_argument("--job-name", help="Planned Slurm job name.")
    parser.add_argument("--purpose", help="Scientific or technical purpose of a planned job.")
    parser.add_argument("--script-path", help="Job script path for a planned job.")
    parser.add_argument("--code-version", help="Git commit or code-version identifier; defaults to the current Git commit.")
    parser.add_argument("--partition", help="Planned Slurm partition (default for planned jobs: batch).")
    parser.add_argument("--qos", help="Planned Slurm QOS, such as debug.")
    parser.add_argument("--nodes", type=int, help="Requested node count for a planned job.")
    parser.add_argument("--cpus", type=int, help="Requested or allocated CPU count for a planned job.")
    parser.add_argument("--wall-time", help="Requested Slurm wall time, such as 02:00:00.")
    parser.add_argument("--expected-node-hours", type=float, help="Expected node-hours for the planned job.")
    parser.add_argument("--output-dir", help="Unique planned output directory.")
    parser.add_argument("--notes", help="Optional planned-job notes.")
    return parser


def execute(args):
    if args.budget <= 0:
        raise ValueError("--budget must be positive")
    if args.command_timeout <= 0:
        raise ValueError("--command-timeout must be positive")
    if args.register_planned and args.link_planned:
        raise ValueError("--register-planned and --link-planned cannot be used together")
    if args.link_planned and len(args.job_id) != 1:
        raise ValueError("--link-planned requires exactly one --job-id")
    for job_id in args.job_id:
        require_job_id(job_id)

    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    ledger_path = os.path.join(results_dir, LEDGER_FILENAME)
    summary_path = os.path.join(results_dir, SUMMARY_FILENAME)
    lock_path = os.path.join(results_dir, LOCK_FILENAME)
    now = utc_timestamp()

    with open(lock_path, "a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        rows = load_rows(ledger_path)

        registered_row = None
        if args.register_planned:
            registered_row = register_planned_row(rows, args, now)
            if args.job_id:
                if len(args.job_id) != 1:
                    raise ValueError("--register-planned accepts at most one --job-id")
                link_planned_row(rows, registered_row, args.job_id[0])

        if args.link_planned:
            planned_row = row_by_record_id(rows, args.link_planned)
            if planned_row is None:
                raise ValueError("planned record {} was not found".format(args.link_planned))
            if planned_row["slurm_job_id"] and planned_row["slurm_job_id"] != args.job_id[0]:
                raise ValueError("planned record {} is already linked to job {}".format(args.link_planned, planned_row["slurm_job_id"]))
            link_planned_row(rows, planned_row, args.job_id[0])

        for job_id in args.job_id:
            import_job(rows, job_id, now)

        warnings = []
        refreshed = not args.no_refresh
        refresh_complete = True
        if refreshed:
            scheduler = SchedulerClient(args.user, args.command_timeout)
            squeue_ok, sacct_ok = refresh_rows(
                rows,
                scheduler,
                args.discover_active,
                args.job_id,
                now,
            )
            refresh_complete = squeue_ok and sacct_ok
            warnings.extend(scheduler.warnings)

        calculate_rows(rows, args.budget, refreshed, refresh_complete, now)
        validate_rows(rows)
        write_rows(ledger_path, rows)
        atomic_write_text(summary_path, render_summary(rows, args.budget, now, warnings))

    metrics = summary_metrics(rows, args.budget)
    print("Ledger: {}".format(ledger_path))
    print("Summary: {}".format(summary_path))
    if registered_row is not None:
        print("Planned record ID: {}".format(registered_row["record_id"]))
    print("Consumed node-hours: {}".format(format_hours(metrics["consumed"])))
    print("Remaining budget: {}".format(format_hours(metrics["remaining"])))
    print("Pending maximum additional exposure: {}".format(format_hours(metrics["pending"])))
    if metrics["projected_remaining"] < 0:
        print(
            "WARNING: pending maximum exposure exceeds remaining budget by {} node-hours".format(
                format_hours(-metrics["projected_remaining"])
            )
        )
    if warnings:
        print("Refresh warnings: {}".format(len(warnings)))
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return execute(args)
    except ValueError as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
