import argparse
import csv
import importlib.util
import os
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "prompt1", "update_compute_ledger.py")
SPEC = importlib.util.spec_from_file_location("update_compute_ledger", SCRIPT)
ledger = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ledger)


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


class FakeRunner(object):
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def __call__(self, args, timeout):
        self.calls.append(args)
        return self.responses(args)


class ComputeLedgerTests(unittest.TestCase):
    def test_duration_parsing_and_formatting(self):
        self.assertEqual(ledger.parse_duration_seconds("02:03:04"), 7384)
        self.assertEqual(ledger.parse_duration_seconds("1-02:03:04"), 93784)
        self.assertEqual(ledger.parse_duration_seconds("03:04"), 184)
        self.assertIsNone(ledger.parse_duration_seconds("Partition_Limit"))
        self.assertEqual(ledger.format_duration(93784), "1-02:03:04")

    def test_register_planned_job_writes_budget_exposure(self):
        with tempfile.TemporaryDirectory() as tempdir:
            args = ledger.build_parser().parse_args(
                [
                    "--results-dir",
                    tempdir,
                    "--no-refresh",
                    "--register-planned",
                    "--job-name",
                    "smoke",
                    "--purpose",
                    "small validation",
                    "--script-path",
                    "job_scripts/smoke.sh",
                    "--nodes",
                    "2",
                    "--wall-time",
                    "02:00:00",
                    "--output-dir",
                    "Results/smoke",
                    "--code-version",
                    "test-commit",
                ]
            )
            ledger.execute(args)

            rows = read_csv(os.path.join(tempdir, ledger.LEDGER_FILENAME))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["job_state"], "PLANNED")
            self.assertEqual(rows[0]["planned_max_node_hours"], "4")
            self.assertEqual(rows[0]["pending_max_additional_node_hours"], "4")
            self.assertEqual(rows[0]["node_hours_consumed"], "0")

    def test_refresh_counts_top_level_allocations_once_and_running_elapsed(self):
        def responses(args):
            if args[0] == "squeue":
                return ledger.CommandResult(
                    True,
                    "202|running|batch|2|64|00:30:00|02:00:00|RUNNING\n",
                    "",
                )
            if args[0] == "sacct":
                return ledger.CommandResult(
                    True,
                    "\n".join(
                        [
                            "101|done|batch|2|64|3600|COMPLETED|0:0|02:00:00|submit|start|end",
                            "101.batch|batch|batch|2|64|3600|COMPLETED|0:0|02:00:00|submit|start|end",
                            "101.extern|extern|batch|2|64|3600|COMPLETED|0:0|02:00:00|submit|start|end",
                            "202|running|batch|2|64|1800|RUNNING|0:0|02:00:00|submit|start|Unknown",
                        ]
                    )
                    + "\n",
                    "",
                )
            if args[0] == "scontrol":
                return ledger.CommandResult(
                    True,
                    "JobId=202 JobName=running JobState=RUNNING Partition=batch AllocNode:Sid=login:123 NumNodes=2 NumCPUs=64 RunTime=00:30:00 TimeLimit=02:00:00 ExitCode=0:0\n",
                    "",
                )
            raise AssertionError("unexpected command: {}".format(args))

        now = "2026-05-30T12:00:00Z"
        rows = [
            ledger.import_job([], "101", now),
        ]
        ledger.import_job(rows, "202", now)
        scheduler = ledger.SchedulerClient("tester", 5, command_runner=FakeRunner(responses))
        squeue_ok, sacct_ok = ledger.refresh_rows(rows, scheduler, False, [], now)
        ledger.calculate_rows(rows, 5000, True, squeue_ok and sacct_ok, now)

        by_job = {row["slurm_job_id"]: row for row in rows}
        self.assertEqual(set(by_job), {"101", "202"})
        self.assertEqual(by_job["101"]["node_hours_consumed"], "2")
        self.assertEqual(by_job["202"]["node_hours_consumed"], "1")
        self.assertEqual(by_job["202"]["partition"], "batch")
        self.assertEqual(by_job["202"]["pending_max_additional_node_hours"], "3")
        self.assertEqual(by_job["202"]["cumulative_node_hours_consumed"], "3")

    def test_missing_scheduler_record_is_flagged(self):
        def responses(args):
            if args[0] in {"squeue", "sacct"}:
                return ledger.CommandResult(True, "", "")
            raise AssertionError("unexpected command: {}".format(args))

        now = "2026-05-30T12:00:00Z"
        rows = []
        ledger.import_job(rows, "303", now)
        scheduler = ledger.SchedulerClient("tester", 5, command_runner=FakeRunner(responses))
        squeue_ok, sacct_ok = ledger.refresh_rows(rows, scheduler, False, [], now)
        ledger.calculate_rows(rows, 5000, True, squeue_ok and sacct_ok, now)

        flags = set(rows[0]["accounting_flags"].split(";"))
        self.assertIn("scheduler_record_not_found", flags)
        self.assertIn("missing_elapsed_time", flags)
        self.assertIn("missing_node_count", flags)

    def test_linking_imported_allocation_keeps_registered_metadata_without_duplicate(self):
        now = "2026-05-30T12:00:00Z"
        rows = []
        imported = ledger.import_job(rows, "404", now)
        imported["job_name"] = "scheduler-name"
        imported["job_state"] = "PENDING"

        args = argparse.Namespace(
            job_name="registered-name",
            purpose="pilot",
            script_path="job_scripts/pilot.sh",
            nodes=2,
            wall_time="01:00:00",
            output_dir="Results/pilot",
            cpus=64,
            code_version="test-commit",
            partition="batch",
            qos="debug",
            expected_node_hours=1.5,
            notes="registered before submission",
        )
        planned = ledger.register_planned_row(rows, args, now)
        ledger.link_planned_row(rows, planned, "404")

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["slurm_job_id"], "404")
        self.assertEqual(rows[0]["job_name"], "registered-name")
        self.assertEqual(rows[0]["purpose"], "pilot")
        self.assertEqual(rows[0]["job_state"], "PENDING")


if __name__ == "__main__":
    unittest.main()
