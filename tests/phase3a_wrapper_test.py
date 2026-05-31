"""Regression tests for the Andes Phase 3a shell wrapper."""

from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "job_scripts" / "phase3a" / "run_phase3a_sampler_andes.sh"


def _write_executable(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(contents).lstrip())
    path.chmod(0o755)


@pytest.fixture
def fake_andes(tmp_path: Path) -> tuple[Path, Path]:
    sfunctor_dir = tmp_path / "sfunctor"
    fake_bin = tmp_path / "bin"
    _write_executable(
        sfunctor_dir / "venv_sfunctor" / "bin" / "activate",
        """
        #!/usr/bin/env bash
        """,
    )
    _write_executable(
        sfunctor_dir / "venv_sfunctor" / "bin" / "python",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        shift
        action="$1"
        shift
        output_root=
        while (( "$#" )); do
          if [[ "$1" == --output-root ]]; then
            output_root="$2"
            break
          fi
          shift
        done
        if [[ "${action}" != plan ]]; then
          echo "unexpected fake action: ${action}" >&2
          exit 41
        fi
        if [[ -e "${output_root}" ]]; then
          echo "fake planner saw non-fresh output root: ${output_root}" >&2
          exit 42
        fi
        mkdir -p "${output_root}"
        : > "${output_root}/PLAN_COMPLETE.json"
        """,
    )
    _write_executable(
        fake_bin / "module",
        """
        #!/usr/bin/env bash
        exit 0
        """,
    )
    _write_executable(
        fake_bin / "srun",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        while (( "$#" )); do
          case "$1" in
            -N|-n|--ntasks|--ntasks-per-node|--cpus-per-task)
              shift 2
              ;;
            --*=*)
              shift
              ;;
            *)
              exec "$@"
              ;;
          esac
        done
        """,
    )
    _write_executable(
        fake_bin / "squeue",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        if [[ "${FAKE_SQUEUE_FAIL:-0}" == 1 ]]; then
          exit 1
        fi
        if [[ -n "${FAKE_SQUEUE_ROWS:-}" ]]; then
          printf '%s\\n' "${FAKE_SQUEUE_ROWS}"
        fi
        """,
    )
    _write_executable(
        fake_bin / "sacct",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        printf '9001|1|1|32|1024K|node|0|512K|2048K|RUNNING|0:0\\n'
        """,
    )
    return sfunctor_dir, fake_bin


def _lock_dir(output_root: Path) -> Path:
    return output_root.parent / f".{output_root.name}.phase3a_action_lock"


def _run_wrapper(
    fake_andes: tuple[Path, Path],
    output_root: Path,
    run_dir: Path,
    **environment: str,
) -> subprocess.CompletedProcess[str]:
    sfunctor_dir, fake_bin = fake_andes
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "SFUNCTOR_DIR": str(sfunctor_dir),
            "PHASE2_ROOT": str(output_root.parent / "phase2"),
            "OUTPUT_ROOT": str(output_root),
            "RUN_DIR": str(run_dir),
            "ACTION": "plan",
            "SLURM_JOB_ID": "9001",
        }
    )
    env.update(environment)
    return subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=output_root.parent,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_plan_wrapper_keeps_output_root_fresh_for_planner(
    tmp_path: Path, fake_andes: tuple[Path, Path]
) -> None:
    output_root = tmp_path / "fresh_campaign"

    result = _run_wrapper(fake_andes, output_root, tmp_path / "run")

    assert result.returncode == 0, result.stderr
    assert (output_root / "PLAN_COMPLETE.json").is_file()
    assert not (output_root / "logs").exists()
    assert not _lock_dir(output_root).exists()
    assert (tmp_path / "run" / "resources" / "sacct_9001.psv").is_file()


def test_plan_wrapper_preserves_lock_owned_by_live_slurm_job(
    tmp_path: Path, fake_andes: tuple[Path, Path]
) -> None:
    output_root = tmp_path / "fresh_campaign"
    lock_dir = _lock_dir(output_root)
    lock_dir.mkdir()
    owner = lock_dir / "owner.txt"
    owner.write_text("job_id=8123\naction=plan\n")

    result = _run_wrapper(
        fake_andes,
        output_root,
        tmp_path / "run",
        FAKE_SQUEUE_ROWS="8123",
    )

    assert result.returncode == 3
    assert "another Phase 3a action holds" in result.stderr
    assert owner.is_file()
    assert not output_root.exists()


def test_plan_wrapper_recovers_lock_for_inactive_slurm_job(
    tmp_path: Path, fake_andes: tuple[Path, Path]
) -> None:
    output_root = tmp_path / "fresh_campaign"
    lock_dir = _lock_dir(output_root)
    lock_dir.mkdir()
    (lock_dir / "owner.txt").write_text("job_id=8123\naction=plan\n")

    result = _run_wrapper(
        fake_andes,
        output_root,
        tmp_path / "run",
        FAKE_SQUEUE_ROWS="7000\n7001",
    )

    assert result.returncode == 0, result.stderr
    assert "Recovered stale Phase 3a action lock for inactive Slurm job 8123" in result.stderr
    assert (output_root / "PLAN_COMPLETE.json").is_file()
    assert not lock_dir.exists()


@pytest.mark.parametrize(
    ("owner_text", "environment"),
    [
        ("job_id=unknown\naction=plan\n", {}),
        ("job_id=8123\naction=plan\n", {"FAKE_SQUEUE_FAIL": "1"}),
        ("job_id=8123\naction=plan\n", {"FAKE_SQUEUE_ROWS": "not-a-job-id"}),
    ],
)
def test_plan_wrapper_preserves_lock_without_verified_inactive_slurm_job(
    tmp_path: Path,
    fake_andes: tuple[Path, Path],
    owner_text: str,
    environment: dict[str, str],
) -> None:
    output_root = tmp_path / "fresh_campaign"
    lock_dir = _lock_dir(output_root)
    lock_dir.mkdir()
    owner = lock_dir / "owner.txt"
    owner.write_text(owner_text)

    result = _run_wrapper(fake_andes, output_root, tmp_path / "run", **environment)

    assert result.returncode == 3
    assert owner.read_text() == owner_text
    assert not output_root.exists()
