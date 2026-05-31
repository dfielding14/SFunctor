"""Runtime regression tests for the Andes Phase 4 shell wrappers."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import textwrap

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = {
    "extract": REPO_ROOT / "job_scripts" / "phase4" / "run_phase4_extract_andes.sh",
    "batch_a": REPO_ROOT / "job_scripts" / "phase4" / "run_phase4_batch_a_sampler_andes.sh",
}
LOCK_SUFFIXES = {
    "extract": "phase4_extract_action_lock",
    "batch_a": "phase4_batch_a_action_lock",
}


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
        printf '%s\n' "$*" >> "${FAKE_PYTHON_LOG}"
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
          printf '%s\n' "${FAKE_SQUEUE_ROWS}"
        fi
        """,
    )
    _write_executable(
        fake_bin / "sacct",
        """
        #!/usr/bin/env bash
        set -euo pipefail
        printf '9001|1|1|32|1024K|node|0|512K|2048K|RUNNING|0:0\n'
        """,
    )
    return sfunctor_dir, fake_bin


def _lock_dir(output_root: Path, wrapper_name: str) -> Path:
    return output_root.parent / f".{output_root.name}.{LOCK_SUFFIXES[wrapper_name]}"


def _run_wrapper(
    fake_andes: tuple[Path, Path],
    wrapper_name: str,
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
            "PHASE2_ROOT": str(output_root.parent / "extract"),
            "OUTPUT_ROOT": str(output_root),
            "RUN_DIR": str(run_dir),
            "ACTION": "extract" if wrapper_name == "extract" else "plan",
            "SLURM_JOB_ID": "9001",
            "FAKE_PYTHON_LOG": str(run_dir / "python.log"),
        }
    )
    env.update(environment)
    return subprocess.run(
        ["bash", str(WRAPPERS[wrapper_name])],
        cwd=output_root.parent,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("wrapper_name", tuple(WRAPPERS))
def test_phase4_wrapper_recovers_inactive_lock_and_archives_sacct(
    tmp_path: Path,
    fake_andes: tuple[Path, Path],
    wrapper_name: str,
) -> None:
    output_root = tmp_path / f"{wrapper_name}_output"
    run_dir = tmp_path / f"{wrapper_name}_run"
    lock_dir = _lock_dir(output_root, wrapper_name)
    lock_dir.mkdir()
    (lock_dir / "owner.txt").write_text("job_id=8123\naction=plan\n")

    result = _run_wrapper(
        fake_andes,
        wrapper_name,
        output_root,
        run_dir,
        FAKE_SQUEUE_ROWS="7000\n7001",
        PHASE4_CLEAN_PARTIAL="1",
        PHASE4_CLEAN_INCOMPLETE="1",
        PHASE4_CLEAN_STALE_LOCK="1",
    )

    assert result.returncode == 0, result.stderr
    assert "Recovered stale Phase 4" in result.stderr
    assert not lock_dir.exists()
    assert (run_dir / "resources" / "sacct_9001.psv").is_file()
    if wrapper_name == "extract":
        arguments = (run_dir / "python.log").read_text()
        assert "--clean-partial" in arguments
        assert "--clean-incomplete" in arguments
        assert "--clean-stale-lock" in arguments


@pytest.mark.parametrize("wrapper_name", tuple(WRAPPERS))
@pytest.mark.parametrize(
    ("owner_text", "environment"),
    [
        ("job_id=8123\naction=plan\n", {"FAKE_SQUEUE_ROWS": "8123"}),
        ("", {}),
        ("job_id=8123\naction=plan\n", {"FAKE_SQUEUE_FAIL": "1"}),
    ],
)
def test_phase4_wrapper_preserves_lock_without_verified_inactive_owner(
    tmp_path: Path,
    fake_andes: tuple[Path, Path],
    wrapper_name: str,
    owner_text: str,
    environment: dict[str, str],
) -> None:
    output_root = tmp_path / f"{wrapper_name}_output"
    lock_dir = _lock_dir(output_root, wrapper_name)
    lock_dir.mkdir()
    if owner_text:
        (lock_dir / "owner.txt").write_text(owner_text)

    result = _run_wrapper(
        fake_andes,
        wrapper_name,
        output_root,
        tmp_path / f"{wrapper_name}_run",
        **environment,
    )

    assert result.returncode == 3
    assert lock_dir.is_dir()
