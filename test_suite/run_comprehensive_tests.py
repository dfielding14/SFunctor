#!/usr/bin/env python3
"""Comprehensive Test Suite Runner for SFunctor.

This runner executes the numbered scripts in ``test_suite/`` and writes a
summary report under ``test_suite/results``.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
import time
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SMALL_FILE = "slice_data/Turb_320_beta100_dedt025_plm_axis2_slice0_file0000.npz"
LARGE_FILE = "slice_data/Turb_2560_beta25_dedt025_plm_axis3_slicem0p375_file0024.npz"

RESULTS_DIR = PROJECT_ROOT / "test_suite" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TESTS = [
    ("00_Setup_Validation", "test_suite.00_setup_and_validation"),
    ("01_Core_Structure_Functions", "test_suite.01_core_structure_functions"),
    ("02_Physics_Calculations", "test_suite.02_physics_calculations"),
    ("03_Time_Series", "test_suite.03_time_series"),
    ("04_Anisotropy", "test_suite.04_anisotropy"),
    ("05_Cross_Correlation", "test_suite.05_cross_correlation"),
    ("06_Wavelet_Decomposition", "test_suite.06_wavelet_decomposition"),
    ("07_Wavelet_Power_Maps", "test_suite.07_wavelet_power_maps"),
    ("08_GPU_Acceleration", "test_suite.08_gpu_acceleration"),
    ("09_Visualization_Gallery", "test_suite.09_visualization_gallery"),
]


def _module_file(module_name: str) -> Path:
    return PROJECT_ROOT / (module_name.replace(".", "/") + ".py")


def _call_with_optional_test_file(fn: Callable[..., object], test_file: str) -> object:
    sig = inspect.signature(fn)
    if len(sig.parameters) == 0:
        return fn()
    return fn(test_file)


def _run_module(module_name: str, test_file: str) -> bool:
    module = importlib.import_module(module_name)

    if hasattr(module, "main") and callable(module.main):
        return bool(_call_with_optional_test_file(module.main, test_file))

    preferred = [
        "test_setup",
        "test_core_structure_functions",
        "test_physics_calculations",
        "test_time_series",
        "test_anisotropy",
    ]
    for name in preferred:
        if hasattr(module, name) and callable(getattr(module, name)):
            return bool(_call_with_optional_test_file(getattr(module, name), test_file))

    test_functions = [
        getattr(module, name)
        for name in dir(module)
        if name.startswith("test_") and callable(getattr(module, name))
    ]
    if len(test_functions) == 1:
        return bool(_call_with_optional_test_file(test_functions[0], test_file))

    raise RuntimeError(
        f"No executable entry point found in {module_name}. Expected main() or one test_* function."
    )


def run_test(test_name: str, test_file: str, module_name: str) -> tuple[bool, float]:
    print(f"\n{'=' * 60}")
    print(f"RUNNING: {test_name}")
    print(f"Data: {test_file}")
    print(f"{'=' * 60}")

    start = time.time()
    try:
        success = _run_module(module_name, test_file)
    except Exception as exc:  # noqa: BLE001
        print(f"  Error running {test_name}: {exc}")
        import traceback

        traceback.print_exc()
        success = False

    elapsed = time.time() - start
    status = "PASS" if success else "FAIL"
    print(f"\n{test_name}: {status} ({elapsed:.1f}s)")
    return success, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SFunctor comprehensive tests")
    parser.add_argument("--fast", action="store_true", help="Use the small dataset")
    parser.add_argument("--full", action="store_true", help="Use the large dataset")
    parser.add_argument("--test", type=str, default=None, help="Run only tests matching this prefix (e.g. 01)")
    args = parser.parse_args()

    if args.full:
        test_file = LARGE_FILE
        print(f"Using LARGE dataset: {test_file}")
    else:
        test_file = SMALL_FILE
        print(f"Using SMALL dataset: {test_file}")
        if not args.fast:
            print("(Use --full for high-resolution coverage)")

    tests = TESTS
    if args.test:
        tests = [(name, module) for name, module in tests if name.startswith(args.test)]
        if not tests:
            print(f"No test found matching: {args.test}")
            return 1

    print(f"\n{'=' * 60}")
    print("SFUNCTOR COMPREHENSIVE TEST SUITE")
    print(f"{'=' * 60}")
    print(f"Tests requested: {len(tests)}")

    results: list[tuple[str, bool, float]] = []
    total_start = time.time()

    for test_name, module_name in tests:
        module_path = _module_file(module_name)
        if not module_path.exists():
            print(f"\nSkipping {test_name} (not implemented: {module_path})")
            continue

        success, elapsed = run_test(test_name, test_file, module_name)
        results.append((test_name, success, elapsed))
        time.sleep(0.25)

    total_elapsed = time.time() - total_start

    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print(f"{'=' * 60}")

    passed = sum(1 for _, ok, _ in results if ok)
    failed = len(results) - passed

    for test_name, ok, elapsed in results:
        status = "PASS" if ok else "FAIL"
        print(f"{test_name:30s}: {status:4s} ({elapsed:6.1f}s)")

    print(f"\n{'=' * 60}")
    print(f"Total: {passed} passed, {failed} failed")
    print(f"Time: {total_elapsed:.1f}s")
    print(f"{'=' * 60}")

    report_file = RESULTS_DIR / "test_report.txt"
    with report_file.open("w", encoding="utf-8") as f:
        f.write("SFunctor Test Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Data file: {test_file}\n")
        f.write(f"Total tests run: {len(results)}\n")
        f.write(f"Passed: {passed}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Total time: {total_elapsed:.1f}s\n\n")
        f.write("Detailed Results:\n")
        for test_name, ok, elapsed in results:
            status = "PASSED" if ok else "FAILED"
            f.write(f"  {test_name}: {status} ({elapsed:.1f}s)\n")

    print(f"\nReport saved to: {report_file}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
