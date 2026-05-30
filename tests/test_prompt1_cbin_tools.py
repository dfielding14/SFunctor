"""Focused synthetic tests for the Prompt 1 cbin census helpers."""

from __future__ import annotations

from pathlib import Path
import struct
import sys
import tempfile
import time
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "prompt1"))
from cbin_tools import (  # noqa: E402
    aggregate_blocks,
    _normalized_moment_error_bound,
    float32_quantization_bound,
    morton_rank,
    parse_binary_shard,
    raw_moment_statistics,
    read_record_fields,
    required_rank_ids,
    robust_nonnegative,
    source_shard_inventory_sha256,
)


def write_synthetic_cbin(path: Path, *, truncate: int = 0) -> np.ndarray:
    header_dump = b"""<mesh>\nnx1 = 4\nnx2 = 4\nnx3 = 4\nnghost = 3\nx1min = -0.5\nx1max = 0.5\nx2min = -0.5\nx2max = 0.5\nx3min = -0.5\nx3max = 0.5\n<meshblock>\nnx1 = 4\nnx2 = 4\nnx3 = 4\n"""
    values = np.arange(8, dtype="<f4").reshape(2, 2, 2)
    prefix = (
        b"Athena binary output version=1.1\n"
        b"  size of preheader=7\n"
        b"  time=6\n"
        b"  cycle=7\n"
        b"  number of moments=1\n"
        b"  coarsening factor=2\n"
        b"  size of location=8\n"
        b"  size of variable=4\n"
        b"  number of variables=1\n"
        b"  variables:  dens\n"
        + f"  header offset={len(header_dump)}\n".encode()
        + header_dump
    )
    record = (
        struct.pack("<6i", 3, 4, 3, 4, 3, 4)
        + struct.pack("<4i", 0, 0, 0, 0)
        + struct.pack("<6d", -0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
        + values.tobytes()
    )
    path.write_bytes((prefix + record)[:-truncate or None])
    return values


class Prompt1CbinToolsTest(unittest.TestCase):
    def test_parse_and_read_preserves_kji_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.cbin"
            expected = write_synthetic_cbin(path)
            shard = parse_binary_shard(path, expect_cbin=True)
            self.assertEqual(shard.records[0].shape_kji, (2, 2, 2))
            np.testing.assert_array_equal(read_record_fields(shard, ("dens",))["dens"], expected)

    def test_truncated_payload_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.cbin"
            write_synthetic_cbin(path, truncate=4)
            with self.assertRaisesRegex(ValueError, "truncated payload"):
                parse_binary_shard(path, expect_cbin=True)

    def test_source_inventory_detects_nonzero_rank_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            basename = "Turb.sample.00001.cbin"
            for rank in range(2):
                path = root / "cbin_mhd_u_bcc_80" / f"rank_{rank:08d}" / basename
                path.parent.mkdir(parents=True)
                path.write_bytes(f"rank={rank}".encode())
            before = source_shard_inventory_sha256(
                root, "mhd_u_bcc", 80, basename, expected_ranks=2
            )
            time.sleep(0.002)
            rank_one = root / "cbin_mhd_u_bcc_80" / "rank_00000001" / basename
            rank_one.write_bytes(b"rank=X")
            after = source_shard_inventory_sha256(
                root, "mhd_u_bcc", 80, basename, expected_ranks=2
            )
            self.assertNotEqual(before, after)

    def test_merge_raw_moments_before_statistics(self) -> None:
        values = np.asarray([[[0.0, 2.0], [0.0, 2.0]], [[0.0, 2.0], [0.0, 2.0]]])
        moments = [aggregate_blocks(values**power, 2) for power in range(1, 5)]
        stats = raw_moment_statistics(*moments)
        self.assertAlmostEqual(float(stats["mean"][0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(stats["variance"][0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(stats["kurtosis"][0, 0, 0]), 1.0)

    def test_tolerance_scale_negative_variance_is_flagged_and_clamped(self) -> None:
        errors = tuple(np.asarray([1.0e-6]) for _ in range(4))
        stats = raw_moment_statistics(
            np.asarray([1.0]),
            np.asarray([1.0 - 1.0e-7]),
            np.asarray([1.0]),
            np.asarray([1.0]),
            abs_error_bounds=errors,
        )
        self.assertEqual(float(stats["variance"][0]), 0.0)
        self.assertTrue(int(stats["moment_flags"][0]) & 1)
        self.assertTrue(int(stats["moment_flags"][0]) & 4)

    def test_material_negative_variance_is_invalid(self) -> None:
        stats = raw_moment_statistics(
            np.asarray([10.0]),
            np.asarray([1.0]),
            np.asarray([1.0]),
            np.asarray([1.0]),
        )
        self.assertTrue(np.isnan(stats["variance"][0]))
        self.assertTrue(int(stats["moment_flags"][0]) & 2)

    def test_cancellation_dominated_standardized_moments_are_unavailable(self) -> None:
        rng = np.random.default_rng(7)
        values = rng.normal(1.0, 0.01, size=1_000_000).astype(np.float32)
        moments = [np.asarray([np.mean(values**power, dtype=np.float32)]) for power in range(1, 5)]
        errors = [float32_quantization_bound(moment) for moment in moments]
        stats = raw_moment_statistics(*moments, abs_error_bounds=errors)
        self.assertTrue(int(stats["moment_flags"][0]) & 16)
        self.assertTrue(int(stats["moment_flags"][0]) & 64)
        self.assertTrue(np.isnan(stats["skewness"][0]))
        self.assertTrue(np.isnan(stats["kurtosis"][0]))

    def test_well_conditioned_standardized_moments_remain_available(self) -> None:
        values = np.asarray([0.0, 0.0, 0.0, 2.0], dtype=np.float32)
        moments = [np.asarray([np.mean(values**power, dtype=np.float32)]) for power in range(1, 5)]
        errors = [float32_quantization_bound(moment) for moment in moments]
        stats = raw_moment_statistics(*moments, abs_error_bounds=errors)
        self.assertFalse(int(stats["moment_flags"][0]) & 16)
        self.assertFalse(int(stats["moment_flags"][0]) & 64)
        self.assertAlmostEqual(float(stats["skewness"][0]), 1.1547005383792517)
        self.assertAlmostEqual(float(stats["kurtosis"][0]), 2.3333333333333335)

    def test_normalized_moment_bound_includes_denominator_uncertainty(self) -> None:
        bound = _normalized_moment_error_bound(
            np.asarray([2.0]),
            np.asarray([0.1]),
            np.asarray([4.0]),
            np.asarray([1.0]),
            denominator_power=1.5,
        )[0]
        numerator_only = 0.1 / 4.0**1.5
        expected = max(
            abs(candidate - 2.0 / 4.0**1.5)
            for candidate in (
                (2.0 - 0.1) / (4.0 - 1.0)**1.5,
                (2.0 - 0.1) / (4.0 + 1.0)**1.5,
                (2.0 + 0.1) / (4.0 - 1.0)**1.5,
                (2.0 + 0.1) / (4.0 + 1.0)**1.5,
            )
        )
        self.assertGreater(bound, numerator_only)
        self.assertAlmostEqual(bound, expected)

    def test_positive_but_unresolved_fourth_moment_suppresses_kurtosis(self) -> None:
        errors = tuple(np.asarray([1.0]) for _ in range(4))
        stats = raw_moment_statistics(
            np.asarray([1000.0]),
            np.asarray([1_000_001.0]),
            np.asarray([1_000_003_000.0]),
            np.asarray([1_000_006_000_001.0]),
            abs_error_bounds=errors,
        )
        self.assertTrue(int(stats["moment_flags"][0]) & 64)
        self.assertTrue(np.isnan(stats["kurtosis"][0]))

    def test_invalid_fourth_moment_does_not_suppress_reliable_skewness(self) -> None:
        stats = raw_moment_statistics(
            np.asarray([0.0]),
            np.asarray([1.0]),
            np.asarray([1.0]),
            np.asarray([-1.0]),
        )
        self.assertTrue(int(stats["moment_flags"][0]) & 8)
        self.assertAlmostEqual(float(stats["skewness"][0]), 1.0)
        self.assertTrue(np.isnan(stats["kurtosis"][0]))

    def test_positive_cancellation_below_bound_is_unavailable(self) -> None:
        values, flags = robust_nonnegative(
            np.asarray([1.0e-9]),
            np.asarray([1.0]),
            abs_error_bound=np.asarray([1.0e-6]),
        )
        self.assertTrue(np.isnan(values[0]))
        self.assertTrue(int(flags[0]) & 4)

    def test_morton_mapping_and_required_rank_counts(self) -> None:
        self.assertEqual(morton_rank(0, 8, 0), 1024)
        self.assertEqual(morton_rank(32, 0, 0), 32768)
        expected_counts = {80: 1, 160: 1, 320: 2, 640: 16, 1280: 128}
        for l_sub, expected in expected_counts.items():
            ranks = required_rank_ids((0, l_sub, 0, l_sub, 0, l_sub))
            self.assertEqual(len(ranks), expected)
            self.assertEqual(len(ranks), len(set(ranks)))


if __name__ == "__main__":
    unittest.main()
