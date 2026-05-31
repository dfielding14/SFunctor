"""Focused synthetic tests for Phase 1 analysis hardening."""

from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "phase1"))
from cbin_tools import PRIMARY_LABELS, _rounding_gamma  # noqa: E402
from run_phase1_catalog import (  # noqa: E402
    _histogram_edges,
    _require_analysis_completion,
    _safe_ratio,
    _safe_spearman,
    _scaled_feature_matrices,
    _select_spatially_separated,
    _write_analysis_completion,
    derive_catalog,
    PILOT_MATCH_PROPERTIES,
)
from validate_reconstruction import (  # noqa: E402
    _regional_error_bounds,
    comparison_rows,
    retained_primary_diagnostics,
)
from verify_phase1_outputs import (  # noqa: E402
    EXPECTED_PRIMARY_ONLY_CATALOG_COLUMNS,
    FLAG_COLUMNS,
    _require_no_sgs_provenance,
    _require_primary_only_catalog,
    _require_primary_raw_cache,
    _verify_foundational_catalog_against_raw,
    _expected_grid_columns,
    _verify_expected_grid,
    _verify_primary_flags,
    _verify_primary_formulas,
)


def _primary_formula_columns() -> dict[str, np.ndarray]:
    b_mean = np.sqrt(14.0)
    delta_b = np.sqrt(15.0)
    return {
        "bcc1_mean": np.asarray([1.0]),
        "bcc2_mean": np.asarray([2.0]),
        "bcc3_mean": np.asarray([3.0]),
        "bcc1_variance": np.asarray([4.0]),
        "bcc2_variance": np.asarray([5.0]),
        "bcc3_variance": np.asarray([6.0]),
        "bcc1_variance_error_bound": np.asarray([0.1]),
        "bcc2_variance_error_bound": np.asarray([0.1]),
        "bcc3_variance_error_bound": np.asarray([0.1]),
        "B_mean": np.asarray([b_mean]),
        "B_mean_error_bound": np.asarray([0.1]),
        "B2_mean": np.asarray([29.0]),
        "B_rms": np.asarray([np.sqrt(29.0)]),
        "deltaB_sq": np.asarray([15.0]),
        "deltaB_sq_error_bound": np.asarray([0.3]),
        "deltaB": np.asarray([delta_b]),
        "B_mean_sq_over_B2_mean": np.asarray([14.0 / 29.0]),
        "deltaB_sq_over_B2_mean": np.asarray([15.0 / 29.0]),
        "dBB": np.asarray([delta_b / b_mean]),
        "dens_mean": np.asarray([2.0]),
        "dens_variance": np.asarray([0.25]),
        "dens_sigma": np.asarray([0.5]),
        "rho_sigma_over_mean": np.asarray([0.25]),
        "magnetic_energy_mean": np.asarray([14.5]),
        "vA_mean_proxy": np.asarray([b_mean / np.sqrt(2.0)]),
        "vA_rms_like_proxy": np.asarray([np.sqrt(29.0 / 2.0)]),
        "mom1_mean": np.asarray([2.0]),
        "mom2_mean": np.asarray([4.0]),
        "mom3_mean": np.asarray([6.0]),
        "mom1_variance": np.asarray([4.0]),
        "mom2_variance": np.asarray([9.0]),
        "mom3_variance": np.asarray([16.0]),
        "mom1_sigma": np.asarray([2.0]),
        "mom2_sigma": np.asarray([3.0]),
        "mom3_sigma": np.asarray([4.0]),
        "u_mass_weighted_mean_x": np.asarray([1.0]),
        "u_mass_weighted_mean_y": np.asarray([2.0]),
        "u_mass_weighted_mean_z": np.asarray([3.0]),
    }


class Phase1AnalysisTest(unittest.TestCase):
    def test_catalog_derivation_is_primary_only(self) -> None:
        source = np.linspace(0.5, 1.2, 8, dtype=np.float32).reshape(2, 2, 2)
        raw = {}
        for base in ("dens", "mom1", "mom2", "mom3", "ener", "bcc1", "bcc2", "bcc3"):
            values = source + {
                "dens": 1.0,
                "mom1": 0.1,
                "mom2": 0.2,
                "mom3": 0.3,
                "ener": 2.0,
                "bcc1": 0.4,
                "bcc2": 0.5,
                "bcc3": 0.6,
            }[base]
            for power, suffix in enumerate(("1st", "2nd", "3rd", "4th"), start=1):
                raw[f"{base}_{suffix}"] = values**power
        columns, metadata = derive_catalog(raw, 80)
        _require_primary_only_catalog(columns, metadata, "synthetic")
        _verify_foundational_catalog_against_raw(columns, raw, 80, "synthetic")
        self.assertIn("dBB", columns)
        self.assertIn("magnetic_energy_mean", columns)
        self.assertIn("vA_mean_proxy", columns)
        self.assertNotIn("Ms_favre", columns)
        self.assertNotIn("MA_energy", columns)
        self.assertNotIn("magnetic_to_kinetic_energy", columns)
        self.assertIn("SGS products are intentionally excluded", metadata["physics_caveat"])

    def test_constant_histogram_is_skipped(self) -> None:
        edges, log_scale = _histogram_edges(np.ones(32))
        self.assertIsNone(edges)
        self.assertFalse(log_scale)

    def test_constant_spearman_is_unavailable(self) -> None:
        self.assertTrue(np.isnan(_safe_spearman(np.ones(8), np.arange(8.0))))

    def test_matching_uses_one_combined_scaler(self) -> None:
        columns = {
            name: np.asarray([0.0, 10.0, 12.0, 14.0])
            for name in PILOT_MATCH_PROPERTIES
        }
        low, high = _scaled_feature_matrices(columns, np.asarray([0, 1]), np.asarray([2, 3]))
        self.assertLess(float(low[1, 0]), float(high[0, 0]))
        self.assertFalse(np.allclose(np.median(low, axis=0), np.median(high, axis=0)))

    def test_spatial_separation_is_periodic_and_global(self) -> None:
        columns = {
            "center_i": np.asarray([100.0, 5000.0, 10140.0]),
            "center_j": np.asarray([100.0, 5000.0, 100.0]),
            "center_k": np.asarray([100.0, 5000.0, 100.0]),
        }
        selected = _select_spatially_separated(
            columns,
            np.asarray([1, 2]),
            count=2,
            minimum_distance_cells=1280.0,
            existing=[0],
        )
        self.assertEqual(selected, [1])

    def test_ratio_rejects_unresolved_positive_denominator(self) -> None:
        values, flags = _safe_ratio(
            np.asarray([1.0]),
            np.asarray([1.0e-5]),
            denominator_error_bound=np.asarray([1.0e-4]),
        )
        self.assertTrue(np.isnan(values[0]))
        self.assertTrue(int(flags[0]) & 2)

    def test_analysis_completion_hashes_outputs_and_rejects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            analysis_dir = root / "analysis"
            analysis_dir.mkdir()
            build_marker = root / "BUILD_COMPLETE.json"
            verify_marker = root / "VERIFY_COMPLETE.json"
            build_marker.write_text("{}\n")
            verify_marker.write_text("{}\n")
            generated = analysis_dir / "summary.csv"
            generated.write_text("value\n1\n")
            _write_analysis_completion(
                analysis_dir,
                artifact_graph_sha256_value="graph",
                build_marker=build_marker,
                verify_marker=verify_marker,
            )
            manifest = _require_analysis_completion(
                analysis_dir,
                artifact_graph_sha256_value="graph",
                build_marker=build_marker,
                verify_marker=verify_marker,
            )
            self.assertEqual(set(manifest["generated_files"]), {"summary.csv"})
            complete = json.loads((analysis_dir / "ANALYSIS_COMPLETE.json").read_text())
            self.assertEqual(complete["generated_file_count"], 1)
            generated.write_text("value\n2\n")
            with self.assertRaisesRegex(ValueError, "stale or incomplete analysis manifest"):
                _require_analysis_completion(
                    analysis_dir,
                    artifact_graph_sha256_value="graph",
                    build_marker=build_marker,
                    verify_marker=verify_marker,
                )

    def test_verifier_rejects_sgs_columns_and_provenance(self) -> None:
        for name in (
            "Ms_favre",
            "u_favre_x",
            "reynolds_favre_xy",
            "ux_bx_mean_volume",
            "magnetic_covariance_xy",
            "internal_energy_mean",
            "Ms_favre_flags",
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "forbidden SGS-derived columns"):
                _require_primary_only_catalog(
                    {name: np.asarray([1.0])},
                    {},
                    "synthetic",
                )
        for payload in (
            {"sgs_cache": "raw_mhd_sgs_80.npz"},
            {"product": "mhd_sgs"},
            {"nested": {"sgs_manifest": "stale"}},
            {"source": "sgs_snapshot"},
        ):
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, "forbidden SGS provenance metadata"):
                _require_no_sgs_provenance(payload, "synthetic")

    def test_verifier_rejects_unknown_catalog_aliases(self) -> None:
        base = {name: np.asarray([1.0]) for name in EXPECTED_PRIMARY_ONLY_CATALOG_COLUMNS}
        for name in ("sonic_mach", "velocity_dispersion", "broken_extra"):
            columns = dict(base)
            columns[name] = np.asarray([1.0])
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "catalog schema mismatch"):
                _require_primary_only_catalog(columns, {}, "synthetic")

    def test_verifier_rejects_extra_raw_payload_field(self) -> None:
        raw = {name: np.asarray([1.0]) for name in PRIMARY_LABELS}
        _require_primary_raw_cache(raw, {}, "synthetic")
        raw["broken_extra"] = np.asarray([1.0])
        with self.assertRaisesRegex(ValueError, "raw payload fields do not match primary schema"):
            _require_primary_raw_cache(raw, {}, "synthetic")

    def test_verifier_rejects_coordinated_formula_mutation(self) -> None:
        columns = _primary_formula_columns()
        _verify_primary_formulas(columns, "synthetic")
        columns["B_mean"] = np.asarray([4.0])
        columns["deltaB"] = np.asarray([4.0])
        columns["dBB"] = np.asarray([1.0])
        with self.assertRaisesRegex(ValueError, "B_mean: formula mismatch"):
            _verify_primary_formulas(columns, "synthetic")

    def test_verifier_rejects_nan_suppression(self) -> None:
        for names in (("dBB",), ("deltaB_sq", "deltaB", "deltaB_sq_over_B2_mean", "dBB")):
            columns = _primary_formula_columns()
            for name in names:
                columns[name] = np.asarray([np.nan])
            with self.subTest(names=names), self.assertRaisesRegex(ValueError, "availability mismatch"):
                _verify_primary_formulas(columns, "synthetic")

    def test_verifier_rejects_coordinated_density_metric_mutation(self) -> None:
        columns = _primary_formula_columns()
        columns["dens_sigma"] = np.asarray([0.75])
        columns["rho_sigma_over_mean"] = np.asarray([0.375])
        with self.assertRaisesRegex(ValueError, "dens_sigma: formula mismatch"):
            _verify_primary_formulas(columns, "synthetic")

    def test_verifier_rejects_flag_and_aggregate_mutation(self) -> None:
        columns = _primary_formula_columns()
        for name in FLAG_COLUMNS:
            columns[name] = np.zeros(1, dtype=np.uint8)
        columns["catalog_validity_flags"] = np.zeros(1, dtype=np.uint32)
        metadata = {"catalog_validity_flag_bits": {name: bit for bit, name in enumerate(FLAG_COLUMNS)}}
        _verify_primary_flags(columns, metadata, "synthetic")
        columns["dBB_flags"] = np.ones(1, dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "dBB_flags: exact-value mismatch"):
            _verify_primary_flags(columns, metadata, "synthetic")
        columns["dBB_flags"] = np.zeros(1, dtype=np.uint8)
        columns["catalog_validity_flags"] = np.ones(1, dtype=np.uint32)
        with self.assertRaisesRegex(ValueError, "catalog_validity_flags: exact-value mismatch"):
            _verify_primary_flags(columns, metadata, "synthetic")

    def test_verifier_rejects_spatial_grid_mutation(self) -> None:
        for name in (
            "center_i", "grid_i", "source_cbin_i0", "source_cbin_i1",
            "parent_L_sub", "parent_subvolume_id", "cell_i0", "cell_i1", "x1_min", "x1_max",
        ):
            columns = _expected_grid_columns(1280)
            columns[name] = columns[name].copy()
            columns[name][0] += 1
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, f"exact grid mismatch for {name}"):
                _verify_expected_grid(columns, 1280, "synthetic")

    def test_retained_validation_diagnostics_are_primary_only(self) -> None:
        moments = {
            f"{base}_{suffix}": value
            for base in ("dens", "mom1", "mom2", "mom3", "ener", "bcc1", "bcc2", "bcc3")
            for suffix, value in zip(("1st", "2nd", "3rd", "4th"), (2.0, 5.0, 14.0, 41.0))
        }
        diagnostics = retained_primary_diagnostics(moments)
        self.assertEqual(diagnostics["dens_variance"], 1.0)
        self.assertEqual(diagnostics["dens_sigma"], 1.0)
        self.assertEqual(diagnostics["rho_sigma_over_mean"], 0.5)
        self.assertEqual(diagnostics["u_mass_weighted_mean_x"], 1.0)
        self.assertEqual(diagnostics["magnetic_energy_mean"], 7.5)
        self.assertNotIn("velocity_dispersion", diagnostics)
        self.assertNotIn("sonic_mach", diagnostics)

    def test_validation_precision_limited_waiver_is_narrow(self) -> None:
        moments = {
            f"{base}_{suffix}": 1.0
            for base in ("dens", "mom1", "mom2", "mom3", "ener", "bcc1", "bcc2", "bcc3")
            for suffix in ("1st", "2nd", "3rd", "4th")
        }
        bounds = {name: 1.0e-3 for name in moments}
        precision_limited: set[str] = set()
        diagnostics = retained_primary_diagnostics(
            moments,
            raw_moment_error_bounds=bounds,
            precision_limited_unavailable=precision_limited,
        )
        self.assertTrue(np.isnan(diagnostics["dens_skewness"]))
        self.assertTrue(np.isnan(diagnostics["dens_kurtosis"]))
        self.assertIn("dens_skewness", precision_limited)
        self.assertIn("dens_kurtosis", precision_limited)
        self.assertNotIn("dens_sigma", precision_limited)

        rows = comparison_rows(
            "retained",
            {"dens_skewness": 1.0},
            {"dens_skewness": float("nan")},
            atol=1.0e-6,
            rtol=1.0e-6,
            precision_limited_unavailable={"dens_skewness"},
        )
        self.assertTrue(rows[0]["passed"])
        self.assertTrue(rows[0]["precision_limited_unavailable"])
        rows = comparison_rows(
            "retained",
            {"dens_skewness": float("nan")},
            {"dens_skewness": float("nan")},
            atol=1.0e-6,
            rtol=1.0e-6,
            paired_nan_allowed_names={"dens_skewness"},
        )
        self.assertTrue(rows[0]["passed"])
        self.assertTrue(rows[0]["paired_nan_unavailable"])
        rows = comparison_rows(
            "retained",
            {"dens_skewness": 1.0},
            {"dens_skewness": 1.2},
            atol=1.0e-6,
            rtol=1.0e-6,
            reconstructed_error_bounds={"dens_skewness": 0.3},
        )
        self.assertTrue(rows[0]["passed"])
        self.assertTrue(rows[0]["writer_uncertainty_bound_applied"])
        rows = comparison_rows(
            "retained-exact",
            {"dens_sigma": 1.0},
            {"dens_sigma": 2.0},
            atol=1.0e-6,
            rtol=1.0e-6,
            reconstructed_error_bounds={"dens_sigma": 10.0},
        )
        self.assertFalse(rows[0]["passed"])
        self.assertFalse(rows[0]["writer_uncertainty_bound_applied"])
        rows = comparison_rows(
            "invented-standardized",
            {"invented_skewness": 1.0},
            {"invented_skewness": 2.0},
            atol=1.0e-6,
            rtol=1.0e-6,
            reconstructed_error_bounds={"invented_skewness": 10.0},
        )
        self.assertFalse(rows[0]["passed"])
        self.assertFalse(rows[0]["writer_uncertainty_bound_applied"])
        for expected, reconstructed, eligible in (
            ({"dens_sigma": 1.0}, {"dens_sigma": float("nan")}, {"dens_skewness"}),
            ({"dens_skewness": 1.0}, {"dens_skewness": 2.0}, {"dens_skewness"}),
            ({"dens_1st": 1.0}, {"dens_1st": float("nan")}, None),
            ({"dens_1st": float("nan")}, {"dens_1st": float("nan")}, None),
            ({"ener_mean": float("nan")}, {"ener_mean": float("nan")}, None),
        ):
            rows = comparison_rows(
                "strict",
                expected,
                reconstructed,
                atol=1.0e-6,
                rtol=1.0e-6,
                precision_limited_unavailable=eligible,
            )
            self.assertFalse(rows[0]["passed"])
        invalid_moments = {
            f"{base}_{suffix}": value
            for base in ("dens", "mom1", "mom2", "mom3", "ener", "bcc1", "bcc2", "bcc3")
            for suffix, value in zip(("1st", "2nd", "3rd", "4th"), (1.0, 1.0, 1.0, 0.0))
        }
        invalid_bounds = {name: 1.0e-3 for name in invalid_moments}
        invalid_precision_limited: set[str] = set()
        invalid_diagnostics = retained_primary_diagnostics(
            invalid_moments,
            raw_moment_error_bounds=invalid_bounds,
            precision_limited_unavailable=invalid_precision_limited,
        )
        self.assertTrue(np.isnan(invalid_diagnostics["ener_kurtosis"]))
        self.assertNotIn("ener_kurtosis", invalid_precision_limited)
        rows = comparison_rows(
            "invalid-standardized",
            {"ener_kurtosis": float("nan")},
            {"ener_kurtosis": float("nan")},
            atol=1.0e-6,
            rtol=1.0e-6,
            paired_nan_allowed_names=invalid_precision_limited,
        )
        self.assertFalse(rows[0]["passed"])

    def test_regional_error_bounds_include_merge_reduction(self) -> None:
        bounds = _regional_error_bounds({"dens_1st": 2.0}, {"dens_1st": 10.0}, 2)
        self.assertGreater(bounds["dens_1st"], 1.0)
        self.assertAlmostEqual(bounds["dens_1st"], (2.0 + _rounding_gamma(2) * 10.0) / 2.0)


if __name__ == "__main__":
    unittest.main()
