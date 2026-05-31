#!/usr/bin/env python3
"""Run the Phase 4 Batch A report compatibility chain without extraction-array reads."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from scripts.phase4 import generate_phase4_batch_a_status_figures as report


class StructuralInputHashes(report.InputHashes):
    """Record declared bindings without re-reading large payload files."""

    def bind_verified(self, path: Path, sha256: str) -> None:
        path = path.resolve()
        if not path.is_file():
            raise RuntimeError(f"required retained artifact is missing: {path}")
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise RuntimeError(f"invalid retained SHA-256 binding for {path}")
        self._hashes[str(path)] = f"declared-structural-preflight-only:{sha256}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(report.inherited._json_builtin(payload), indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exercise Phase 4 Batch A report compatibility without extraction-array "
            "SHA-256 recomputation or primitive-array traversal."
        )
    )
    parser.add_argument("--phase1-root", type=Path, required=True)
    parser.add_argument("--extraction-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--ledger-summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output_json.resolve()
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite immutable preflight output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_summary, _ = report._bind_ledger_summary_snapshot(
        args.ledger_summary, report.InputHashes()
    )
    hashes = StructuralInputHashes()
    verified, manifests = report.verify_inputs(
        phase1_root=args.phase1_root,
        extraction_root=args.extraction_root,
        release_root=args.release_root,
        ledger_summary=ledger_summary,
        input_hashes=hashes,
    )
    groups = report._load_groups(verified, hashes)
    catalog_rows, phase1_metadata = report._phase1_catalog_rows(verified, hashes)
    pressure_convention, gaps = report._pressure_convention(verified, hashes)
    operational_rows, sampler_resources, resource_gaps = report._operational_rows(
        verified, manifests, groups, hashes
    )
    examples = report._select_examples(catalog_rows)
    curve_rows, correlations = report._curve_scale_rows(verified, catalog_rows, groups)
    with tempfile.TemporaryDirectory(prefix=".phase4-report-plot-preflight-", dir=output_path.parent) as tmp:
        plot_output = Path(tmp)
        report.workflow_schematic(plot_output)
        report.dbb_census(catalog_rows, examples, plot_output)
        report.support_vs_ell(verified, groups, plot_output)
        report.representative_curves_with_bands(groups, examples, plot_output)
        report.curve_ratio_census(verified, groups, plot_output)
        report.matched_pair_curve_comparison(catalog_rows, groups, plot_output)
        report.local_slope_effective_block_diagnostic(groups, examples, plot_output)
        report.environment_trend_summary(curve_rows, correlations, plot_output)
        report.runtime_storage_summary(verified, catalog_rows, operational_rows, plot_output)
        rendered_nonprimitive_figure_count = len(tuple(plot_output.glob("*.png")))
    payload = {
        "schema_version": 1,
        "status": "passed",
        "purpose": (
            "Structural compatibility preflight only. Extraction-array SHA-256 declarations "
            "are shape- and path-checked but not recomputed here. Reduction payload bindings "
            "remain verified. Primitive arrays are not traversed."
        ),
        "extraction_array_sha256_recomputation_performed": False,
        "reduction_payload_bindings_verified": True,
        "primitive_array_traversal_performed": False,
        "nonprimitive_plot_render_compatibility_performed": True,
        "rendered_nonprimitive_figure_count": rendered_nonprimitive_figure_count,
        "skipped_plot_requires_primitive_array_traversal": (
            "phase4_batch_a_representative_extraction_slice_montage.png"
        ),
        "full_publication_still_required": True,
        "cube_count": len(verified.cube_ids),
        "manifest_count": len(manifests),
        "group_count": len(groups),
        "catalog_row_count": len(catalog_rows),
        "curve_scale_row_count": len(curve_rows),
        "correlation_count": len(correlations),
        "representative_examples": examples,
        "phase1_catalog_metadata_present": bool(phase1_metadata),
        "pressure_convention": pressure_convention,
        "gaps": [*gaps, *resource_gaps],
        "sampler_resources": sampler_resources,
        "operational_cube_count": len(operational_rows),
        "structural_input_bindings": hashes.as_dict(),
    }
    with tempfile.TemporaryDirectory(prefix=f".{output_path.name}.", dir=output_path.parent) as tmp:
        temporary = Path(tmp) / output_path.name
        _write_json(temporary, payload)
        temporary.rename(output_path)
    print(f"Wrote Phase 4 Batch A report structural preflight: {output_path}")


if __name__ == "__main__":
    main()
