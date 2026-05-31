#!/usr/bin/env python3
"""Generate PHASE1_STATUS_UPDATE.md figures from the verified Phase 1 artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import NullFormatter
import numpy as np


DEFAULT_RUN_DIR = Path(
    "/lustre/orion/ast207/proj-shared/dfielding/Production_plm/"
    "prompt1_catalog/t6_final_primary_20260530"
)
DEFAULT_OUTPUT_DIR = Path("figures/phase1_status_update")
L_SUB_VALUES = (80, 160, 320, 640, 1280)
COLORS = {
    80: "#4c78a8",
    160: "#72b7b2",
    320: "#54a24b",
    640: "#f58518",
    1280: "#e45756",
}
CORRELATION_PROPERTIES = (
    "dBB",
    "B_mean",
    "deltaB",
    "B_rms",
    "magnetic_energy_mean",
    "vA_mean_proxy",
    "vA_rms_like_proxy",
    "u_mass_weighted_mean_x",
    "u_mass_weighted_mean_y",
    "u_mass_weighted_mean_z",
    "dens_mean",
    "rho_sigma_over_mean",
    "dens_skewness",
    "dens_kurtosis",
    "mom1_sigma",
    "mom2_sigma",
    "mom3_sigma",
    "ener_mean",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def load_catalog(run_dir: Path, l_sub: int, names: Iterable[str]) -> dict[str, np.ndarray]:
    path = run_dir / "catalogs" / f"catalog_L{l_sub}.npz"
    with np.load(path) as payload:
        return {name: np.array(payload[name]) for name in names}


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    size: tuple[float, float],
    text: str,
    *,
    facecolor: str,
    edgecolor: str = "#333333",
    textcolor: str = "#111111",
    linestyle: str = "-",
) -> None:
    x, y = xy
    width, height = size
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.035",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.5,
        linestyle=linestyle,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=9,
        color=textcolor,
        linespacing=1.35,
    )


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.4,
            color="#555555",
        )
    )


def make_workflow_overview(output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    boxes = [
        ((0.3, 2.35), (1.65, 1.0), "Primary cbin\n$mhd\\_u\\_bcc$\nraw moments", "#d9edf7"),
        ((2.45, 2.35), (1.65, 1.0), "Strict streamed\n$L=80$ cache", "#d9edf7"),
        ((4.6, 2.35), (1.65, 1.0), "Five tiled\ncatalogs\n$L=80...1280$", "#d9edf7"),
        ((6.75, 2.35), (1.65, 1.0), "Structured\nsecond-pass\nverification", "#dff0d8"),
        ((8.9, 2.35), (1.65, 1.0), "Census analysis\nand pilot\nproposal", "#dff0d8"),
    ]
    for xy, size, text, color in boxes:
        add_box(ax, xy, size, text, facecolor=color)
    for x in (1.95, 4.1, 6.25, 8.4):
        add_arrow(ax, (x, 2.85), (x + 0.45, 2.85))

    add_box(
        ax,
        (1.75, 0.45),
        (2.4, 0.9),
        "Direct primitive checks\nplus cbin hierarchy checks",
        facecolor="#fff2cc",
    )
    add_arrow(ax, (2.95, 1.35), (2.95, 2.35))
    add_box(
        ax,
        (5.2, 0.45),
        (2.35, 0.9),
        "Broken SGS products\nexplicitly excluded",
        facecolor="#f4cccc",
        edgecolor="#b22222",
        textcolor="#7a0000",
        linestyle="--",
    )
    add_arrow(ax, (6.4, 1.35), (6.4, 2.05))
    ax.text(6.4, 1.73, "not used", ha="center", va="center", fontsize=8, color="#7a0000")
    ax.text(
        0.3,
        4.05,
        "Phase 1 primary-only census workflow (schematic)",
        fontsize=13,
        weight="bold",
    )
    ax.text(
        0.3,
        3.75,
        "Blue: build path. Green: audited outputs. Yellow: validation gate. Red: intentionally excluded input.",
        fontsize=9,
        color="#444444",
    )
    return save(fig, output_dir, "workflow_overview_schematic.png")


def make_domain_tiling_schematic(output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.2, 1.0]})
    ax = axes[0]
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle((0.08, 0.08), 0.82, 0.82, fill=False, linewidth=2, edgecolor="#333333"))
    for pos in np.linspace(0.08, 0.90, 9):
        ax.plot([pos, pos], [0.08, 0.90], color="#e45756", linewidth=0.65, alpha=0.75)
        ax.plot([0.08, 0.90], [pos, pos], color="#e45756", linewidth=0.65, alpha=0.75)
    for pos in np.linspace(0.08, 0.90, 17):
        ax.plot([pos, pos], [0.08, 0.90], color="#4c78a8", linewidth=0.28, alpha=0.45)
        ax.plot([0.08, 0.90], [pos, pos], color="#4c78a8", linewidth=0.28, alpha=0.45)
    ax.text(0.49, 0.96, "One domain face (schematic)", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(0.49, 0.02, "Red grid: $L_{sub}=1280$; blue grid: $L_{sub}=640$", ha="center", fontsize=9)
    ax.text(
        0.49,
        -0.045,
        "The actual census tiles the periodic $10240^3$ domain at five exact scales.",
        ha="center",
        fontsize=8,
        color="#555555",
    )

    ax = axes[1]
    ax.axis("off")
    ax.text(0.02, 0.96, "Exact census hierarchy", fontsize=12, weight="bold", va="top")
    rows = [
        ("$L_{sub}/\\Delta x$", "cubes", "cubes per axis"),
        ("80", "2,097,152", "128"),
        ("160", "262,144", "64"),
        ("320", "32,768", "32"),
        ("640", "4,096", "16"),
        ("1280", "512", "8"),
    ]
    y = 0.82
    for row_index, row in enumerate(rows):
        color = "#eeeeee" if row_index == 0 else "#ffffff"
        ax.add_patch(Rectangle((0.0, y - 0.08), 0.98, 0.105, facecolor=color, edgecolor="#cccccc"))
        for x, text in zip((0.06, 0.42, 0.78), row):
            ax.text(x, y - 0.025, text, fontsize=9, weight="bold" if row_index == 0 else "normal")
        y -= 0.115
    ax.text(
        0.02,
        0.08,
        "The $L_{sub}=640$ pilot scale retains 4,096 choices\nwhile limiting each later extraction to 16 primitive rank files.",
        fontsize=9,
        linespacing=1.4,
    )
    return save(fig, output_dir, "domain_tiling_schematic.png")


def make_dbb_slice(run_dir: Path, output_dir: Path) -> Path:
    dbb = load_catalog(run_dir, 80, ("dBB",))["dBB"].reshape((128, 128, 128))
    image = dbb[64]
    vmin, vmax = np.quantile(image, [0.01, 0.99])
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    plotted = ax.imshow(
        image,
        origin="lower",
        cmap="magma",
        norm=LogNorm(vmin=float(vmin), vmax=float(vmax)),
        interpolation="nearest",
        extent=(0, 10240, 0, 10240),
    )
    ax.set_xlabel("$x_1 / \\Delta x$")
    ax.set_ylabel("$x_2 / \\Delta x$")
    ax.set_title("$dBB$ on one $L_{sub}=80$ census layer with $k_0=5120$")
    colorbar = fig.colorbar(plotted, ax=ax, pad=0.02)
    colorbar.set_label("$dBB = \\delta B / B_{mean}$")
    return save(fig, output_dir, "dBB_L80_midplane_map.png")


def quantile_rows(run_dir: Path, prop: str) -> list[dict[str, str]]:
    rows = read_csv(run_dir / "analysis" / "distribution_quantiles.csv")
    return [row for row in rows if row["property"] == prop]


def make_dbb_quantile_trend(run_dir: Path, output_dir: Path) -> Path:
    rows = quantile_rows(run_dir, "dBB")
    x = np.array([int(row["L_sub"]) for row in rows])
    q16 = np.array([float(row["q0.16"]) for row in rows])
    q50 = np.array([float(row["q0.5"]) for row in rows])
    q84 = np.array([float(row["q0.84"]) for row in rows])
    q95 = np.array([float(row["q0.95"]) for row in rows])
    counts = [int(row["count"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.fill_between(x, q16, q84, color="#4c78a8", alpha=0.22, label="16th-84th percentile")
    ax.plot(x, q50, "o-", color="#1f4e79", linewidth=2.3, label="median")
    ax.plot(x, q95, "s--", color="#e45756", linewidth=1.6, label="95th percentile")
    for xi, yi, count in zip(x, q50, counts):
        ax.annotate(f"$N={count:,}$", (xi, yi), xytext=(0, -18), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(x, [str(value) for value in x])
    ax.set_xlabel("$L_{sub}/\\Delta x$")
    ax.set_ylabel("$dBB = \\delta B / B_{mean}$")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25, which="both")
    return save(fig, output_dir, "dBB_quantile_trend_by_Lsub.png")


def make_dbb_distributions(run_dir: Path, output_dir: Path) -> Path:
    bins = np.geomspace(0.02, 150.0, 110)
    centers = np.sqrt(bins[:-1] * bins[1:])
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for l_sub in L_SUB_VALUES:
        dbb = load_catalog(run_dir, l_sub, ("dBB",))["dBB"]
        hist, _ = np.histogram(dbb, bins=bins, density=True)
        ax.plot(centers, hist, color=COLORS[l_sub], linewidth=1.8, label=f"$L_{{sub}}={l_sub}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$dBB = \\delta B / B_{mean}$")
    ax.set_ylabel("Probability density")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.22, which="both")
    return save(fig, output_dir, "dBB_distribution_by_Lsub.png")


def make_magnetic_correlation(run_dir: Path, output_dir: Path) -> Path:
    payload = load_catalog(run_dir, 640, ("dBB", "B_mean", "deltaB"))
    rows = read_csv(run_dir / "analysis" / "spearman_correlations.csv")
    corr = {
        row["property_2"]: float(row["spearman_r"])
        for row in rows
        if int(row["L_sub"]) == 640 and row["property_1"] == "dBB"
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for ax, prop, label in zip(axes, ("B_mean", "deltaB"), ("$B_{mean}$", "$\\delta B$")):
        plotted = ax.hexbin(
            payload[prop],
            payload["dBB"],
            gridsize=45,
            bins="log",
            xscale="log",
            yscale="log",
            mincnt=1,
            cmap="viridis",
        )
        ax.set_xlabel(label)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.text(
            0.04,
            0.94,
            f"Spearman $r_s={corr[prop]:.5f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        )
        fig.colorbar(plotted, ax=ax, pad=0.02, label="subvolumes per hexbin")
        ax.grid(alpha=0.16, which="both")
    axes[0].set_ylabel("$dBB = \\delta B / B_{mean}$")
    fig.suptitle("$L_{sub}=640$: $dBB$ associations with $B_{mean}$ and $\\delta B$")
    return save(fig, output_dir, "dBB_magnetic_correlations_L640.png")


def make_environment_quantile_trends(run_dir: Path, output_dir: Path) -> Path:
    x = np.array(L_SUB_VALUES)
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.4), sharex=True)
    properties = (
        ("dens_mean", "$\\langle\\rho\\rangle_V$", False),
        ("rho_sigma_over_mean", "$\\sigma_\\rho/\\langle\\rho\\rangle_V$", True),
        ("dens_skewness", "density skewness", False),
        ("dens_kurtosis", "density kurtosis", False),
    )
    for ax, (prop, label, log_y) in zip(axes.flat, properties):
        rows = quantile_rows(run_dir, prop)
        q16 = np.array([float(row["q0.16"]) for row in rows])
        q50 = np.array([float(row["q0.5"]) for row in rows])
        q84 = np.array([float(row["q0.84"]) for row in rows])
        ax.fill_between(x, q16, q84, color="#72b7b2", alpha=0.28)
        ax.plot(x, q50, "o-", color="#1f4e79", linewidth=2)
        ax.set_ylabel(label)
        if log_y:
            ax.set_yscale("log")
        ax.grid(alpha=0.25, which="both")

    momentum_ax = axes.flat[4]
    velocity_ax = axes.flat[5]
    component_colors = ("#4c78a8", "#f58518", "#54a24b")
    for axis, prefix, ylabel, title in (
        (momentum_ax, "mom", "conserved-momentum $\\sigma$", "Not primitive $\\delta u$"),
        (
            velocity_ax,
            "u_mass_weighted_mean_",
            "$\\langle u_i\\rangle_\\rho$",
            "Exact coarse mass-weighted means",
        ),
    ):
        for component, color in zip(("x", "y", "z"), component_colors):
            prop = f"{prefix}{'123'[('x', 'y', 'z').index(component)]}_sigma" if prefix == "mom" else f"{prefix}{component}"
            rows = quantile_rows(run_dir, prop)
            q16 = np.array([float(row["q0.16"]) for row in rows])
            q50 = np.array([float(row["q0.5"]) for row in rows])
            q84 = np.array([float(row["q0.84"]) for row in rows])
            axis.fill_between(x, q16, q84, color=color, alpha=0.08)
            axis.plot(x, q50, "o-", color=color, linewidth=1.6, label=f"${component}$")
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontsize=9)
        axis.grid(alpha=0.25, which="both")
        axis.legend(fontsize=8, ncol=3)

    for ax in axes.flat:
        ax.set_xscale("log", base=2)
        ax.set_xticks(x, [str(value) for value in x])
        ax.set_xlabel("$L_{sub}/\\Delta x$")
    fig.suptitle("Supported non-magnetic environmental summaries across census scales")
    fig.tight_layout()
    return save(fig, output_dir, "environmental_quantile_trends_by_Lsub.png")


def make_nonmagnetic_dbb_correlations(run_dir: Path, output_dir: Path) -> Path:
    properties = (
        ("dens_mean", "$\\langle\\rho\\rangle_V$"),
        ("rho_sigma_over_mean", "$\\sigma_\\rho/\\langle\\rho\\rangle_V$"),
        ("dens_skewness", "density skewness"),
        ("mom1_sigma", "$\\sigma_{\\rho u_x}$"),
        ("mom2_sigma", "$\\sigma_{\\rho u_y}$"),
        ("mom3_sigma", "$\\sigma_{\\rho u_z}$"),
    )
    names = ("dBB", *(prop for prop, _ in properties))
    payload = load_catalog(run_dir, 640, names)
    rows = read_csv(run_dir / "analysis" / "spearman_correlations.csv")
    corr = {
        row["property_2"]: float(row["spearman_r"])
        for row in rows
        if int(row["L_sub"]) == 640 and row["property_1"] == "dBB"
    }
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 8.1), sharey=True)
    for ax, (prop, label) in zip(axes.flat, properties):
        values = payload[prop]
        positive_x = np.all(values > 0.0) and prop != "dens_mean"
        plotted = ax.hexbin(
            values,
            payload["dBB"],
            gridsize=42,
            bins="log",
            xscale="log" if positive_x else "linear",
            yscale="log",
            mincnt=1,
            cmap="viridis",
        )
        ax.set_xlabel(label)
        if positive_x:
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.text(
            0.04,
            0.94,
            f"Spearman $r_s={corr[prop]:.5f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        )
        fig.colorbar(plotted, ax=ax, pad=0.02, label="subvolumes per hexbin")
        ax.grid(alpha=0.16, which="both")
    axes[0, 0].set_ylabel("$dBB = \\delta B / B_{mean}$")
    axes[1, 0].set_ylabel("$dBB = \\delta B / B_{mean}$")
    fig.suptitle("$L_{sub}=640$: supported non-magnetic associations with $dBB$")
    fig.tight_layout()
    return save(fig, output_dir, "dBB_nonmagnetic_correlations_L640.png")


def make_correlation_matrix(run_dir: Path, output_dir: Path) -> Path:
    rows = read_csv(run_dir / "analysis" / "spearman_correlations.csv")
    selected = [row for row in rows if int(row["L_sub"]) == 640]
    lookup = {
        (row["property_1"], row["property_2"]): float(row["spearman_r"])
        for row in selected
    }
    matrix = np.array([
        [lookup[(row, column)] for column in CORRELATION_PROPERTIES]
        for row in CORRELATION_PROPERTIES
    ])
    labels = (
        "dBB",
        "Bmean",
        "deltaB",
        "Brms",
        "Emag",
        "vAmean proxy",
        "vArms proxy",
        "<ux>rho",
        "<uy>rho",
        "<uz>rho",
        "<rho>",
        "sigmarho/<rho>",
        "rho skew",
        "rho kurt",
        "sigma(rho ux)",
        "sigma(rho uy)",
        "sigma(rho uz)",
        "<Etot>",
    )
    fig, ax = plt.subplots(figsize=(8.7, 7.5))
    image = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_title("$L_{sub}=640$ supported-property Spearman matrix")
    fig.colorbar(image, ax=ax, pad=0.02, label="Spearman rank correlation")
    fig.tight_layout()
    return save(fig, output_dir, "spearman_correlation_matrix_L640.png")


def make_conditional_bands(run_dir: Path, output_dir: Path) -> Path:
    rows = read_csv(run_dir / "analysis" / "conditional_dBB_bands.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
    for ax, prop, label in zip(axes, ("B_mean", "deltaB"), ("$B_{mean}$", "$\\delta B$")):
        selected = [
            row for row in rows if int(row["L_sub"]) == 640 and row["property"] == prop
        ]
        selected.sort(key=lambda row: int(row["property_quantile_bin"]))
        x = np.arange(1, 6)
        lo = np.array([float(row["dBB_q16"]) for row in selected])
        med = np.array([float(row["dBB_median"]) for row in selected])
        hi = np.array([float(row["dBB_q84"]) for row in selected])
        ax.fill_between(x, lo, hi, color="#72b7b2", alpha=0.3)
        ax.plot(x, med, "o-", color="#1f4e79", linewidth=2)
        ax.set_xlabel(f"{label} quintile (low to high)")
        ax.set_xticks(x)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("conditional $dBB$ median and 16th-84th percentile")
    fig.suptitle("$L_{sub}=640$ conditional magnetic census")
    return save(fig, output_dir, "conditional_dBB_magnetic_quintiles_L640.png")


def make_cross_scale_persistence(run_dir: Path, output_dir: Path) -> Path:
    rows = read_csv(run_dir / "analysis" / "cross_scale_dBB.csv")
    x = np.array([int(row["L_sub"]) for row in rows])
    child_mean = np.array([float(row["coarse_dBB_vs_child_mean_spearman"]) for row in rows])
    child_median = np.array([float(row["coarse_dBB_vs_child_median_spearman"]) for row in rows])
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.plot(x, child_mean, "o-", color="#4c78a8", linewidth=2, label="coarse vs $L=80$ descendant mean")
    ax.plot(x, child_median, "s--", color="#e45756", linewidth=2, label="coarse vs $L=80$ descendant median")
    ax.set_xscale("log", base=2)
    ax.set_xticks(x, [str(value) for value in x])
    ax.set_ylim(0.68, 1.01)
    ax.set_xlabel("$L_{sub}/\\Delta x$")
    ax.set_ylabel("Spearman $r_s$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return save(fig, output_dir, "cross_scale_dBB_persistence.png")


def make_validation_residuals(run_dir: Path, output_dir: Path) -> Path:
    results = read_json(run_dir / "validation" / "validation_results.json")
    cases = list(results["cases"])
    names = [str(row["case"]) for row in cases]
    values = [float(row["max_relative_difference"]) for row in cases]
    fig, ax = plt.subplots(figsize=(8.5, 4.7))
    bars = ax.barh(names, values, color="#54a24b")
    ax.set_xscale("log")
    ax.set_xlabel("maximum raw-moment relative difference")
    ax.grid(alpha=0.25, axis="x", which="both")
    for bar, value in zip(bars, values):
        ax.text(value * 1.08, bar.get_y() + bar.get_height() / 2, f"{value:.2e}", va="center", fontsize=8)
    ax.set_xlim(min(values) / 2, max(values) * 5)
    ax.set_title("Direct primitive-to-cbin reconstruction checks: all cases passed")
    return save(fig, output_dir, "validation_raw_moment_residuals.png")


def make_flagged_fraction(run_dir: Path, output_dir: Path) -> Path:
    verification = read_json(run_dir / "verification" / "verification_results.json")
    catalogs = list(verification["catalogs"])
    x = np.array([int(row["L_sub"]) for row in catalogs])
    flagged = np.array([int(row["validity_flagged_rows"]) for row in catalogs])
    total = np.array([int(row["rows"]) for row in catalogs])
    fraction = flagged / total
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.plot(x, np.maximum(fraction, 1.0e-7), "o-", color="#e45756", linewidth=2)
    for xi, yi, n_flagged, n_total in zip(x, np.maximum(fraction, 1.0e-7), flagged, total):
        ax.annotate(f"{n_flagged:,} / {n_total:,}", (xi, yi), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(x, [str(value) for value in x])
    ax.set_xlabel("$L_{sub}/\\Delta x$")
    ax.set_ylabel("rows with $\\geq 1$ diagnostic flag / all rows")
    ax.set_ylim(5.0e-8, 0.3)
    ax.grid(alpha=0.25, which="both")
    ax.text(1280, 1.2e-7, "zero flagged rows", ha="right", fontsize=8)
    ax.text(0.03, 0.06, "All dBB rows are finite and unflagged.", transform=ax.transAxes, fontsize=8)
    return save(fig, output_dir, "catalog_flagged_fraction_by_Lsub.png")


def pilot_role_group(role: str) -> str:
    if role.startswith("representative:"):
        return "representative"
    if role.startswith("matched:"):
        return "matched pair"
    return "targeted outlier"


def make_pilot_selection(run_dir: Path, output_dir: Path) -> Path:
    catalog = load_catalog(run_dir, 640, ("B_mean", "deltaB", "dBB"))
    pilot_rows = read_csv(run_dir / "analysis" / "pilot_sample.csv")
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    plotted = ax.hexbin(
        catalog["B_mean"],
        catalog["deltaB"],
        C=catalog["dBB"],
        reduce_C_function=np.median,
        gridsize=48,
        xscale="log",
        yscale="log",
        mincnt=1,
        cmap="viridis",
        norm=LogNorm(),
    )
    styles = {
        "representative": ("o", "#ffffff"),
        "matched pair": ("s", "#f58518"),
        "targeted outlier": ("*", "#e45756"),
    }
    for group, (marker, color) in styles.items():
        rows = [row for row in pilot_rows if pilot_role_group(row["role"]) == group]
        ax.scatter(
            [float(row["B_mean"]) for row in rows],
            [float(row["deltaB"]) for row in rows],
            marker=marker,
            s=78 if marker != "*" else 150,
            facecolors=color,
            edgecolors="#222222",
            linewidths=0.8,
            label=f"{group} ({len(rows)})",
        )
    ax.set_xlabel("$B_{mean}$")
    ax.set_ylabel("$\\delta B$")
    ax.legend(fontsize=8, loc="upper left")
    colorbar = fig.colorbar(plotted, ax=ax, pad=0.02)
    colorbar.set_label("median $dBB$ in census hexbin (log scale)")
    ax.grid(alpha=0.18, which="both")
    ax.set_title("$L_{sub}=640$ census and proposed 21-region extraction pilot")
    return save(fig, output_dir, "pilot_selection_L640.png")


def make_extraction_estimates(run_dir: Path, output_dir: Path) -> Path:
    verification = read_json(run_dir / "verification" / "verification_results.json")
    catalogs = list(verification["catalogs"])
    x = np.array([int(row["L_sub"]) for row in catalogs])
    ranks = np.array([int(row["required_rank_count_max"]) for row in catalogs])
    payload_gib = 8.0 * 4.0 * x.astype(float) ** 3 / 1024.0**3
    rank0_bytes = int(
        read_json(run_dir / "validation" / "VALIDATION_COMPLETE.json")["full_snapshot_identity"]["rank0_file_size"]
    )
    read_gib = ranks * rank0_bytes / 1024.0**3
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(x, payload_gib, "o-", color="#4c78a8", linewidth=2, label="eight-field cube payload")
    ax.plot(x, read_gib, "s--", color="#e45756", linewidth=2, label="full primitive rank-file reads")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(x, [str(value) for value in x])
    ax.set_xlabel("$L_{sub}/\\Delta x$")
    ax.set_ylabel("GiB per proposed extraction")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(x, ranks, "d:", color="#54a24b", linewidth=1.8, label="rank files")
    ax2.set_yscale("log")
    ax2.set_ylabel("primitive rank files touched")
    ax2.legend(fontsize=8, loc="lower right")
    ax.set_title("Extraction planning estimates, not measured end-to-end benchmarks")
    return save(fig, output_dir, "pilot_extraction_scaling_estimates.png")


def make_primary_only_change_schematic(output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4.0)
    ax.axis("off")
    ax.text(0.2, 3.65, "Workflow hardening after the SGS data-quality finding (schematic)", fontsize=13, weight="bold")
    add_box(ax, (0.35, 2.0), (3.4, 1.0), "Earlier attempt\nprimary moments + SGS-derived channels", facecolor="#f4cccc", edgecolor="#b22222")
    add_box(ax, (6.65, 2.0), (3.4, 1.0), "Trusted final workflow\nprimary $mhd\\_u\\_bcc$ moments only", facecolor="#dff0d8", edgecolor="#38761d")
    add_arrow(ax, (3.75, 2.5), (6.65, 2.5))
    ax.text(5.2, 2.77, "cancel, delete obsolete artifacts,\nrebuild and re-verify", ha="center", va="center", fontsize=9)
    add_box(ax, (0.9, 0.45), (2.3, 0.75), "SGS-backed catalogs\nremoved", facecolor="#eeeeee", edgecolor="#777777")
    add_box(ax, (4.1, 0.45), (2.3, 0.75), "Unavailable physics\nlisted explicitly", facecolor="#fff2cc", edgecolor="#aa8800")
    add_box(ax, (7.3, 0.45), (2.3, 0.75), "Primary artifact graph\nsecond-pass verified", facecolor="#d9ead3", edgecolor="#38761d")
    return save(fig, output_dir, "primary_only_workflow_change_schematic.png")


def status_metrics(run_dir: Path) -> dict[str, object]:
    distribution_rows = quantile_rows(run_dir, "dBB")
    verification = read_json(run_dir / "verification" / "verification_results.json")
    analysis = read_json(run_dir / "analysis" / "ANALYSIS_COMPLETE.json")
    validation = read_json(run_dir / "validation" / "validation_results.json")
    pilot = read_csv(run_dir / "analysis" / "pilot_sample.csv")
    pilot_meta = read_json(run_dir / "analysis" / "pilot_sample_metadata.json")
    correlations = read_csv(run_dir / "analysis" / "spearman_correlations.csv")
    corr_640 = {
        row["property_2"]: float(row["spearman_r"])
        for row in correlations
        if int(row["L_sub"]) == 640 and row["property_1"] == "dBB"
    }
    return {
        "run_dir": str(run_dir),
        "artifact_graph_sha256": verification["artifact_graph_sha256"],
        "analysis_generated_file_count": analysis["generated_file_count"],
        "validation_comparison_count": validation["comparison_count"],
        "validation_failed_comparison_count": validation["failed_comparison_count"],
        "validation_precision_limited_unavailable_count": validation["precision_limited_unavailable_count"],
        "catalogs": verification["catalogs"],
        "dBB_quantiles": {
            row["L_sub"]: {
                "count": int(row["count"]),
                "q16": float(row["q0.16"]),
                "median": float(row["q0.5"]),
                "q84": float(row["q0.84"]),
                "q95": float(row["q0.95"]),
                "q99": float(row["q0.99"]),
            }
            for row in distribution_rows
        },
        "L640_spearman": {
            "dBB_vs_B_mean": corr_640["B_mean"],
            "dBB_vs_deltaB": corr_640["deltaB"],
        },
        "pilot_row_count": len(pilot),
        "pilot_primary_L_sub": pilot_meta["primary_pilot_L_sub"],
        "pilot_matched_pair_count": pilot_meta["matched_pair_count"],
    }


def input_hashes(run_dir: Path) -> dict[str, str]:
    paths = [
        run_dir / "validation" / "validation_results.json",
        run_dir / "verification" / "verification_results.json",
        run_dir / "analysis" / "distribution_quantiles.csv",
        run_dir / "analysis" / "spearman_correlations.csv",
        run_dir / "analysis" / "conditional_dBB_bands.csv",
        run_dir / "analysis" / "cross_scale_dBB.csv",
        run_dir / "analysis" / "pilot_sample.csv",
        run_dir / "analysis" / "pilot_sample_metadata.json",
    ]
    paths.extend(run_dir / "catalogs" / f"catalog_L{l_sub}.npz" for l_sub in L_SUB_VALUES)
    return {str(path.relative_to(run_dir)): sha256(path) for path in paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated = [
        make_workflow_overview(args.output_dir),
        make_primary_only_change_schematic(args.output_dir),
        make_domain_tiling_schematic(args.output_dir),
        make_dbb_slice(args.run_dir, args.output_dir),
        make_dbb_quantile_trend(args.run_dir, args.output_dir),
        make_dbb_distributions(args.run_dir, args.output_dir),
        make_magnetic_correlation(args.run_dir, args.output_dir),
        make_environment_quantile_trends(args.run_dir, args.output_dir),
        make_nonmagnetic_dbb_correlations(args.run_dir, args.output_dir),
        make_correlation_matrix(args.run_dir, args.output_dir),
        make_conditional_bands(args.run_dir, args.output_dir),
        make_cross_scale_persistence(args.run_dir, args.output_dir),
        make_validation_residuals(args.run_dir, args.output_dir),
        make_flagged_fraction(args.run_dir, args.output_dir),
        make_pilot_selection(args.run_dir, args.output_dir),
        make_extraction_estimates(args.run_dir, args.output_dir),
    ]
    manifest = {
        "run_dir": str(args.run_dir),
        "generator_sha256": sha256(Path(__file__)),
        "input_sha256": input_hashes(args.run_dir),
        "figures": {path.name: sha256(path) for path in generated},
        "metrics": status_metrics(args.run_dir),
    }
    manifest_path = args.output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(generated)} figures and {manifest_path}")


if __name__ == "__main__":
    main()
