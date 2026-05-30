#!/usr/bin/env python3
"""
Compare structure-function outputs across simulations using saved npz files.

Overlays:
- Isotropic S2/S3 for selected channels.
- Anisotropic S2 (iso/L/perp and L/xi/lambda) mirroring plot_structure_functions layouts.
- Alignment overlays are omitted unless a future saved schema provides first
  moments.  The current anisotropic sidecar stores second moments only.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import cmasher as cmr  # type: ignore

CHANNEL_LABELS = {
    "D_V": r"$\delta v$",
    "D_B": r"$\delta B$",
    "D_RHO": r"$\delta \rho$",
    "D_VA": r"$\delta v_A$",
    "D_ZPLUS": r"$\delta z^+$",
    "D_ZMINUS": r"$\delta z^-$",
    "D_OMEGA": r"$\delta \omega$",
    "D_J": r"$\delta j$",
    "D_CURV": r"$\delta K$",
    "D_GRAD_RHO": r"$\delta |\nabla \rho|$",
    "D_B_over_Bmean_loc": r"$\delta B_\ell / \overline{B}_\ell$",
    # Alignment ratios
    "D_Vperp_D_Bperp_CROSS_MAG_RATIO": r"$\sin \theta_{vB}$",
    "D_Vperp_D_Omegaperp_CROSS_MAG_RATIO": r"$\sin \theta_{v\omega}$",
    "D_Bperp_D_Jperp_CROSS_MAG_RATIO": r"$\sin \theta_{Bj}$",
    "D_Omegaperp_D_Jperp_CROSS_MAG_RATIO": r"$\sin \theta_{\omega j}$",
}

DIRECTIONS = ("iso", "L", "perp", "xi", "lambda")
ALIGNMENT_SPECS = [
    {
        "name": "vB",
        "ratio": "D_Vperp_D_Bperp_CROSS_MAG_RATIO",
        "cross": "D_Vperp_CROSS_D_Bperp",
        "mag": "D_Vperp_D_Bperp_MAG",
        "label": r"$\sin \theta_{vB} = \langle|\delta v_\perp \times \delta B_\perp|\rangle / \langle|\delta v_\perp||\delta B_\perp|\rangle$",
        "label_prime": r"$\sin \theta'_{vB} = \langle|\delta v_\perp \times \delta B_\perp| / (|\delta v_\perp||\delta B_\perp|)\rangle$",
    },
    {
        "name": "vOmega",
        "ratio": "D_Vperp_D_Omegaperp_CROSS_MAG_RATIO",
        "cross": "D_Vperp_CROSS_D_Omegaperp",
        "mag": "D_Vperp_D_Omegaperp_MAG",
        "label": r"$\sin \theta_{v\omega} = \langle|\delta v_\perp \times \delta \omega_\perp|\rangle / \langle|\delta v_\perp||\delta \omega_\perp|\rangle$",
        "label_prime": r"$\sin \theta'_{v\omega} = \langle|\delta v_\perp \times \delta \omega_\perp| / (|\delta v_\perp||\delta \omega_\perp|)\rangle$",
    },
    {
        "name": "Bj",
        "ratio": "D_Bperp_D_Jperp_CROSS_MAG_RATIO",
        "cross": "D_Bperp_CROSS_D_Jperp",
        "mag": "D_Bperp_D_Jperp_MAG",
        "label": r"$\sin \theta_{Bj} = \langle|\delta B_\perp \times \delta j_\perp|\rangle / \langle|\delta B_\perp||\delta j_\perp|\rangle$",
        "label_prime": r"$\sin \theta'_{Bj} = \langle|\delta B_\perp \times \delta j_\perp| / (|\delta B_\perp||\delta j_\perp|)\rangle$",
    },
    {
        "name": "omegaJ",
        "ratio": "D_Omegaperp_D_Jperp_CROSS_MAG_RATIO",
        "cross": "D_Omegaperp_CROSS_D_Jperp",
        "mag": "D_Omegaperp_D_Jperp_MAG",
        "label": r"$\sin \theta_{\omega j} = \langle|\delta \omega_\perp \times \delta j_\perp|\rangle / \langle|\delta \omega_\perp||\delta j_\perp|\rangle$",
        "label_prime": r"$\sin \theta'_{\omega j} = \langle|\delta \omega_\perp \times \delta j_\perp| / (|\delta \omega_\perp||\delta j_\perp|)\rangle$",
    },
]
_WARNED_INVALID_ALIGNMENT_SCHEMA = False


def apply_house_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "figure.dpi": 150,
        }
    )


@dataclass
class RunData:
    resolution: int
    beta: float
    sw: int
    s2_path: Optional[Path] = None
    s3_path: Optional[Path] = None
    aniso_path: Optional[Path] = None
    _s2_cache: Optional[dict] = field(default=None, init=False, repr=False)
    _s3_cache: Optional[dict] = field(default=None, init=False, repr=False)
    _aniso_cache: Optional[dict] = field(default=None, init=False, repr=False)

    def label(self) -> str:
        beta_str = f"{self.beta:g}"
        return f"res{self.resolution}_beta{beta_str}_sw{self.sw}"

    def _load_sf(self, order: int) -> Optional[dict]:
        if order == 2 and self._s2_cache is not None:
            return self._s2_cache
        if order == 3 and self._s3_cache is not None:
            return self._s3_cache
        path = self.s2_path if order == 2 else self.s3_path
        if path is None:
            return None
        data = np.load(path, allow_pickle=True)
        cache = {
            "ell": data["ell_centers"],
            "channels": list(data["channels"]),
            "values": data["S2"] if order == 2 else data["S3"],
        }
        if order == 2:
            self._s2_cache = cache
        else:
            self._s3_cache = cache
        return cache

    def get_sf(self, order: int, channel: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        cache = self._load_sf(order)
        if cache is None:
            return None, None
        if channel not in cache["channels"]:
            return None, None
        idx = cache["channels"].index(channel)
        return cache["ell"], cache["values"][idx]

    def _load_aniso(self) -> Optional[dict]:
        if self._aniso_cache is not None:
            return self._aniso_cache
        if self.aniso_path is None:
            return None
        data = np.load(self.aniso_path, allow_pickle=True)
        result: Dict[str, Dict[str, np.ndarray]] = {}
        for key in data.files:
            if key in {"ell_centers", "ell_bin_edges", "theta_bin_edges", "phi_bin_edges", "theta_wedge_bins", "phi_wedge_bins"}:
                continue
            for suffix in DIRECTIONS:
                suf = f"_{suffix}"
                if key.endswith(suf):
                    chan = key[: -len(suf)]
                    result.setdefault(chan, {})[suffix] = data[key]
                    break
        result["ell_centers"] = data["ell_centers"]
        self._aniso_cache = result
        return result

    def get_aniso(self, channel: str) -> Tuple[Optional[np.ndarray], Optional[Dict[str, np.ndarray]]]:
        cache = self._load_aniso()
        if cache is None or channel not in cache:
            return None, None
        return cache["ell_centers"], cache[channel]


def parse_run_from_path(path: Path) -> Optional[Tuple[int, float, int]]:
    """
    Extract (resolution, beta, sw) from a path like:
    results_Turb_5120_beta25_dedt025_plm/.../sw3_.../
    """
    m = re.search(r"results_Turb_(\d+)_beta([0-9.]+)_", str(path))
    m_sw = re.search(r"sw(\d+)", str(path))
    if not m or not m_sw:
        return None
    res = int(m.group(1))
    beta = float(m.group(2))
    sw = int(m_sw.group(1))
    return res, beta, sw


def discover_runs(results_root: Path) -> Dict[Tuple[int, float, int], RunData]:
    runs: Dict[Tuple[int, float, int], RunData] = {}
    for npz in results_root.rglob("*_structure_functions_2nd.npz"):
        meta = parse_run_from_path(npz)
        if meta is None:
            continue
        key = meta
        run = runs.get(key, RunData(*meta))
        run.s2_path = npz
        runs[key] = run
    for npz in results_root.rglob("*_structure_functions_3rd.npz"):
        meta = parse_run_from_path(npz)
        if meta is None:
            continue
        key = meta
        run = runs.get(key, RunData(*meta))
        run.s3_path = npz
        runs[key] = run
    for npz in results_root.rglob("*_anisotropic_S2.npz"):
        meta = parse_run_from_path(npz)
        if meta is None:
            continue
        key = meta
        run = runs.get(key, RunData(*meta))
        run.aniso_path = npz
        runs[key] = run
    return runs


def group_sw_sweep(runs: Dict[Tuple[int, float, int], RunData]) -> List[Tuple[str, List[RunData]]]:
    grouped: Dict[Tuple[int, float], List[RunData]] = {}
    for (res, beta, _), run in runs.items():
        if res not in (5120, 10240):
            continue  # only high-res sweeps
        grouped.setdefault((res, beta), []).append(run)
    result = []
    for (res, beta), items in grouped.items():
        if len(items) < 2:
            continue
        items_sorted = sorted(items, key=lambda r: r.sw)
        desc = f"sw sweep (res={res}, beta={beta:g})"
        result.append((desc, items_sorted))
    return result


def group_resolution_beta25(runs: Dict[Tuple[int, float, int], RunData]) -> List[Tuple[str, List[RunData]]]:
    result = []
    grouped: Dict[int, List[RunData]] = {}
    for (res, beta, sw), run in runs.items():
        if not np.isclose(beta, 25.0):
            continue
        grouped.setdefault(sw, []).append(run)
    for sw, items in grouped.items():
        if len(items) < 2:
            continue
        items_sorted = sorted(items, key=lambda r: r.resolution)
        desc = f"resolution sweep beta=25 (sw={sw})"
        result.append((desc, items_sorted))
    return result


def group_beta_res5120(runs: Dict[Tuple[int, float, int], RunData]) -> List[Tuple[str, List[RunData]]]:
    result = []
    grouped: Dict[int, List[RunData]] = {}
    for (res, beta, sw), run in runs.items():
        if res != 5120:
            continue
        grouped.setdefault(sw, []).append(run)
    for sw, items in grouped.items():
        if len(items) < 2:
            continue
        items_sorted = sorted(items, key=lambda r: r.beta)
        desc = f"beta sweep res=5120 (sw={sw})"
        result.append((desc, items_sorted))
    return result


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def plot_isotropic_overlays(
    runs_list: List[RunData],
    channel: str,
    order: int,
    title: str,
    out_path: Path,
    colors: np.ndarray,
    xlabel: str = r"$\ell$",
    scale_x=None,
):
    plt.figure(figsize=(6.5, 5.0))
    plotted = False
    for color, run in zip(colors, runs_list):
        ell, vals = run.get_sf(order, channel)
        if ell is None or vals is None:
            continue
        ell_plot = scale_x(ell, run) if scale_x else ell
        mask = np.isfinite(vals) & (vals > 0) & np.isfinite(ell_plot) & (ell_plot > 0)
        if not np.any(mask):
            continue
        label = run.label()
        fit = best_powerlaw_fit(ell_plot[mask], vals[mask])
        if fit is not None:
            slope, intercept, start, end = fit
            x_fit = np.logspace(np.log10(start), np.log10(end), 200)
            y_fit = 10 ** (intercept + slope * np.log10(x_fit))
            plt.loglog(x_fit, y_fit, color=color, linestyle="--", lw=3.0, alpha=0.5)
            label += rf" ($\propto \ell^{{{slope:.2f}}}$)"
        plt.loglog(ell_plot[mask], vals[mask], color=color, lw=1.8, label=label)
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.xlabel(xlabel)
    plt.ylabel(rf"$S_{{{order}}}$({CHANNEL_LABELS.get(channel, channel)})")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    ensure_dir(out_path.parent)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Created: {out_path}")


def _add_dir_legend(ax, colors, runs_list):
    handles = [plt.Line2D([0], [0], color=c, lw=2) for c in colors]
    labels = [r.label() for r in runs_list]
    return ax.legend(handles, labels, fontsize=8, loc="best", title="runs")


def _add_style_legend(ax, linestyles):
    handles = [
        plt.Line2D([0], [0], color="k", lw=1.8, linestyle=ls, label=lbl)
        for ls, lbl in linestyles
    ]
    return ax.legend(handles=handles, fontsize=8, loc="lower left", title="direction")


def plot_anisotropic_overlays(
    runs_list: List[RunData],
    channel: str,
    title: str,
    out_path: Path,
    colors: np.ndarray,
    xlabel: str = r"$\ell$",
    scale_x=None,
):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6), sharex="col")
    ax_ul = axes[0, 0]  # ell_par vs S2 (solid), ell_perp (dashed)
    ax_ll = axes[1, 0]  # ell_par vs S2 (solid), xi (dashed), lambda (dotted)
    ax_ur = axes[0, 1]  # matching ell_perp vs ell_par
    ax_lr = axes[1, 1]  # xi/lambda vs ell_par

    def find_matching_scales(target_x: np.ndarray, target_y: np.ndarray, ref_x: np.ndarray, ref_y: np.ndarray) -> np.ndarray:
        """For each target_y, find ref_x where ref_y matches target_y using log-space interpolation."""
        matches = np.full_like(target_x, np.nan, dtype=float)
        valid_target = np.isfinite(target_x) & np.isfinite(target_y) & (target_x > 0) & (target_y > 0)
        valid_ref = np.isfinite(ref_x) & np.isfinite(ref_y) & (ref_x > 0) & (ref_y > 0)
        if np.count_nonzero(valid_ref) < 2:
            return matches
        rx = ref_x[valid_ref]
        ry = ref_y[valid_ref]
        for idx, (tx, ty) in enumerate(zip(target_x, target_y)):
            if not valid_target[idx]:
                continue
            log_ty = np.log(ty)
            for j in range(len(rx) - 1):
                y0, y1 = ry[j], ry[j + 1]
                if y0 <= 0 or y1 <= 0:
                    continue
                log_y0 = np.log(y0)
                log_y1 = np.log(y1)
                if (log_ty - log_y0) * (log_ty - log_y1) <= 0:
                    log_x0 = np.log(rx[j])
                    log_x1 = np.log(rx[j + 1])
                    t = (log_ty - log_y0) / (log_y1 - log_y0) if log_y1 != log_y0 else 0.0
                    log_match = log_x0 + t * (log_x1 - log_x0)
                    matches[idx] = np.exp(log_match)
                    break
        return matches

    plotted_any = False
    for color, run in zip(colors, runs_list):
        ell, stats = run.get_aniso(channel)
        if ell is None or stats is None:
            continue
        ell_plot = scale_x(ell, run) if scale_x else ell
        mask_L = np.isfinite(stats.get("L", [])) & (stats.get("L", []) > 0)
        mask_perp = np.isfinite(stats.get("perp", [])) & (stats.get("perp", []) > 0)
        mask_xi = np.isfinite(stats.get("xi", [])) & (stats.get("xi", []) > 0)
        mask_lambda = np.isfinite(stats.get("lambda", [])) & (stats.get("lambda", []) > 0)

        # Upper left: ell_par (solid) and ell_perp (dashed)
        if np.any(mask_L):
            fit_L = best_powerlaw_fit(ell_plot[mask_L], stats["L"][mask_L])
            if fit_L is not None:
                slope, intercept, start, end = fit_L
                x_fit = np.logspace(np.log10(start), np.log10(end), 200)
                y_fit = 10 ** (intercept + slope * np.log10(x_fit))
                ax_ul.loglog(x_fit, y_fit, color=color, linestyle="--", lw=3.0, alpha=0.5)
            ax_ul.loglog(ell_plot[mask_L], stats["L"][mask_L], color=color, linestyle="-", lw=1.6)
        if np.any(mask_perp):
            fit_perp = best_powerlaw_fit(ell_plot[mask_perp], stats["perp"][mask_perp])
            if fit_perp is not None:
                slope, intercept, start, end = fit_perp
                x_fit = np.logspace(np.log10(start), np.log10(end), 200)
                y_fit = 10 ** (intercept + slope * np.log10(x_fit))
                ax_ul.loglog(x_fit, y_fit, color=color, linestyle="--", lw=3.0, alpha=0.5)
            ax_ul.loglog(ell_plot[mask_perp], stats["perp"][mask_perp], color=color, linestyle="--", lw=1.3)

        # Lower left: ell_par (solid), xi (dashed), lambda (dotted)
        if np.any(mask_L):
            fit_L_ll = best_powerlaw_fit(ell_plot[mask_L], stats["L"][mask_L])
            if fit_L_ll is not None:
                slope, intercept, start, end = fit_L_ll
                x_fit = np.logspace(np.log10(start), np.log10(end), 200)
                y_fit = 10 ** (intercept + slope * np.log10(x_fit))
                ax_ll.loglog(x_fit, y_fit, color=color, linestyle="--", lw=3.0, alpha=0.5)
            ax_ll.loglog(ell_plot[mask_L], stats["L"][mask_L], color=color, linestyle="-", lw=1.6)
        if np.any(mask_xi):
            fit_xi = best_powerlaw_fit(ell_plot[mask_xi], stats["xi"][mask_xi])
            if fit_xi is not None:
                slope, intercept, start, end = fit_xi
                x_fit = np.logspace(np.log10(start), np.log10(end), 200)
                y_fit = 10 ** (intercept + slope * np.log10(x_fit))
                ax_ll.loglog(x_fit, y_fit, color=color, linestyle="--", lw=3.0, alpha=0.5)
            ax_ll.loglog(ell_plot[mask_xi], stats["xi"][mask_xi], color=color, linestyle="--", lw=1.3)
        if np.any(mask_lambda):
            fit_lambda = best_powerlaw_fit(ell_plot[mask_lambda], stats["lambda"][mask_lambda])
            if fit_lambda is not None:
                slope, intercept, start, end = fit_lambda
                x_fit = np.logspace(np.log10(start), np.log10(end), 200)
                y_fit = 10 ** (intercept + slope * np.log10(x_fit))
                ax_ll.loglog(x_fit, y_fit, color=color, linestyle="--", lw=3.0, alpha=0.5)
            ax_ll.loglog(ell_plot[mask_lambda], stats["lambda"][mask_lambda], color=color, linestyle=":", lw=1.3)

        # Upper right: matching ell_perp vs ell_par
        match_perp = find_matching_scales(ell_plot, stats.get("L", np.full_like(ell_plot, np.nan)), ell_plot, stats.get("perp", np.full_like(ell_plot, np.nan)))
        match_mask = np.isfinite(match_perp) & np.isfinite(ell_plot) & (match_perp > 0) & (ell_plot > 0)
        if np.any(match_mask):
            ax_ur.loglog(ell_plot[match_mask], match_perp[match_mask], color=color, linestyle="-", lw=1.4)

        # Lower right: matched xi (solid) and lambda (dashed) vs ell_par
        match_xi = find_matching_scales(ell_plot, stats.get("L", np.full_like(ell_plot, np.nan)), ell_plot, stats.get("xi", np.full_like(ell_plot, np.nan)))
        match_lambda = find_matching_scales(ell_plot, stats.get("L", np.full_like(ell_plot, np.nan)), ell_plot, stats.get("lambda", np.full_like(ell_plot, np.nan)))
        mask_match_xi = np.isfinite(match_xi) & np.isfinite(ell_plot) & (match_xi > 0) & (ell_plot > 0)
        mask_match_lambda = np.isfinite(match_lambda) & np.isfinite(ell_plot) & (match_lambda > 0) & (ell_plot > 0)
        if np.any(mask_match_xi):
            ax_lr.loglog(ell_plot[mask_match_xi], match_xi[mask_match_xi], color=color, linestyle="-", lw=1.4)
        if np.any(mask_match_lambda):
            ax_lr.loglog(ell_plot[mask_match_lambda], match_lambda[mask_match_lambda], color=color, linestyle="--", lw=1.4)

        if any([np.any(mask_L), np.any(mask_perp), np.any(mask_xi), np.any(mask_lambda)]):
            plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return

    ax_ul.set_xlabel(xlabel)
    ax_ul.set_ylabel(r"$S_2$" + f"({CHANNEL_LABELS.get(channel, channel)})")
    ax_ul.grid(True, which="both", alpha=0.3)

    ax_ll.set_xlabel(xlabel)
    ax_ll.set_ylabel(r"$S_2$" + f"({CHANNEL_LABELS.get(channel, channel)})")
    ax_ll.grid(True, which="both", alpha=0.3)

    ax_ur.set_xlabel(r"$\ell_\parallel$")
    ax_ur.set_ylabel(r"$\ell_\perp$ (match)")
    ax_ur.grid(True, which="both", alpha=0.3)

    ax_lr.set_xlabel(r"$\ell_\parallel$")
    ax_lr.set_ylabel(r"$\xi,\,\lambda$ (match)")
    ax_lr.grid(True, which="both", alpha=0.3)

    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_yscale("log")

    run_leg = _add_dir_legend(ax_ul, colors, runs_list)
    style_handles_ul = [
        plt.Line2D([0], [0], color="k", linestyle="-", lw=1.6, label=r"$\ell_\parallel$"),
        plt.Line2D([0], [0], color="k", linestyle="--", lw=1.3, label=r"$\ell_\perp$"),
    ]
    style_leg_ul = ax_ul.legend(handles=style_handles_ul, fontsize=8, loc="lower left", title="direction")
    style_handles_ll = [
        plt.Line2D([0], [0], color="k", linestyle="-", lw=1.6, label=r"$\ell_\parallel$"),
        plt.Line2D([0], [0], color="k", linestyle="--", lw=1.3, label=r"$\xi$"),
        plt.Line2D([0], [0], color="k", linestyle=":", lw=1.3, label=r"$\lambda$"),
    ]
    style_leg_ll = ax_ll.legend(handles=style_handles_ll, fontsize=8, loc="lower left", title="direction")
    ax_ul.add_artist(run_leg)
    ax_ul.add_artist(style_leg_ul)
    ax_ll.add_artist(style_leg_ll)

    ensure_dir(out_path.parent)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Created: {out_path}")


def extract_theta_values(run: RunData, spec: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Return no alignment curve from the second-moment-only sidecar schema.

    Reconstructing alignment here would silently report ``<cross^2> /
    <product^2>`` or ``<sin(theta)^2>`` as first-moment statistics.
    """

    global _WARNED_INVALID_ALIGNMENT_SCHEMA
    if not _WARNED_INVALID_ALIGNMENT_SCHEMA:
        print("Skipping cross-run alignment overlays: saved anisotropic sidecars contain S2, not required first moments.")
        _WARNED_INVALID_ALIGNMENT_SCHEMA = True
    return None, None, None


def plot_alignment_variant(
    runs_list: List[RunData],
    spec: dict,
    variant: str,
    title: str,
    out_path: Path,
    colors: np.ndarray,
    xlabel: str = r"$\ell$",
    scale_x=None,
):
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    plotted = False
    for color, run in zip(colors, runs_list):
        ell, theta_mean, theta_prime = extract_theta_values(run, spec)
        if ell is None:
            continue
        theta_vals = theta_prime if variant == "prime" else theta_mean
        if theta_vals is None:
            continue
        ell_plot = scale_x(ell, run) if scale_x else ell
        mask = np.isfinite(theta_vals) & (theta_vals > 0) & np.isfinite(ell_plot) & (ell_plot > 0)
        if not np.any(mask):
            continue
        fit = best_powerlaw_fit(ell_plot[mask], theta_vals[mask])
        label = run.label()
        if fit is not None:
            slope, intercept, start, end = fit
            x_fit = np.logspace(np.log10(start), np.log10(end), 200)
            y_fit = 10 ** (intercept + slope * np.log10(x_fit))
            ax.loglog(x_fit, y_fit, color=color, linestyle="--", alpha=0.7, lw=1.2)
            label += rf" ($\propto \ell^{{{slope:.2f}}}$)"
        ax.loglog(ell_plot[mask], theta_vals[mask], color=color, linestyle="-", lw=1.6, label=label)
        fit = best_powerlaw_fit(ell_plot[mask], theta_vals[mask])
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel(xlabel)
    ylab = spec["label_prime"] if variant == "prime" else spec["label"]
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    ensure_dir(out_path.parent)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Created: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Overlay structure-function comparisons across simulations.")
    parser.add_argument("--results-root", type=Path, default=Path("Results"), help="Root directory containing simulation outputs.")
    parser.add_argument("--output-root", type=Path, default=None, help="Directory for comparison plots (default: <results-root>/comparisons).")
    parser.add_argument(
        "--channels",
        nargs="+",
        default=[
            "D_V",
            "D_B",
            "D_RHO",
            "D_VA",
            "D_ZPLUS",
            "D_ZMINUS",
            "D_OMEGA",
            "D_J",
            "D_CURV",
            "D_GRAD_RHO",
            "D_B_over_Bmean_loc",
        ],
        help="Channels to plot (isotropic + anisotropic).",
    )
    args = parser.parse_args()

    apply_house_style()
    output_root = args.output_root or args.results_root / "comparisons"
    ensure_dir(output_root)

    runs = discover_runs(args.results_root)
    if not runs:
        print("No runs found.")
        return 1
    print(f"Discovered {len(runs)} runs with available npz files.")

    # Build grouping lists
    sw_groups = group_sw_sweep(runs)
    res_groups = group_resolution_beta25(runs)
    beta_groups = group_beta_res5120(runs)

    def process_groups(groups: Iterable[Tuple[str, List[RunData]]], subdir: str, cmap, xlabel=r"$\ell$", scale_x=None):
        for desc, run_list in groups:
            base_dir = output_root / subdir
            ensure_dir(base_dir / "isotropic")
            ensure_dir(base_dir / "anisotropic")
            ensure_dir(base_dir / "alignment")
            colors = cmap(np.linspace(0, 1, len(run_list)))
            for channel in args.channels:
                # Isotropic S2/S3
                plot_isotropic_overlays(
                    run_list,
                    channel,
                    order=2,
                    title=f"S2 {channel} ({desc})",
                    out_path=base_dir / "isotropic" / f"{channel}_S2_{sanitize(desc)}.png",
                    colors=colors,
                    xlabel=xlabel,
                    scale_x=scale_x,
                )
                plot_isotropic_overlays(
                    run_list,
                    channel,
                    order=3,
                    title=f"S3 {channel} ({desc})",
                    out_path=base_dir / "isotropic" / f"{channel}_S3_{sanitize(desc)}.png",
                    colors=colors,
                    xlabel=xlabel,
                    scale_x=scale_x,
                )
                # Anisotropic S2
                plot_anisotropic_overlays(
                    run_list,
                    channel,
                    title=f"Anisotropic S2 {channel} ({desc})",
                    out_path=base_dir / "anisotropic" / f"{channel}_anisotropic_{sanitize(desc)}.png",
                    colors=colors,
                    xlabel=xlabel,
                    scale_x=scale_x,
                )
            # Alignment overlays
            for spec in ALIGNMENT_SPECS:
                plot_alignment_variant(
                    run_list,
                    spec,
                    variant="theta",
                    title=f"{spec['name']} ({desc})",
                    out_path=base_dir / "alignment" / f"theta_{spec['name']}_{sanitize(desc)}.png",
                    colors=colors,
                    xlabel=xlabel,
                    scale_x=scale_x,
                )
                plot_alignment_variant(
                    run_list,
                    spec,
                    variant="prime",
                    title=f"{spec['name']} prime ({desc})",
                    out_path=base_dir / "alignment" / f"theta_prime_{spec['name']}_{sanitize(desc)}.png",
                    colors=colors,
                    xlabel=xlabel,
                    scale_x=scale_x,
                )

    process_groups(sw_groups, "sw_sweep", cmr.cosmic)
    process_groups(
        res_groups,
        "res_sweep_beta25",
        cmr.lavender,
        xlabel=r"$\ell / L_{\mathrm{box}}$",
        scale_x=lambda ell, run: ell / float(run.resolution),
    )
    process_groups(beta_groups, "beta_sweep_res5120", cmr.dusk)
    print("Done.")
    return 0


def sanitize(text: str) -> str:
    # Keep filenames readable and filesystem-friendly.
    text = text.replace("-", "_")
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def best_powerlaw_fit(x_vals: np.ndarray, y_vals: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """
    Fit best 8x window in log-log space. Returns (slope, intercept, start, end) or None.
    """
    mask = np.isfinite(x_vals) & np.isfinite(y_vals) & (x_vals > 0) & (y_vals > 0)
    if not np.any(mask):
        return None
    x = x_vals[mask]
    y = y_vals[mask]
    if x.size < 3:
        return None
    x_max = np.nanmax(x)
    max_end = x_max / 2.0
    start_candidates = np.unique(x[(x * 8.0 <= max_end)])
    if start_candidates.size == 0:
        start_candidates = np.unique(x)
    best = None
    for start in start_candidates:
        end_limit = start * 8.0
        fit_mask = (x >= start) & (x <= end_limit)
        if np.count_nonzero(fit_mask) < 3:
            continue
        logx = np.log10(x[fit_mask])
        logy = np.log10(y[fit_mask])
        slope, intercept = np.polyfit(logx, logy, 1)
        resid = logy - (slope * logx + intercept)
        chi2 = np.sum(resid ** 2)
        end_actual = x[fit_mask].max()
        if best is None or chi2 < best[0] - 1e-12 or (
            np.isclose(chi2, best[0]) and (start < best[3] - 1e-12 or end_actual > best[4])
        ):
            best = (chi2, slope, intercept, start, end_actual)
    if best is None:
        return None
    _, slope, intercept, start, end_actual = best
    return slope, intercept, start, end_actual


if __name__ == "__main__":
    raise SystemExit(main())
