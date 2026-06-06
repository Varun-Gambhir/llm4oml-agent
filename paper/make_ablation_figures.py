#!/usr/bin/env python3
"""
Generate paper-grade RACS ablation figures.

Recommended input:
  batch_results_full_ablation/ablation_results.csv

Why this script exists:
  - Uses RACS terminology instead of legacy CRS labels.
  - Avoids "human score" wording unless the target is truly human-labeled.
  - Produces compact single-column and double-column PDF/PNG figures.
  - Avoids marking the default point if that exact weight setting is not on the grid.

Example:
  python paper/make_ablation_figures.py \
      --input batch_results_full_ablation/ablation_results.csv \
      --output-dir paper/figures \
      --target-label "verdict-derived quality proxy"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_W_ERR = 0.50
DEFAULT_W_RP = 0.40


def _setup_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.grid": False,
    })


def _draw_heatmap(
    ax: plt.Axes,
    pivot: pd.DataFrame,
    *,
    cmap: str = "viridis",
    value_fmt: str = ".3f",
    cbar_label: str | None = None,
) -> None:
    values = pivot.values.astype(float)
    im = ax.imshow(values, cmap=cmap, aspect="auto", origin="lower")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{y:.2f}" for y in pivot.index])

    ax.set_xticks(np.arange(-0.5, len(pivot.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    finite_vals = values[np.isfinite(values)]
    threshold = np.nanmean(finite_vals) if finite_vals.size else 0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if not np.isfinite(val):
                continue
            color = "white" if val > threshold else "black"
            ax.text(j, i, format(val, value_fmt), ha="center", va="center", fontsize=7, color=color)

    if cbar_label:
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)


def _load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"w_err", "w_tfp", "w_rp_pen", "spearman_r"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for col in ["w_err", "w_tfp", "w_rp_pen", "spearman_r"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["w_err", "w_rp_pen", "spearman_r"]).copy()
    df["w_err"] = df["w_err"].round(4)
    df["w_rp_pen"] = df["w_rp_pen"].round(4)
    return df


def _has_default(df: pd.DataFrame) -> bool:
    mask = (
        ((df["w_err"] - DEFAULT_W_ERR).abs() < 1e-9)
        & ((df["w_rp_pen"] - DEFAULT_W_RP).abs() < 1e-9)
    )
    return bool(mask.any())


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)


def plot_spearman_heatmap(df: pd.DataFrame, out_dir: Path, target_label: str) -> None:
    pivot = df.pivot_table(
        index="w_rp_pen",
        columns="w_err",
        values="spearman_r",
        aggfunc="mean",
    ).sort_index(ascending=True)

    n_rows, n_cols = pivot.shape
    fig_w = 3.45 if n_cols <= 5 else 6.9
    fig_h = 2.55 if n_rows <= 5 else 3.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    _draw_heatmap(ax, pivot, cmap="viridis", value_fmt=".3f", cbar_label="Spearman rho")

    ax.set_title("RACS Weight Sensitivity")
    ax.set_xlabel(r"$w_{\mathrm{err}}$  ($w_{\mathrm{tfp}}=1-w_{\mathrm{err}}$)")
    ax.set_ylabel(r"$w_{\mathrm{rp}}$ regression penalty")

    if _has_default(df):
        x_vals = list(pivot.columns)
        y_vals = list(pivot.index)
        x = x_vals.index(round(DEFAULT_W_ERR, 4))
        y = y_vals.index(round(DEFAULT_W_RP, 4))
        ax.scatter([x], [y], marker="*", s=180, color="red", edgecolor="white", linewidth=0.8)
        ax.text(
            x + 0.15,
            y - 0.25,
            "default",
            color="white",
            fontsize=7,
            weight="bold",
            ha="left",
            va="center",
        )
    _save(fig, out_dir, "racs_spearman_heatmap")


def plot_balance_lines(df: pd.DataFrame, out_dir: Path, target_label: str) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 2.55))

    penalties = sorted(df["w_rp_pen"].unique())
    cmap = plt.get_cmap("viridis")
    palette = [cmap(i / max(1, len(penalties) - 1)) for i in range(len(penalties))]

    for color, penalty in zip(palette, penalties):
        sub = df[df["w_rp_pen"] == penalty].sort_values("w_err")
        ax.plot(
            sub["w_err"],
            sub["spearman_r"],
            marker="o",
            markersize=3,
            linewidth=1.2,
            color=color,
            label=fr"$w_{{rp}}={penalty:.2f}$",
        )

    ax.axvline(DEFAULT_W_ERR, color="red", linestyle=":", linewidth=1.2)
    ax.text(
        DEFAULT_W_ERR + 0.01,
        ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
        r"default $w_{\rm err}$",
        color="red",
        fontsize=7,
        rotation=90,
        va="bottom",
    )

    ax.set_title("ERR/TFP Balance Sensitivity")
    ax.set_xlabel(r"$w_{\mathrm{err}}$  ($w_{\mathrm{tfp}}=1-w_{\mathrm{err}}$)")
    ax.set_ylabel("Spearman rho")
    ax.grid(True, alpha=0.28)
    ax.legend(frameon=True, ncol=1, loc="best", title="Penalty")
    ax.text(
        0.99,
        -0.24,
        f"target: {target_label}",
        transform=ax.transAxes,
        fontsize=7,
        ha="right",
        va="top",
        color="dimgray",
    )

    _save(fig, out_dir, "racs_balance_lines")


def plot_metric_panel(df: pd.DataFrame, out_dir: Path, target_label: str) -> None:
    """Optional two-panel figure for papers with space."""
    available = [c for c in ["spearman_r", "mean_crs", "convergence_rate"] if c in df.columns]
    if len(available) < 2:
        return

    metrics = [
        ("spearman_r", "Spearman rho"),
        ("mean_crs", "Mean RACS"),
        ("convergence_rate", "Convergence rate"),
    ]
    metrics = [m for m in metrics if m[0] in df.columns]

    fig, axes = plt.subplots(1, len(metrics), figsize=(3.45 * len(metrics), 2.45))
    if len(metrics) == 1:
        axes = [axes]

    for ax, (col, label) in zip(axes, metrics):
        pivot = df.pivot_table(
            index="w_rp_pen",
            columns="w_err",
            values=col,
            aggfunc="mean",
        ).sort_index(ascending=True)
        _draw_heatmap(
            ax,
            pivot,
            cmap="viridis",
            value_fmt=".3f" if col != "convergence_rate" else ".2f",
            cbar_label=None,
        )
        ax.set_title(label)
        ax.set_xlabel(r"$w_{\mathrm{err}}$")
        ax.set_ylabel(r"$w_{\mathrm{rp}}$" if ax is axes[0] else "")

    fig.suptitle(f"RACS Ablation Summary ({target_label})", y=1.04, fontsize=10)
    _save(fig, out_dir, "racs_ablation_metric_panel")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures"))
    parser.add_argument(
        "--target-label",
        default="verdict-derived quality proxy",
        help="Label printed on plots. Use 'human expert score' only if labels are actually human-rated.",
    )
    args = parser.parse_args()

    _setup_style()
    df = _load_results(args.input)

    plot_spearman_heatmap(df, args.output_dir, args.target_label)
    plot_balance_lines(df, args.output_dir, args.target_label)
    plot_metric_panel(df, args.output_dir, args.target_label)

    print(f"Generated figures in {args.output_dir}")


if __name__ == "__main__":
    main()
