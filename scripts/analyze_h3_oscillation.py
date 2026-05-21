#!/usr/bin/env python3
# ============================================================================
# scripts/analyze_h3_oscillation.py
# ============================================================================
"""
H3 Hypothesis Analysis: Regression Penalty Impact on Oscillatory Behavior
=========================================================================

HYPOTHESIS H3:
  Imposing a strict regression penalty factor dynamically compresses 
  oscillatory correction behavior and accelerates path convergence.

WHAT THIS DOES:
  1. Loads existing ablation run results
  2. For each ablation configuration (w_err, w_tfp, w_rp_pen):
     - Extracts correction trajectory (error sequences per iteration)
     - Computes oscillation metrics (toggles per step, convergence, etc.)
  3. Analyzes relationship between w_rp_pen and oscillation
  4. Tests whether higher penalty reduces oscillation (H3 prediction)
  5. Generates visualizations and statistical report

USAGE:
    # Analyze existing ablation results
    python scripts/analyze_h3_oscillation.py \\
        --ablation-dir ablation_results \\
        --batch-results batch_results \\
        --output-dir h3_results

    # Generate fresh analysis with specific ablation runs
    python scripts/analyze_h3_oscillation.py \\
        --ablation-dir ablation_results \\
        --batch-results batch_results \\
        --output-dir h3_results \\
        --from-logs

EXPECTED OUTPUT:
    h3_analysis.json         — Structured H3 test results
    h3_oscillation_data.csv  — Per-trajectory oscillation metrics
    h3_report.txt            — Human-readable summary
    h3_penalty_effect.png    — Oscillation vs penalty weight plot
    h3_convergence_effect.png — Convergence iteration vs penalty
    h3_stability_heatmap.png  — Stability score heatmap

If H3 is TRUE:
    • Higher w_rp_pen → Lower oscillation_index
    • Higher w_rp_pen → Faster convergence (fewer iterations)
    • Spearman correlation (penalty, oscillation) should be negative & significant
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics.oscillation_analyzer import OscillationAnalyzer, H3AnalysisResult
from src.metrics.error_tracker import ErrorTracker

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("h3_analysis.log"),
        logging.StreamHandler(),
    ],
)

# ============================================================================
# Data Loading
# ============================================================================

def load_execution_logs(batch_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load execution logs from batch_results directory.
    
    Structure:
        batch_results/
            algorithm_000/
                execution_log_*.json  (one per run config)
            algorithm_001/
                ...
    
    Returns mapping: task_id → list of execution logs (one per config)
    """
    exec_logs = {}
    
    for algo_dir in sorted(batch_dir.glob("algorithm_*")):
        if not algo_dir.is_dir():
            continue
        
        task_id = algo_dir.name
        task_logs = []
        
        for log_file in sorted(algo_dir.glob("execution_log_*.json")):
            try:
                with open(log_file) as f:
                    log = json.load(f)
                    task_logs.append(log)
            except Exception as e:
                logger.warning(f"Failed to load {log_file}: {e}")
        
        if task_logs:
            exec_logs[task_id] = task_logs
    
    return exec_logs


def extract_error_trajectories(
    execution_log: Dict[str, Any],
) -> Optional[List[set]]:
    """
    Extract error set sequence from an execution log.
    
    Execution log structure:
        {
            "iterations": [
                {
                    "iteration": 1,
                    "nodes": {
                        "evaluator": {
                            "hallucination_steps": [...],
                            "missing_step_indices": [...],
                            "operator_error_steps": [...],
                            "assumption_violation_steps": [...]
                        }
                    }
                },
                ...
            ]
        }
    
    Returns
    -------
    List[set] where list[i] = union of all error steps at iteration i
    """
    trajectories = []
    
    if "iterations" not in execution_log:
        return None
    
    for iter_data in execution_log["iterations"]:
        evaluator = iter_data.get("nodes", {}).get("evaluator", {})
        
        # Collect all error steps
        error_steps = set()
        error_steps.update(evaluator.get("hallucination_steps", []))
        error_steps.update(evaluator.get("missing_step_indices", []))
        error_steps.update(evaluator.get("operator_error_steps", []))
        error_steps.update(evaluator.get("assumption_violation_steps", []))
        
        trajectories.append(error_steps)
    
    return trajectories if trajectories else None


def extract_weight_config(execution_log: Dict[str, Any]) -> Dict[str, float]:
    """Extract weight configuration from execution log."""
    config = execution_log.get("config", {})
    return {
        "w_err": config.get("w_err", 0.5),
        "w_tfp": config.get("w_tfp", 0.5),
        "w_rp_pen": config.get("w_rp_pen", 0.4),
    }


# ============================================================================
# Analysis
# ============================================================================

def analyze_h3(
    batch_results_dir: str,
    output_dir: str,
) -> None:
    """
    Main H3 analysis pipeline.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load execution logs
    logger.info(f"Loading execution logs from {batch_results_dir}...")
    batch_path = Path(batch_results_dir)
    exec_logs = load_execution_logs(batch_path)
    logger.info(f"Loaded {sum(len(logs) for logs in exec_logs.values())} execution logs")
    
    # Initialize analysis
    h3_analysis = H3AnalysisResult()
    all_metrics = []
    
    # Analyze each task × configuration
    analyzer = OscillationAnalyzer()
    config_count = 0
    
    for task_id, task_logs in sorted(exec_logs.items()):
        for log_idx, execution_log in enumerate(task_logs):
            config_count += 1
            
            # Extract trajectory
            trajectories = extract_error_trajectories(execution_log)
            if not trajectories:
                logger.warning(f"{task_id} config {log_idx}: no trajectory data")
                continue
            
            # Extract weights
            weights = extract_weight_config(execution_log)
            w_rp_pen = weights["w_rp_pen"]
            
            # Analyze oscillation
            metrics = analyzer.analyze_trajectory(
                error_sequence=trajectories,
                w_rp_pen=w_rp_pen,
                task_id=task_id,
            )
            
            h3_analysis.add_metrics(w_rp_pen, metrics)
            all_metrics.append({
                "task_id": task_id,
                "config_idx": log_idx,
                "w_err": weights["w_err"],
                "w_tfp": weights["w_tfp"],
                "w_rp_pen": w_rp_pen,
                "num_iterations": metrics.num_iterations,
                "total_steps": metrics.total_steps,
                "convergence_iteration": metrics.convergence_iteration,
                "converged": metrics.converged,
                "steps_with_oscillation": metrics.steps_with_oscillation,
                "total_toggles": metrics.total_toggles,
                "mean_toggles_per_step": metrics.mean_toggles_per_step,
                "max_toggles_single_step": metrics.max_toggles_single_step,
                "oscillation_index": metrics.oscillation_index,
                "stability_score": metrics.stability_score,
            })
    
    logger.info(f"Analyzed {config_count} configurations")
    
    # Test H3 hypothesis
    logger.info("Testing H3 hypothesis...")
    h3_result = h3_analysis.test_h3_hypothesis()
    
    # Save detailed metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_csv = output_path / "h3_oscillation_data.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    logger.info(f"Saved metrics to {metrics_csv}")
    
    # Save H3 test results
    h3_json = output_path / "h3_analysis.json"
    with open(h3_json, "w") as f:
        json.dump(h3_result, f, indent=2)
    logger.info(f"Saved H3 results to {h3_json}")
    
    # Generate visualizations
    _generate_visualizations(metrics_df, output_path)
    
    # Generate report
    _generate_report(h3_result, metrics_df, output_path)


def _generate_visualizations(df: pd.DataFrame, output_path: Path) -> None:
    """Generate H3 visualization plots."""
    
    # Plot 1: Oscillation vs Penalty Weight
    fig, ax = plt.subplots(figsize=(10, 6))
    
    penalty_groups = df.groupby("w_rp_pen")
    penalties = sorted(penalty_groups.groups.keys())
    oscillations = [df[df["w_rp_pen"] == p]["oscillation_index"].mean() for p in penalties]
    oscillations_std = [df[df["w_rp_pen"] == p]["oscillation_index"].std() for p in penalties]
    
    ax.errorbar(penalties, oscillations, yerr=oscillations_std, marker="o", capsize=5, linewidth=2)
    ax.set_xlabel("Regression Penalty Weight (w_rp_pen)", fontsize=12)
    ax.set_ylabel("Mean Oscillation Index", fontsize=12)
    ax.set_title("H3: Effect of Penalty on Oscillatory Behavior", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    
    plot_file = output_path / "h3_penalty_effect.png"
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    logger.info(f"Saved plot: {plot_file}")
    plt.close(fig)
    
    # Plot 2: Convergence Iteration vs Penalty
    fig, ax = plt.subplots(figsize=(10, 6))
    
    convergences = [df[df["w_rp_pen"] == p]["convergence_iteration"].mean() for p in penalties]
    convergences_std = [df[df["w_rp_pen"] == p]["convergence_iteration"].std() for p in penalties]
    
    ax.errorbar(penalties, convergences, yerr=convergences_std, marker="s", capsize=5, linewidth=2, color="orange")
    ax.set_xlabel("Regression Penalty Weight (w_rp_pen)", fontsize=12)
    ax.set_ylabel("Mean Convergence Iteration", fontsize=12)
    ax.set_title("H3: Effect of Penalty on Convergence Speed", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    
    plot_file = output_path / "h3_convergence_effect.png"
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    logger.info(f"Saved plot: {plot_file}")
    plt.close(fig)
    
    # Plot 3: Stability Score Heatmap (by config)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create pivot table
    pivot_data = df.pivot_table(
        values="stability_score",
        index="w_err",
        columns="w_rp_pen",
        aggfunc="mean"
    )
    
    im = ax.imshow(pivot_data.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot_data.columns)))
    ax.set_yticks(range(len(pivot_data.index)))
    ax.set_xticklabels([f"{x:.1f}" for x in pivot_data.columns])
    ax.set_yticklabels([f"{y:.1f}" for y in pivot_data.index])
    ax.set_xlabel("Regression Penalty (w_rp_pen)", fontsize=12)
    ax.set_ylabel("Error Resolution Weight (w_err)", fontsize=12)
    ax.set_title("Stability Score Heatmap (Higher = Better)", fontsize=14, fontweight="bold")
    
    plt.colorbar(im, ax=ax, label="Stability Score")
    
    plot_file = output_path / "h3_stability_heatmap.png"
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    logger.info(f"Saved plot: {plot_file}")
    plt.close(fig)


def _generate_report(
    h3_result: Dict[str, Any],
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate human-readable H3 report."""
    
    lines = []
    lines.append("=" * 80)
    lines.append("H3 HYPOTHESIS TEST REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    lines.append("HYPOTHESIS:")
    lines.append(
        "  Imposing a strict regression penalty factor dynamically compresses"
    )
    lines.append(
        "  oscillatory correction behavior and accelerates path convergence."
    )
    lines.append("")
    
    lines.append("METHODOLOGY:")
    lines.append(f"  • Analyzed {len(df)} trajectories across weight configurations")
    lines.append(f"  • Computed oscillation metrics per trajectory:")
    lines.append(f"    - Oscillation Index: [0,1] measure of step toggles")
    lines.append(f"    - Stability Score: [0,1] inverse of oscillation")
    lines.append(f"    - Convergence Iteration: # of iterations to stable state")
    lines.append(f"  • Tested correlation: penalty weight vs oscillation behavior")
    lines.append("")
    
    lines.append("RESULTS:")
    lines.append("-" * 80)
    
    if h3_result.get("h3_verified"):
        lines.append("✅ H3 VERIFIED")
    else:
        lines.append("❌ H3 NOT VERIFIED")
    lines.append("")
    
    # Penalty effect on oscillation
    penalty_effect = h3_result.get("penalty_effect_on_oscillation", {})
    slope = penalty_effect.get("slope")
    lines.append("Penalty Effect on Oscillation:")
    lines.append(f"  Slope: {slope:.6f}")
    lines.append(f"  Interpretation: {penalty_effect.get('interpretation')}")
    lines.append("")
    
    # Penalty effect on convergence
    conv_effect = h3_result.get("penalty_effect_on_convergence", {})
    conv_slope = conv_effect.get("slope")
    lines.append("Penalty Effect on Convergence:")
    lines.append(f"  Slope: {conv_slope:.6f}")
    lines.append(f"  Interpretation: {conv_effect.get('interpretation')}")
    lines.append("")
    
    # Statistical significance
    spearman = h3_result.get("spearman_correlation", {})
    lines.append("Statistical Significance:")
    lines.append(f"  Spearman r: {spearman.get('r')}")
    lines.append(f"  p-value: {spearman.get('p_value')}")
    lines.append(f"  Significant (p<0.05): {spearman.get('significant')}")
    lines.append("")
    
    # Summary statistics by penalty
    summary = h3_result.get("summary_statistics", {})
    lines.append("Summary Statistics by Penalty Weight:")
    lines.append("-" * 80)
    for w_rp_pen in sorted(summary.keys()):
        stats = summary[w_rp_pen]
        lines.append(f"\nw_rp_pen = {w_rp_pen}:")
        lines.append(f"  Mean Oscillation Index: {stats['mean_oscillation_index']:.4f} ± {stats['std_oscillation_index']:.4f}")
        lines.append(f"  Mean Convergence Iter:  {stats['mean_convergence_iteration']:.2f} ± {stats['std_convergence_iteration']:.2f}")
        lines.append(f"  Mean Stability Score:   {stats['mean_stability_score']:.4f} ± {stats['std_stability_score']:.4f}")
        lines.append(f"  Trajectories:           {stats['num_trajectories']}")
    
    lines.append("")
    lines.append("=" * 80)
    lines.append("INTERPRETATION:")
    lines.append("=" * 80)
    
    if h3_result.get("h3_verified"):
        lines.append(
            "✅ FINDING: Higher regression penalty REDUCES oscillation and accelerates "
            "convergence, supporting H3."
        )
        lines.append("")
        lines.append("IMPLICATION: Aggressive regression penalties (w_rp_pen ~ 0.6-0.8)")
        lines.append("stabilize the correction loop and prevent step-level oscillations.")
    else:
        lines.append(
            "❌ FINDING: No clear relationship between penalty and oscillation found. "
            "H3 is NOT supported by the data."
        )
        lines.append("")
        lines.append("POSSIBLE EXPLANATIONS:")
        lines.append("  • Oscillation may be inherent to the proof domain")
        lines.append("  • Penalty weight may not be the primary driver of oscillation")
        lines.append("  • Sample size too small for statistical significance")
    
    lines.append("")
    lines.append("=" * 80)
    
    report_text = "\n".join(lines)
    
    # Save report
    report_file = output_path / "h3_report.txt"
    with open(report_file, "w") as f:
        f.write(report_text)
    
    logger.info(f"Saved report: {report_file}")
    print(report_text)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="H3 Hypothesis Analysis: Oscillation Effects")
    parser.add_argument("--batch-results", required=True, help="Batch results directory")
    parser.add_argument("--output-dir", default="h3_results", help="Output directory")
    
    args = parser.parse_args()
    
    analyze_h3(
        batch_results_dir=args.batch_results,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
