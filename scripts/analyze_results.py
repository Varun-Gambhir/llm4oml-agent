# ============================================================================
# File: scripts/analyze_results.py
# ============================================================================
"""Analyze batch results and compute aggregate metrics."""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path


def analyze_results(csv_path: str, json_path: str):
    """Generate analysis report from batch results."""
    
    df = pd.read_csv(csv_path)
    
    with open(json_path, 'r') as f:
        logs = json.load(f)
    
    print("\n" + "="*60)
    print("CONVERGENCE PROOF AGENT v2.0 - ANALYSIS REPORT")
    print("="*60)
    
    # Overall statistics
    print("\n## Overall Statistics")
    print(f"Total Algorithms: {len(df)}")
    print(f"Mean Final CRS: {df['CRS_Final'].mean():.3f}")
    print(f"Mean ERR: {df['ERR_Final'].mean():.3f}")
    print(f"Mean RP: {df['RP_Final'].mean():.3f}")
    print(f"Mean TFP: {df['TFP_Final'].mean():.3f}")
    
    # Verdict distribution
    print("\n## Verdict Distribution")
    if 'Overall Score (0–5)' in df.columns:
        verdict_map = {5: "PASS", 4: "PASS_MINOR", 3: "CONDITIONAL", 2: "FAIL", 1: "REJECT"}
        for score, verdict in verdict_map.items():
            count = (df['Overall Score (0–5)'] == score).sum()
            pct = count / len(df) * 100
            print(f"{verdict:15s}: {count:3d} ({pct:5.1f}%)")
    
    # CRS distribution
    print("\n## CRS Score Distribution")
    bins = [0, 0.4, 0.7, 1.0]
    labels = ["Weak (<0.4)", "Partial (0.4-0.7)", "Strong (≥0.7)"]
    crs_dist = pd.cut(df['CRS_Final'], bins=bins, labels=labels)
    print(crs_dist.value_counts())
    
    # Error resolution
    print("\n## Error Resolution")
    print(f"Total Errors Fixed: {df['Errors_Fixed'].sum()}")
    print(f"Total Errors Introduced: {df['Errors_Introduced'].sum()}")
    print(f"Net Improvement: {df['Net_Improvement'].sum()}")
    print(f"Mean Net Improvement per Algorithm: {df['Net_Improvement'].mean():.2f}")
    
    # Judge reliability
    if 'Mean_JRS' in df.columns and df['Mean_JRS'].sum() > 0:
        print("\n## Multi-Judge Statistics")
        print(f"Mean Judge Reliability Score (JRS): {df['Mean_JRS'].mean():.3f}")
        print(f"Mean Weighted CFRS: {df['Weighted_CFRS'].mean():.3f}")
        print(f"Algorithms with Outlier Judges: {(df['Outlier_Judges'] != '').sum()}")
    
    # Top performers
    print("\n## Top 5 Algorithms by CRS")
    top5 = df.nlargest(5, 'CRS_Final')[['Problem Statement', 'CRS_Final', 'Overall Score (0–5)']]
    for idx, row in top5.iterrows():
        algo_short = row['Problem Statement'][:40] + "..."
        print(f"{idx+1}. CRS={row['CRS_Final']:.3f}, Score={row['Overall Score (0–5)']} - {algo_short}")
    
    print("\n" + "="*60)
    print("Report complete!")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze batch results")
    parser.add_argument("--csv", default="processed_results_v2.csv", help="Results CSV")
    parser.add_argument("--json", default="batch_execution_v2.json", help="Execution log")
    
    args = parser.parse_args()
    
    analyze_results(args.csv, args.json)


if __name__ == "__main__":
    main()