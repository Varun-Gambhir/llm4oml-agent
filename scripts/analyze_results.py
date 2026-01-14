# ============================================================================
# File: scripts/analyze_results.py (COMPLETE VERSION)
# ============================================================================
"""Comprehensive result analysis with visualizations and detailed reports."""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class ResultsAnalyzer:
    """Comprehensive analysis of batch processing results."""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.csv_path = self.results_dir / "results.csv"
        self.json_path = self.output_dir / "batch_execution_log.json"
        self.summary_path = self.output_dir / "summary.txt"
        
        # Load data
        if not self.output_csv.exists():
            raise FileNotFoundError(f"Results CSV not found: {self.output_csv}")
        
        self.df = pd.read_csv(self.output_csv)
        
        if self.output_json.exists():
            with open(self.output_json, 'r') as f:
                self.batch_log = json.load(f)
        else:
            self.batch_log = {}
    
    def analyze(self):
        """Generate comprehensive analysis."""
        print(f"\n{'='*80}")
        print("DETAILED ANALYSIS OF BATCH RESULTS")
        print(f"{'='*80}\n")
        
        df = pd.read_csv(self.results_csv)
        completed = df[df['Processing_Status'] == 'COMPLETED']
        
        if len(completed_df) == 0:
            print("No completed algorithms to analyze.")
            return
        
        self._print_basic_stats(completed_df)
        self._print_convergence_analysis(completed_df)
        self._print_crs_analysis(completed_df)
        self._print_error_analysis(completed_df)
        
        if 'Num_Judges' in completed_df.columns and completed_df['Num_Judges'].max() > 1:
            self._print_judge_analysis(completed_df)
        
        self._print_correlation_analysis(completed_df)
    
    def _print_section(self, title: str):
        """Print section header."""
        print(f"\n{'='*80}")
        print(f"{title:^80}")
        print(f"{'='*80}\n")
    
    def analyze_verdict_distribution(self, df: pd.DataFrame):
        """Analyze verdict distribution."""
        print("\n" + "="*80)
        print("VERDICT DISTRIBUTION")
        print("="*80)
        
        verdict_counts = df['V2_Verdict'].value_counts()
        for verdict, count in verdict_counts.items():
            pct = count / len(df) * 100
            print(f"{verdict:15s}: {count:3d} ({pct:5.1f}%)")
    
    def analyze_crs_distribution(self, df: pd.DataFrame):
        """Analyze CRS distribution."""
        print(f"\n{'-'*80}")
        print("CRS SCORE DISTRIBUTION")
        print(f"{'-'*80}")
        
        bins = [0, 0.4, 0.7, 1.0]
        labels = ["Weak (<0.4)", "Partial (0.4-0.7)", "Strong (≥0.7)"]
        
        completed = df[df['Processing_Status'] == 'COMPLETED']
        if len(completed) > 0:
            crs_dist = pd.cut(completed_df['CRS_Final'], bins=bins, labels=labels, include_lowest=True)
            print(crs_dist.value_counts())
            
            # Plot if matplotlib available
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                completed_df['CRS_Final'].hist(bins=20)
                plt.xlabel('CRS Score')
                plt.ylabel('Frequency')
                plt.title('CRS Score Distribution')
                plt.savefig(self.output_dir / 'crs_distribution.png')
                print(f"📊 Saved visualization to {self.output_dir / 'crs_distribution.png'}")
            except Exception as e:
                print(f"Could not generate plots: {e}")
        
        return summary_text


def main():
    parser = argparse.ArgumentParser(description="Analyze batch processing results")
    parser.add_argument("--results-dir", required=True, help="Path to batch results directory")
    parser.add_argument("--output", default="analysis_report.txt", help="Output report file")
    
    args = parser.parse_args()
    
    analyzer = ResultsAnalyzer(args.results)
    analyzer.generate_full_report()
    
    print(f"\n✅ Analysis complete! Check {args.output}/summary.txt for full report.")


if __name__ == "__main__":
    main()