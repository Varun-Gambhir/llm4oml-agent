# ============================================================================
# File: scripts/analyze_results.py (COMPLETE VERSION)
# ============================================================================
"""Comprehensive result analysis with statistics, visualizations, and detailed reports."""

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
        self.json_path = self.results_dir / "batch_execution_log.json"
        self.summary_path = self.results_dir / "summary.txt"
        self.analysis_path = self.results_dir / "detailed_analysis.txt"
        
        # Load data
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Results CSV not found: {self.csv_path}")
        
        self.df = pd.read_csv(self.csv_path)
        
        if self.json_path.exists():
            with open(self.json_path, 'r') as f:
                self.batch_log = json.load(f)
        else:
            self.batch_log = {}
        
        # Filter to completed algorithms
        self.completed_df = self.df[self.df['Processing_Status'] == 'COMPLETED'].copy()
    
    def generate_full_report(self):
        """Generate comprehensive analysis report."""
        print(f"\n{'='*80}")
        print("DETAILED ANALYSIS OF BATCH RESULTS")
        print(f"{'='*80}\n")
        
        if len(self.completed_df) == 0:
            print("❌ No completed algorithms to analyze.")
            return
        
        report_lines = []
        
        # Header
        report_lines.append("="*80)
        report_lines.append("COMPREHENSIVE ANALYSIS REPORT")
        report_lines.append("="*80)
        report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Results Directory: {self.results_dir}")
        report_lines.append(f"Total Algorithms: {len(self.df)}")
        report_lines.append(f"Completed: {len(self.completed_df)}")
        report_lines.append(f"Failed: {(self.df['Processing_Status'] == 'FAILED').sum()}")
        
        # 1. Basic Statistics
        report_lines.extend(self._analyze_basic_stats())
        
        # 2. Verdict Distribution
        report_lines.extend(self._analyze_verdict_distribution())
        
        # 3. Convergence Analysis
        report_lines.extend(self._analyze_convergence())
        
        # 4. CRS Analysis
        report_lines.extend(self._analyze_crs())
        
        # 5. Error Analysis
        report_lines.extend(self._analyze_errors())
        
        # 6. Judge Analysis (if multi-judge)
        if self.completed_df['Num_Judges'].max() > 1:
            report_lines.extend(self._analyze_judges())
        
        # 7. Correlation Analysis
        report_lines.extend(self._analyze_correlations())
        
        # 8. Performance Analysis
        report_lines.extend(self._analyze_performance())
        
        # 9. Top/Bottom Performers
        report_lines.extend(self._analyze_performers())
        
        # Write report
        report_text = "\n".join(report_lines)
        with open(self.analysis_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        
        # Print to console
        print(report_text)
        print(f"\n✅ Detailed analysis saved to: {self.analysis_path}")
        
        # Generate visualizations if matplotlib available
        self._generate_visualizations()
    
    def _analyze_basic_stats(self) -> list:
        """Analyze basic statistics."""
        lines = []
        lines.append(f"\n{'='*80}")
        lines.append("1. BASIC STATISTICS")
        lines.append(f"{'='*80}")
        
        df = self.completed_df
        
        lines.append(f"\nProcessing Time:")
        lines.append(f"  Total: {df['Processing_Time_Seconds'].sum():.2f} seconds ({df['Processing_Time_Seconds'].sum()/3600:.2f} hours)")
        lines.append(f"  Mean: {df['Processing_Time_Seconds'].mean():.2f} seconds")
        lines.append(f"  Median: {df['Processing_Time_Seconds'].median():.2f} seconds")
        lines.append(f"  Min: {df['Processing_Time_Seconds'].min():.2f} seconds")
        lines.append(f"  Max: {df['Processing_Time_Seconds'].max():.2f} seconds")
        
        lines.append(f"\nIterations:")
        lines.append(f"  Mean: {df['Total_Iterations'].mean():.2f}")
        lines.append(f"  Median: {df['Total_Iterations'].median():.1f}")
        lines.append(f"  Mode: {df['Total_Iterations'].mode().values[0] if len(df['Total_Iterations'].mode()) > 0 else 'N/A'}")
        lines.append(f"  Range: {df['Total_Iterations'].min()}-{df['Total_Iterations'].max()}")
        
        return lines
    
    def _analyze_verdict_distribution(self) -> list:
        """Analyze verdict distribution."""
        lines = []
        lines.append(f"\n{'='*80}")
        lines.append("2. VERDICT DISTRIBUTION")
        lines.append(f"{'='*80}")
        
        df = self.completed_df
        
        # V1 (Initial) Verdicts
        lines.append(f"\nV1 (Initial) Verdicts:")
        v1_counts = df['V1_Verdict'].value_counts().sort_index()
        for verdict, count in v1_counts.items():
            pct = count / len(df) * 100
            lines.append(f"  {verdict:15s}: {count:3d} ({pct:5.1f}%)")
        
        # V2 (Final) Verdicts
        lines.append(f"\nV2 (Final) Verdicts:")
        v2_counts = df['V2_Verdict'].value_counts().sort_index()
        for verdict, count in v2_counts.items():
            pct = count / len(df) * 100
            lines.append(f"  {verdict:15s}: {count:3d} ({pct:5.1f}%)")
        
        # Verdict Changes
        lines.append(f"\nVerdict Changes (V1 → V2):")
        improved = ((df['V1_Verdict'] == 'FAIL') & (df['V2_Verdict'].isin(['PASS', 'PASS_MINOR', 'CONDITIONAL']))).sum()
        degraded = ((df['V1_Verdict'].isin(['PASS', 'PASS_MINOR'])) & (df['V2_Verdict'].isin(['FAIL', 'REJECT']))).sum()
        unchanged = (df['V1_Verdict'] == df['V2_Verdict']).sum()
        
        lines.append(f"  Improved: {improved} ({improved/len(df)*100:.1f}%)")
        lines.append(f"  Degraded: {degraded} ({degraded/len(df)*100:.1f}%)")
        lines.append(f"  Unchanged: {unchanged} ({unchanged/len(df)*100:.1f}%)")
        
        return lines
    
    def _analyze_convergence(self) -> list:
        """Analyze convergence statistics."""
        lines = []
        lines.append(f"\n{'='*80}")
        lines.append("3. CONVERGENCE ANALYSIS")
        lines.append(f"{'='*80}")
        
        df = self.completed_df
        
        converged = df['Converged'].sum()
        lines.append(f"\nConvergence Rate: {converged}/{len(df)} ({converged/len(df)*100:.1f}%)")
        
        # Convergence by iteration count
        lines.append(f"\nConvergence by Iteration:")
        for iter_num in sorted(df['Total_Iterations'].unique()):
            iter_df = df[df['Total_Iterations'] == iter_num]
            iter_converged = iter_df['Converged'].sum()
            lines.append(f"  {iter_num} iterations: {iter_converged}/{len(iter_df)} ({iter_converged/len(iter_df)*100:.1f}% converged)")
        
        # Average iterations for converged vs not converged
        converged_df = df[df['Converged'] == True]
        not_converged_df = df[df['Converged'] == False]
        
        if len(converged_df) > 0:
            lines.append(f"\nAverage Iterations (Converged): {converged_df['Total_Iterations'].mean():.2f}")
        if len(not_converged_df) > 0:
            lines.append(f"Average Iterations (Not Converged): {not_converged_df['Total_Iterations'].mean():.2f}")
        
        return lines
    
    def _analyze_crs(self) -> list:
        """Analyze CRS metrics."""
        lines = []
        lines.append(f"\n{'='*80}")
        lines.append("4. CORRECTION REASONING SCORE (CRS) ANALYSIS")
        lines.append(f"{'='*80}")
        
        df = self.completed_df
        
        # Overall CRS statistics
        lines.append(f"\nCRS Statistics:")
        lines.append(f"  Mean: {df['CRS_Final'].mean():.3f}")
        lines.append(f"  Median: {df['CRS_Final'].median():.3f}")
        lines.append(f"  Std Dev: {df['CRS_Final'].std():.3f}")
        lines.append(f"  Min: {df['CRS_Final'].min():.3f}")
        lines.append(f"  Max: {df['CRS_Final'].max():.3f}")
        
        # CRS Components
        lines.append(f"\nCRS Components (Mean):")
        lines.append(f"  ERR (Error Resolution Rate): {df['ERR_Final'].mean():.3f}")
        lines.append(f"  RP (Regression Penalty): {df['RP_Final'].mean():.3f}")
        lines.append(f"  TFP (Targeted Fix Precision): {df['TFP_Final'].mean():.3f}")
        
        # CRS Distribution
        lines.append(f"\nCRS Distribution:")
        strong = (df['CRS_Final'] >= 0.7).sum()
        partial = ((df['CRS_Final'] >= 0.4) & (df['CRS_Final'] < 0.7)).sum()
        weak = (df['CRS_Final'] < 0.4).sum()
        
        lines.append(f"  Strong (≥0.7):     {strong:3d} ({strong/len(df)*100:5.1f}%)")
        lines.append(f"  Partial (0.4-0.7): {partial:3d} ({partial/len(df)*100:5.1f}%)")
        lines.append(f"  Weak (<0.4):       {weak:3d} ({weak/len(df)*100:5.1f}%)")
        
        # CRS by verdict
        lines.append(f"\nCRS by Final Verdict:")
        for verdict in sorted(df['V2_Verdict'].unique()):
            verdict_df = df[df['V2_Verdict'] == verdict]
            if len(verdict_df) > 0:
                lines.append(f"  {verdict:15s}: Mean CRS = {verdict_df['CRS_Final'].mean():.3f}")
        
        # CRS progression
        lines.append(f"\nCRS Progression (Mean):")
        if df['CRS_Iter_1'].sum() > 0:
            lines.append(f"  Iteration 1: {df[df['CRS_Iter_1'] > 0]['CRS_Iter_1'].mean():.3f}")
        if df['CRS_Iter_2'].sum() > 0:
            lines.append(f"  Iteration 2: {df[df['CRS_Iter_2'] > 0]['CRS_Iter_2'].mean():.3f}")
        if df['CRS_Iter_3'].sum() > 0:
            lines.append(f"  Iteration 3: {df[df['CRS_Iter_3'] > 0]['CRS_Iter_3'].mean():.3f}")
        
        return lines
    
    def _analyze_errors(self) -> list:
        """Analyze error tracking metrics."""
        lines = []
        lines.append(f"\n{'='*80}")
        lines.append("5. ERROR RESOLUTION ANALYSIS")
        lines.append(f"{'='*80}")
        
        df = self.completed_df
        
        lines.append(f"\nError Counts:")
        lines.append(f"  Total Errors V1 (Initial): {df['Total_Errors_V1'].sum()}")
        lines.append(f"  Total Errors Final: {df['Total_Errors_Final'].sum()}")
        lines.append(f"  Total Errors Fixed: {df['Errors_Fixed'].sum()}")
        lines.append(f"  Total Errors Introduced: {df['Errors_Introduced'].sum()}")
        lines.append(f"  Net Improvement: {df['Net_Improvement'].sum()}")
        
        lines.append(f"\nPer-Algorithm Averages:")
        lines.append(f"  Avg Errors V1: {df['Total_Errors_V1'].mean():.2f}")
        lines.append(f"  Avg Errors Final: {df['Total_Errors_Final'].mean():.2f}")
        lines.append(f"  Avg Errors Fixed: {df['Errors_Fixed'].mean():.2f}")
        lines.append(f"  Avg Errors Introduced: {df['Errors_Introduced'].mean():.2f}")
        lines.append(f"  Avg Net Improvement: {df['Net_Improvement'].mean():.2f}")
        
        # Error reduction rate
        error_reduction = ((df['Total_Errors_V1'] - df['Total_Errors_Final']) / df['Total_Errors_V1'].replace(0, 1)).mean()
        lines.append(f"\nAverage Error Reduction Rate: {error_reduction*100:.1f}%")
        
        # Error types (V1 vs Final)
        lines.append(f"\nError Types (Initial vs Final):")
        ha_v1 = (df['V1_Hallucination_Error'] == 'yes').sum()
        ha_v2 = (df['V2_Hallucination_Error'] == 'yes').sum()
        ms_v1 = (df['V1_Missing_Step'] == 'yes').sum()
        ms_v2 = (df['V2_Missing_Step'] == 'yes').sum()
        op_v1 = (df['V1_Operator_Error'] == 'yes').sum()
        op_v2 = (df['V2_Operator_Error'] == 'yes').sum()
        
        lines.append(f"  Hallucination: {ha_v1} → {ha_v2} (Δ {ha_v2-ha_v1:+d})")
        lines.append(f"  Missing Steps: {ms_v1} → {ms_v2} (Δ {ms_v2-ms_v1:+d})")
        lines.append(f"  Operator Errors: {op_v1} → {op_v2} (Δ {op_v2-op_v1:+d})")
        
        return lines
    
    def _analyze_judges(self) -> list:
        """Analyze multi-judge metrics."""
        lines = []
        lines.append(f"\n{'='*80}")
        lines.append("6. MULTI-JUDGE ANALYSIS")
        lines.append(f"{'='*80}")
        
        df = self.completed_df
        judge_df = df[df['Num_Judges'] > 1]
        
        if len(judge_df) == 0:
            lines.append("\nNo multi-judge evaluations found.")
            return lines
        
        lines.append(f"\nMulti-Judge Statistics:")
        lines.append(f"  Number of Judges Used: {int(judge_df['Num_Judges'].max())}")
        lines.append(f"  Algorithms with Multi-Judge: {len(judge_df)}")
        
        lines.append(f"\nJudge Reliability (JRS):")
        lines.append(f"  Mean JRS: {judge_df['Mean_JRS'].mean():.3f}")
        lines.append(f"  Median JRS: {judge_df['Mean_JRS'].median():.3f}")
        lines.append(f"  Min JRS: {judge_df['Mean_JRS'].min():.3f}")
        lines.append(f"  Max JRS: {judge_df['Mean_JRS'].max():.3f}")
        
        lines.append(f"\nWeighted CFRS:")
        lines.append(f"  Mean: {judge_df['Weighted_CFRS'].mean():.3f}")
        lines.append(f"  Median: {judge_df['Weighted_CFRS'].median():.3f}")
        
        # Outlier analysis
        has_outliers = judge_df['Outlier_Judges'].str.len() > 0
        lines.append(f"\nOutlier Judges:")
        lines.append(f"  Algorithms with Outliers: {has_outliers.sum()} ({has_outliers.sum()/len(judge_df)*100:.1f}%)")
        
        return lines
    
    def _analyze_correlations(self) -> list:
        """Analyze correlations between metrics."""
        lines = []
        lines.append(f"\n{'='*80}")
        lines.append("7. CORRELATION ANALYSIS")
        lines.append(f"{'='*80}")
        
        df = self.completed_df
        
        # Key correlations
        correlations = [
            ('CRS_Final', 'Overall_Score', 'CRS vs Overall Score'),
            ('Total_Iterations', 'CRS_Final', 'Iterations vs CRS'),
            ('Total_Errors_V1', 'CRS_Final', 'Initial Errors vs CRS'),
            ('ERR_Final', 'CRS_Final', 'ERR vs CRS'),
            ('Processing_Time_Seconds', 'Total_Iterations', 'Time vs Iterations'),
            ('Proof_Length_Growth', 'CRS_Final', 'Proof Growth vs CRS'),
        ]
        
        lines.append("\nKey Correlations:")
        for col1, col2, desc in correlations:
            if col1 in df.columns and col2 in df.columns:
                corr = df[col1].corr(df[col2])
                lines.append(f"  {desc:40s}: {corr:+.3f}")
        
        # Correlation matrix for CRS components
        lines.append("\nCRS Components Correlation Matrix:")
        crs_cols = ['ERR_Final', 'RP_Final', 'TFP_Final', 'CRS_Final']
        if all(col in df.columns for col in crs_cols):
            corr_matrix = df[crs_cols].corr()
            lines.append("\n" + corr_matrix.to_string())
        
        return lines
    
    def _analyze_performance(self) -> list:
        """Analyze performance metrics."""
        lines = []
        lines.append(f"\n{'='*80}")
        lines.append("8. PERFORMANCE ANALYSIS")
        lines.append(f"{'='*80}")
        
        df = self.completed_df
        
        # Completeness scores
        lines.append(f"\nCompleteness Scores:")
        lines.append(f"  V1 Mean: {df['V1_Completeness_Score'].mean():.2f}")
        lines.append(f"  V2 Mean: {df['V2_Completeness_Score'].mean():.2f}")
        lines.append(f"  Improvement: {(df['V2_Completeness_Score'] - df['V1_Completeness_Score']).mean():+.2f}")
        
        # Assumption scores
        lines.append(f"\nAssumption Use Scores:")
        lines.append(f"  V1 Mean: {df['V1_Assumption_Score'].mean():.2f}")
        lines.append(f"  V2 Mean: {df['V2_Assumption_Score'].mean():.2f}")
        lines.append(f"  Improvement: {(df['V2_Assumption_Score'] - df['V1_Assumption_Score']).mean():+.2f}")
        
        # Proof length analysis
        lines.append(f"\nProof Length Analysis:")
        lines.append(f"  Initial Mean: {df['Initial_Proof_Length'].mean():.0f} chars")
        lines.append(f"  Final Mean: {df['Final_Proof_Length'].mean():.0f} chars")
        lines.append(f"  Mean Growth: {df['Proof_Length_Growth'].mean():.0f} chars ({df['Proof_Length_Growth'].mean()/df['Initial_Proof_Length'].mean()*100:.1f}%)")
        
        return lines
    
    def _analyze_performers(self) -> list:
        """Analyze top and bottom performers."""
        lines = []
        lines.append(f"\n{'='*80}")
        lines.append("9. TOP AND BOTTOM PERFORMERS")
        lines.append(f"{'='*80}")
        
        df = self.completed_df
        
        # Top 5 by CRS
        lines.append(f"\nTop 5 Algorithms by CRS:")
        top5 = df.nlargest(5, 'CRS_Final')
        for i, (idx, row) in enumerate(top5.iterrows(), 1):
            algo = str(row['Problem Statement'])[:60] + "..."
            lines.append(f"\n{i}. {algo}")
            lines.append(f"   CRS: {row['CRS_Final']:.3f} | Score: {row['Overall_Score']}/5 | Iterations: {row['Total_Iterations']}")
            lines.append(f"   ERR: {row['ERR_Final']:.3f} | RP: {row['RP_Final']:.3f} | TFP: {row['TFP_Final']:.3f}")
            lines.append(f"   Errors Fixed: {row['Errors_Fixed']} | Introduced: {row['Errors_Introduced']} | Net: {row['Net_Improvement']:+d}")
        
        # Bottom 5 by CRS
        if len(df) >= 5:
            lines.append(f"\nBottom 5 Algorithms by CRS:")
            bottom5 = df.nsmallest(5, 'CRS_Final')
            for i, (idx, row) in enumerate(bottom5.iterrows(), 1):
                algo = str(row['Problem Statement'])[:60] + "..."
                lines.append(f"\n{i}. {algo}")
                lines.append(f"   CRS: {row['CRS_Final']:.3f} | Score: {row['Overall_Score']}/5 | Iterations: {row['Total_Iterations']}")
                lines.append(f"   ERR: {row['ERR_Final']:.3f} | RP: {row['RP_Final']:.3f} | TFP: {row['TFP_Final']:.3f}")
                lines.append(f"   Errors Fixed: {row['Errors_Fixed']} | Introduced: {row['Errors_Introduced']} | Net: {row['Net_Improvement']:+d}")
        
        # Most improved
        lines.append(f"\nMost Improved (by Net Error Reduction):")
        most_improved = df.nlargest(5, 'Net_Improvement')
        for i, (idx, row) in enumerate(most_improved.iterrows(), 1):
            algo = str(row['Problem Statement'])[:60] + "..."
            lines.append(f"{i}. {algo}")
            lines.append(f"   Net Improvement: {row['Net_Improvement']:+d} errors | CRS: {row['CRS_Final']:.3f}")
        
        return lines
    
    def _generate_visualizations(self):
        """Generate visualization plots if matplotlib is available."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            sns.set_style("whitegrid")
            df = self.completed_df
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Batch Processing Analysis', fontsize=16, fontweight='bold')
            
            # 1. CRS Distribution
            axes[0, 0].hist(df['CRS_Final'], bins=20, edgecolor='black')
            axes[0, 0].set_xlabel('CRS Score')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('CRS Distribution')
            axes[0, 0].axvline(0.7, color='g', linestyle='--', label='Strong (≥0.7)')
            axes[0, 0].axvline(0.4, color='orange', linestyle='--', label='Partial (≥0.4)')
            axes[0, 0].legend()
            
            # 2. Verdict Distribution
            verdict_counts = df['V2_Verdict'].value_counts()
            axes[0, 1].bar(verdict_counts.index, verdict_counts.values)
            axes[0, 1].set_xlabel('Verdict')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Final Verdict Distribution')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 3. CRS vs Overall Score
            axes[0, 2].scatter(df['CRS_Final'], df['Overall_Score'], alpha=0.6)
            axes[0, 2].set_xlabel('CRS')
            axes[0, 2].set_ylabel('Overall Score')
            axes[0, 2].set_title('CRS vs Overall Score')
            
            # 4. Error Resolution
            error_data = {
                'Initial': df['Total_Errors_V1'].sum(),
                'Final': df['Total_Errors_Final'].sum(),
                'Fixed': df['Errors_Fixed'].sum(),
                'Introduced': df['Errors_Introduced'].sum()
            }
            axes[1, 0].bar(error_data.keys(), error_data.values())
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('Error Resolution Summary')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 5. Iterations vs CRS
            axes[1, 1].scatter(df['Total_Iterations'], df['CRS_Final'], alpha=0.6)
            axes[1, 1].set_xlabel('Iterations')
            axes[1, 1].set_ylabel('CRS')
            axes[1, 1].set_title('Iterations vs CRS')
            
            # 6. CRS Components
            crs_components = {
                'ERR': df['ERR_Final'].mean(),
                'RP': df['RP_Final'].mean(),
                'TFP': df['TFP_Final'].mean()
            }
            axes[1, 2].bar(crs_components.keys(), crs_components.values())
            axes[1, 2].set_ylabel('Mean Value')
            axes[1, 2].set_title('Mean CRS Components')
            axes[1, 2].set_ylim([0, 1])
            
            plt.tight_layout()
            plot_path = self.results_dir / "analysis_plots.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualizations saved to: {plot_path}")
            
        except ImportError:
            print("⚠️  matplotlib not available. Skipping visualizations.")
            print("   Install with: pip install matplotlib seaborn")
        except Exception as e:
            print(f"⚠️  Could not generate visualizations: {e}")


def main():
    parser = argparse.ArgumentParser(description="Analyze batch processing results")
    parser.add_argument("--results-dir", required=True, help="Path to batch results directory")
    parser.add_argument("--output", help="Custom output path for analysis report")
    
    args = parser.parse_args()
    
    try:
        analyzer = ResultsAnalyzer(args.results_dir)
        analyzer.generate_full_report()
        
        print(f"\n✅ Analysis complete!")
        print(f"   CSV: {analyzer.csv_path}")
        print(f"   Summary: {analyzer.summary_path}")
        print(f"   Detailed Analysis: {analyzer.analysis_path}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()