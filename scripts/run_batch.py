# ============================================================================
# File: scripts/run_batch.py (COMPLETE VERSION)
# ============================================================================
"""Batch processor with complete state tracking and CSV generation."""

import pandas as pd
import json
import argparse
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.workflow import ProofWorkflow


class BatchProcessor:
    """Process multiple algorithms with full tracking and CSV output."""
    
    def __init__(
        self,
        input_csv: str,
        output_dir: str,
        prover_model: str,
        evaluator_models: list,
        use_multi_judge: bool,
        max_iterations: int
    ):
        self.input_csv = input_csv
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Output files
        self.output_csv = self.output_dir / "results.csv"
        self.output_json = self.output_dir / "batch_execution_log.json"
        self.summary_txt = self.output_dir / "summary.txt"
        
        self.prover_model = prover_model
        self.evaluator_models = evaluator_models
        self.use_multi_judge = use_multi_judge
        self.max_iterations = max_iterations
        
        # Batch-level log
        self.batch_log = {
            "metadata": {
                "start_time": datetime.now().isoformat(),
                "input_file": str(input_csv),
                "configuration": {
                    "prover_model": prover_model,
                    "evaluator_models": evaluator_models,
                    "use_multi_judge": use_multi_judge,
                    "max_iterations": max_iterations
                }
            },
            "algorithms": []
        }
    
    def process(self):
        """Process all algorithms in CSV."""
        print(f"\n{'='*80}")
        print(f"BATCH PROCESSING STARTED")
        print(f"Input: {self.input_csv}")
        print(f"Output Directory: {self.output_dir}")
        print(f"{'='*80}\n")
        
        # Load CSV
        try:
            df = pd.read_csv(self.input_csv, header=1)
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            return
        
        # Filter empty rows
        df = df[df['Problem Statement'].notna()].reset_index(drop=True)
        total_algos = len(df)
        print(f"📊 Found {total_algos} algorithms to process.\n")
        
        # Prepare output dataframe
        results_df = self._prepare_results_dataframe(df)
        
        # Process each row
        for index, row in df.iterrows():
            print(f"\n{'#'*80}")
            print(f"# Processing Algorithm {index + 1}/{total_algos}")
            print(f"{'#'*80}")
            
            try:
                self._process_single_algorithm(index, row, results_df)
                
                # Save progress after each algorithm
                results_df.to_csv(self.output_csv, index=False)
                self._save_batch_log()
                
                print(f"✓ Progress saved. Completed {index + 1}/{total_algos}")
                
            except Exception as e:
                print(f"❌ Error processing row {index}: {e}")
                self.batch_log["algorithms"].append({
                    "row_index": index,
                    "error": str(e),
                    "status": "failed"
                })
                
                # Mark as failed in CSV
                results_df.at[index, 'Processing_Status'] = 'FAILED'
                results_df.at[index, 'Error_Message'] = str(e)
                results_df.to_csv(self.output_csv, index=False)
        
        # Finalize
        self._finalize_batch(results_df)
        
        print(f"\n{'='*80}")
        print(f"✅ BATCH PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"Results CSV: {self.output_csv}")
        print(f"Execution Log: {self.output_json}")
        print(f"Summary: {self.summary_txt}")
        print(f"{'='*80}\n")
    
    def _prepare_results_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare results dataframe with all necessary columns."""
        results_df = df.copy()
        
        # Add V2.0 columns
        new_columns = {
            # Processing Status
            'Processing_Status': 'PENDING',
            'Error_Message': '',
            'Processing_Time_Seconds': 0.0,
            
            # V1 Metrics (First Iteration)
            'V1_Hallucination_Error': '',
            'V1_Missing_Step': '',
            'V1_Operator_Error': '',
            'V1_Completeness_Score': 0,
            'V1_Assumption_Score': 0,
            'V1_Verdict': '',
            
            # V2 Metrics (Final Iteration)
            'V2_Hallucination_Error': '',
            'V2_Missing_Step': '',
            'V2_Operator_Error': '',
            'V2_Completeness_Score': 0,
            'V2_Assumption_Score': 0,
            'V2_Verdict': '',
            'Overall_Score': 0,
            
            # Iteration Information
            'Total_Iterations': 0,
            'Converged': False,
            
            # CRS Metrics (per iteration)
            'CRS_Iter_1': 0.0,
            'CRS_Iter_2': 0.0,
            'CRS_Iter_3': 0.0,
            'CRS_Final': 0.0,
            
            # CRS Components (final)
            'ERR_Final': 0.0,
            'RP_Final': 0.0,
            'TFP_Final': 0.0,
            
            # Error Tracking
            'Total_Errors_V1': 0,
            'Total_Errors_Final': 0,
            'Errors_Fixed': 0,
            'Errors_Introduced': 0,
            'Net_Improvement': 0,
            
            # Multi-Judge Metrics
            'Num_Judges': 0,
            'Mean_JRS': 0.0,
            'Outlier_Judges': '',
            'Judge_Agreement_ESA': 0.0,
            'Weighted_CFRS': 0.0,
            
            # Proof Information
            'Final_Proof_Length': 0,
            'Proof_Length_Growth': 0,
            
            # Feedback Summary
            'Final_Feedback_Preview': '',
            
            # Model Information
            'Model_Used': ''
        }
        
        for col, default in new_columns.items():
            if col not in results_df.columns:
                results_df[col] = default
        
        return results_df
    
    def _process_single_algorithm(self, index: int, row: pd.Series, results_df: pd.DataFrame):
        """Process a single algorithm and update results."""
        algo_text = row['Problem Statement']
        assump_text = row['Assumptions Used']
        
        print(f"Algorithm: {algo_text[:70]}...")
        print(f"Assumptions: {assump_text[:70]}...")
        
        # Create algorithm-specific output directory
        algo_dir = self.output_dir / f"algorithm_{index:03d}"
        algo_dir.mkdir(exist_ok=True)
        
        # Track processing time
        import time
        start_time = time.time()
        
        # Run workflow
        workflow = ProofWorkflow(
            prover_model=self.prover_model,
            evaluator_models=self.evaluator_models,
            use_multi_judge=self.use_multi_judge,
            max_iterations=self.max_iterations,
            output_dir=str(algo_dir)
        )
        
        final_state, tracker = workflow.run(algo_text, assump_text)
        
        processing_time = time.time() - start_time
        
        # Get execution log
        exec_log = tracker.get_log()
        
        # Update results dataframe
        self._update_results_row(index, results_df, exec_log, final_state, processing_time)
        
        # Save to batch log
        self.batch_log["algorithms"].append({
            "row_index": index,
            "algorithm": algo_text,
            "status": "completed",
            "processing_time": processing_time,
            "final_verdict": final_state.get("verdict"),
            "iterations": exec_log.get("final_results", {}).get("total_iterations", 0),
            "output_directory": str(algo_dir),
            "execution_log": exec_log  # Full execution log
        })
    
    def _update_results_row(
        self,
        index: int,
        df: pd.DataFrame,
        exec_log: Dict,
        final_state: Dict,
        processing_time: float
    ):
        """Update a single row in results dataframe."""
        to_yn = lambda x: "yes" if x else "no"
        
        # Processing Status
        df.at[index, 'Processing_Status'] = 'COMPLETED'
        df.at[index, 'Processing_Time_Seconds'] = round(processing_time, 2)
        df.at[index, 'Model_Used'] = f"Prover: {self.prover_model}, Judges: {len(self.evaluator_models)}"
        
        # Iteration Information
        iterations = exec_log.get("iterations", [])
        df.at[index, 'Total_Iterations'] = len(iterations)
        df.at[index, 'Converged'] = exec_log.get("final_results", {}).get("convergence_achieved", False)
        
        # V1 Metrics (First Iteration)
        if len(iterations) >= 1 and "evaluator" in iterations[0]["nodes"]:
            v1_eval = iterations[0]["nodes"]["evaluator"]
            v1_metrics = v1_eval.get("metrics", {})
            
            df.at[index, 'V1_Hallucination_Error'] = to_yn(v1_metrics.get("HA", False))
            df.at[index, 'V1_Missing_Step'] = to_yn(v1_metrics.get("MS", False))
            df.at[index, 'V1_Operator_Error'] = to_yn(v1_metrics.get("OP", False))
            df.at[index, 'V1_Completeness_Score'] = v1_metrics.get("completeness_score", 0)
            df.at[index, 'V1_Assumption_Score'] = v1_metrics.get("assumption_use_score", 0)
            df.at[index, 'V1_Verdict'] = v1_eval.get("verdict", "")
            
            # Count V1 errors
            v1_error_set = v1_eval.get("error_set", {})
            total_v1_errors = sum(len(v1_error_set.get(k, [])) for k in 
                                  ["hallucinations", "missing_steps", "operator_errors", "assumption_violations"])
            df.at[index, 'Total_Errors_V1'] = total_v1_errors
        
        # V2 Metrics (Final Iteration)
        if len(iterations) >= 1 and "evaluator" in iterations[-1]["nodes"]:
            vf_eval = iterations[-1]["nodes"]["evaluator"]
            vf_metrics = vf_eval.get("metrics", {})
            
            df.at[index, 'V2_Hallucination_Error'] = to_yn(vf_metrics.get("HA", False))
            df.at[index, 'V2_Missing_Step'] = to_yn(vf_metrics.get("MS", False))
            df.at[index, 'V2_Operator_Error'] = to_yn(vf_metrics.get("OP", False))
            df.at[index, 'V2_Completeness_Score'] = vf_metrics.get("completeness_score", 0)
            df.at[index, 'V2_Assumption_Score'] = vf_metrics.get("assumption_use_score", 0)
            df.at[index, 'V2_Verdict'] = vf_eval.get("verdict", "")
            
            # Overall score
            verdict_scores = {"PASS": 5, "PASS_MINOR": 4, "CONDITIONAL": 3, "FAIL": 2, "REJECT": 1}
            df.at[index, 'Overall_Score'] = verdict_scores.get(vf_eval.get("verdict", "FAIL"), 0)
            
            # Count final errors
            vf_error_set = vf_eval.get("error_set", {})
            total_vf_errors = sum(len(vf_error_set.get(k, [])) for k in 
                                  ["hallucinations", "missing_steps", "operator_errors", "assumption_violations"])
            df.at[index, 'Total_Errors_Final'] = total_vf_errors
            
            # Feedback preview
            feedback = vf_eval.get("feedback", "")
            df.at[index, 'Final_Feedback_Preview'] = feedback[:200] + "..." if len(feedback) > 200 else feedback
        
        # CRS Metrics (for each iteration that has them)
        for i, iteration in enumerate(iterations):
            if "evaluator" in iteration["nodes"]:
                corr_metrics = iteration["nodes"]["evaluator"].get("correction_metrics")
                if corr_metrics:
                    crs_value = corr_metrics.get("correction_reasoning_score", 0.0)
                    if i < 3:  # Only store first 3
                        df.at[index, f'CRS_Iter_{i+1}'] = round(crs_value, 3)
                    
                    # Always update final
                    if i == len(iterations) - 1:
                        df.at[index, 'CRS_Final'] = round(crs_value, 3)
                        df.at[index, 'ERR_Final'] = round(corr_metrics.get("error_resolution_rate", 0.0), 3)
                        df.at[index, 'RP_Final'] = round(corr_metrics.get("regression_penalty", 0.0), 3)
                        df.at[index, 'TFP_Final'] = round(corr_metrics.get("targeted_fix_precision", 0.0), 3)
                        
                        df.at[index, 'Errors_Fixed'] = corr_metrics.get("errors_fixed", 0)
                        df.at[index, 'Errors_Introduced'] = corr_metrics.get("errors_introduced", 0)
        
        # Net improvement
        df.at[index, 'Net_Improvement'] = df.at[index, 'Errors_Fixed'] - df.at[index, 'Errors_Introduced']
        
        # Multi-Judge Metrics (from final iteration)
        if len(iterations) >= 1 and "evaluator" in iterations[-1]["nodes"]:
            final_eval = iterations[-1]["nodes"]["evaluator"]
            judge_evals = final_eval.get("judge_evaluations", [])
            judge_reliability = final_eval.get("judge_reliability", [])
            
            if judge_evals:
                df.at[index, 'Num_Judges'] = len(judge_evals)
                df.at[index, 'Weighted_CFRS'] = round(final_eval.get("metrics", {}).get("weighted_cfrs", 0.0), 3)
            
            if judge_reliability:
                df.at[index, 'Mean_JRS'] = round(sum(judge_reliability) / len(judge_reliability), 3)
        
        # Proof Information
        if len(iterations) >= 1:
            if "prover" in iterations[0]["nodes"]:
                initial_length = iterations[0]["nodes"]["prover"].get("proof_length", 0)
            else:
                initial_length = 0
            
            if "prover" in iterations[-1]["nodes"]:
                final_length = iterations[-1]["nodes"]["prover"].get("proof_length", 0)
            else:
                final_length = 0
            
            df.at[index, 'Final_Proof_Length'] = final_length
            df.at[index, 'Proof_Length_Growth'] = final_length - initial_length
    
    def _save_batch_log(self):
        """Save batch execution log."""
        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(self.batch_log, f, indent=2, default=str)
    
    def _finalize_batch(self, results_df: pd.DataFrame):
        """Finalize batch processing with summary."""
        self.batch_log["metadata"]["end_time"] = datetime.now().isoformat()
        self.batch_log["metadata"]["total_algorithms"] = len(results_df)
        self.batch_log["metadata"]["completed"] = (results_df['Processing_Status'] == 'COMPLETED').sum()
        self.batch_log["metadata"]["failed"] = (results_df['Processing_Status'] == 'FAILED').sum()
        
        # Save final batch log
        self._save_batch_log()
        
        # Generate summary report
        self._generate_summary(results_df)
    
    def _generate_summary(self, df: pd.DataFrame):
        """Generate text summary of batch results."""
        summary_lines = []
        summary_lines.append("="*80)
        summary_lines.append("BATCH PROCESSING SUMMARY")
        summary_lines.append("="*80)
        summary_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append(f"Input File: {self.input_csv}")
        summary_lines.append(f"Output Directory: {self.output_dir}")
        
        summary_lines.append(f"\n{'-'*80}")
        summary_lines.append("CONFIGURATION")
        summary_lines.append(f"{'-'*80}")
        summary_lines.append(f"Prover Model: {self.prover_model}")
        summary_lines.append(f"Evaluator Models: {', '.join(self.evaluator_models)}")
        summary_lines.append(f"Multi-Judge: {self.use_multi_judge}")
        summary_lines.append(f"Max Iterations: {self.max_iterations}")
        
        summary_lines.append(f"\n{'-'*80}")
        summary_lines.append("PROCESSING STATISTICS")
        summary_lines.append(f"{'-'*80}")
        summary_lines.append(f"Total Algorithms: {len(df)}")
        summary_lines.append(f"Completed: {(df['Processing_Status'] == 'COMPLETED').sum()}")
        summary_lines.append(f"Failed: {(df['Processing_Status'] == 'FAILED').sum()}")
        summary_lines.append(f"Total Processing Time: {df['Processing_Time_Seconds'].sum():.2f} seconds")
        summary_lines.append(f"Average Time per Algorithm: {df['Processing_Time_Seconds'].mean():.2f} seconds")
        
        completed_df = df[df['Processing_Status'] == 'COMPLETED']
        
        if len(completed_df) > 0:
            summary_lines.append(f"\n{'-'*80}")
            summary_lines.append("VERDICT DISTRIBUTION")
            summary_lines.append(f"{'-'*80}")
            verdict_counts = completed_df['V2_Verdict'].value_counts()
            for verdict, count in verdict_counts.items():
                pct = count / len(completed_df) * 100
                summary_lines.append(f"{verdict:15s}: {count:3d} ({pct:5.1f}%)")
            
            summary_lines.append(f"\n{'-'*80}")
            summary_lines.append("CONVERGENCE STATISTICS")
            summary_lines.append(f"{'-'*80}")
            converged = completed_df['Converged'].sum()
            summary_lines.append(f"Converged (PASS/PASS_MINOR): {converged} ({converged/len(completed_df)*100:.1f}%)")
            summary_lines.append(f"Average Iterations: {completed_df['Total_Iterations'].mean():.2f}")
            
            summary_lines.append(f"\n{'-'*80}")
            summary_lines.append("CORRECTION METRICS (CRS)")
            summary_lines.append(f"{'-'*80}")
            summary_lines.append(f"Mean CRS: {completed_df['CRS_Final'].mean():.3f}")
            summary_lines.append(f"Mean ERR: {completed_df['ERR_Final'].mean():.3f}")
            summary_lines.append(f"Mean RP: {completed_df['RP_Final'].mean():.3f}")
            summary_lines.append(f"Mean TFP: {completed_df['TFP_Final'].mean():.3f}")
            
            crs_strong = (completed_df['CRS_Final'] >= 0.7).sum()
            crs_partial = ((completed_df['CRS_Final'] >= 0.4) & (completed_df['CRS_Final'] < 0.7)).sum()
            crs_weak = (completed_df['CRS_Final'] < 0.4).sum()
            
            summary_lines.append(f"\nCRS Distribution:")
            summary_lines.append(f"  Strong (≥0.7):     {crs_strong:3d} ({crs_strong/len(completed_df)*100:5.1f}%)")
            summary_lines.append(f"  Partial (0.4-0.7): {crs_partial:3d} ({crs_partial/len(completed_df)*100:5.1f}%)")
            summary_lines.append(f"  Weak (<0.4):       {crs_weak:3d} ({crs_weak/len(completed_df)*100:5.1f}%)")
            
            summary_lines.append(f"\n{'-'*80}")
            summary_lines.append("ERROR RESOLUTION")
            summary_lines.append(f"{'-'*80}")
            summary_lines.append(f"Total Errors Fixed: {completed_df['Errors_Fixed'].sum()}")
            summary_lines.append(f"Total Errors Introduced: {completed_df['Errors_Introduced'].sum()}")
            summary_lines.append(f"Net Improvement: {completed_df['Net_Improvement'].sum()}")
            summary_lines.append(f"Average Net Improvement per Algorithm: {completed_df['Net_Improvement'].mean():.2f}")
            
            if self.use_multi_judge:
                summary_lines.append(f"\n{'-'*80}")
                summary_lines.append("MULTI-JUDGE STATISTICS")
                summary_lines.append(f"{'-'*80}")
                summary_lines.append(f"Number of Judges: {completed_df['Num_Judges'].max()}")
                summary_lines.append(f"Mean Judge Reliability (JRS): {completed_df['Mean_JRS'].mean():.3f}")
                summary_lines.append(f"Mean Weighted CFRS: {completed_df['Weighted_CFRS'].mean():.3f}")
            
            summary_lines.append(f"\n{'-'*80}")
            summary_lines.append("TOP 5 ALGORITHMS BY CRS")
            summary_lines.append(f"{'-'*80}")
            top5 = completed_df.nlargest(5, 'CRS_Final')
            for i, (idx, row) in enumerate(top5.iterrows(), 1):
                algo = row['Problem Statement'][:50] + "..."
                summary_lines.append(f"{i}. CRS={row['CRS_Final']:.3f}, Score={row['Overall_Score']}, Iter={row['Total_Iterations']}")
                summary_lines.append(f"   {algo}")
            
            summary_lines.append(f"\n{'-'*80}")
            summary_lines.append("BOTTOM 5 ALGORITHMS BY CRS")
            summary_lines.append(f"{'-'*80}")
            bottom5 = completed_df.nsmallest(5, 'CRS_Final')
            for i, (idx, row) in enumerate(bottom5.iterrows(), 1):
                algo = row['Problem Statement'][:50] + "..."
                summary_lines.append(f"{i}. CRS={row['CRS_Final']:.3f}, Score={row['Overall_Score']}, Iter={row['Total_Iterations']}")
                summary_lines.append(f"   {algo}")
        
        summary_lines.append(f"\n{'='*80}")
        summary_lines.append("END OF SUMMARY")
        summary_lines.append(f"{'='*80}\n")
        
        # Write to file
        summary_text = "\n".join(summary_lines)
        with open(self.summary_txt, "w", encoding="utf-8") as f:
            f.write(summary_text)
        
        # Also print to console
        print("\n" + summary_text)


def main():
    parser = argparse.ArgumentParser(description="Batch process convergence proofs")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", default="batch_results", help="Output directory")
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="Prover model")
    parser.add_argument("--judges", nargs="+", help="List of judge models")
    parser.add_argument("--multi-judge", action="store_true", help="Use multi-judge")
    parser.add_argument("--max-iter", type=int, default=3, help="Max iterations")
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    
    # Setup evaluator models
    evaluator_models = args.judges if args.judges else [args.model]
    
    # Run batch processor
    processor = BatchProcessor(
        input_csv=args.input,
        output_dir=args.output,
        prover_model=args.model,
        evaluator_models=evaluator_models,
        use_multi_judge=args.multi_judge,
        max_iterations=args.max_iter
    )
    
    processor.process()


if __name__ == "__main__":
    main()