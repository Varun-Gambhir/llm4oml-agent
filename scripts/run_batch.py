# ============================================================================
# File: scripts/run_batch.py (COMPLETE WITH BATCHPROCESSOR CLASS)
# ============================================================================
"""Batch processor with complete state tracking, CSV generation, and provider support."""

import pandas as pd
import json
import argparse
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
import sys
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.workflow import ProofWorkflow


class BatchProcessor:
    """Process multiple algorithms with full tracking and CSV output."""
    
    def __init__(
        self,
        input_csv: str,
        output_dir: str,
        provider_name: str,
        prover_model: str,
        evaluator_models: list,
        api_key: str = None,
        use_multi_judge: bool = False,
        max_iterations: int = 3,
        temperature: float = 0.5,
        timeout: int = 600,
        max_retries: int = 3
    ):
        self.input_csv = input_csv
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Configuration
        self.provider_name = provider_name
        self.prover_model = prover_model
        self.evaluator_models = evaluator_models
        self.api_key = api_key
        self.use_multi_judge = use_multi_judge
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Output files
        self.output_csv = self.output_dir / "results.csv"
        self.output_json = self.output_dir / "batch_execution_log.json"
        self.summary_txt = self.output_dir / "summary.txt"
        
        # Batch-level log
        self.batch_log = {
            "metadata": {
                "start_time": datetime.now().isoformat(),
                "input_file": str(input_csv),
                "configuration": {
                    "provider": provider_name,
                    "prover_model": prover_model,
                    "evaluator_models": evaluator_models,
                    "use_multi_judge": use_multi_judge,
                    "max_iterations": max_iterations,
                    "temperature": temperature,
                    "timeout": timeout,
                    "max_retries": max_retries
                }
            },
            "algorithms": []
        }
    
    def process(self):
        """Process all algorithms in CSV."""
        print(f"\n{'='*80}")
        print(f"BATCH PROCESSING STARTED")
        print(f"{'='*80}")
        print(f"Input: {self.input_csv}")
        print(f"Output: {self.output_dir}")
        print(f"Provider: {self.provider_name}")
        print(f"Model: {self.prover_model}")
        print(f"Multi-Judge: {self.use_multi_judge}")
        print(f"Judges: {self.evaluator_models}")
        print(f"Max Iterations: {self.max_iterations}")
        print(f"Timeout: {self.timeout}s")
        print(f"Max Retries: {self.max_retries}")
        print(f"{'='*80}\n")
        
        # Load CSV
        try:
            # Try reading with header in row 1 (index 1)
            df = pd.read_csv(self.input_csv)
        except Exception as e:
            print(f"❌ Error reading CSV with header=1, trying header=0: {e}")
            try:
                df = pd.read_csv(self.input_csv, header=0)
            except Exception as e2:
                print(f"❌ Error reading CSV: {e2}")
                return
        
        # Filter empty rows
        df = df[df['Problem Statement'].notna()].reset_index(drop=True)
        total_algos = len(df)
        print(f"📊 Found {total_algos} algorithms to process.\n")
        
        # Prepare output dataframe
        results_df = self._prepare_results_dataframe(df)
        
        # Process each row
        successful = 0
        failed = 0
        
        for index, row in df.iterrows():
            print(f"\n{'#'*80}")
            print(f"# Algorithm {index + 1}/{total_algos}")
            print(f"{'#'*80}")
            
            try:
                self._process_single_algorithm(index, row, results_df)
                successful += 1
                
                # Save progress after each algorithm
                results_df.to_csv(self.output_csv, index=False)
                self._save_batch_log()
                
                print(f"\n✅ Completed {index + 1}/{total_algos} (Success: {successful}, Failed: {failed})")
                
            except Exception as e:
                failed += 1
                print(f"\n❌ Error processing row {index}: {e}")
                import traceback
                traceback.print_exc()
                
                self.batch_log["algorithms"].append({
                    "row_index": index,
                    "error": str(e),
                    "status": "failed"
                })
                
                # Mark as failed in CSV
                results_df.at[index, 'Processing_Status'] = 'FAILED'
                results_df.at[index, 'Error_Message'] = str(e)[:500]  # Truncate long errors
                results_df.to_csv(self.output_csv, index=False)
                self._save_batch_log()
        
        # Finalize
        self._finalize_batch(results_df)
        
        print(f"\n{'='*80}")
        print(f"✅ BATCH PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"Total: {total_algos} | Success: {successful} | Failed: {failed}")
        print(f"\nOutput Files:")
        print(f"  📊 Results CSV: {self.output_csv}")
        print(f"  📄 Summary Report: {self.summary_txt}")
        print(f"  📝 Execution Log: {self.output_json}")
        print(f"{'='*80}\n")
    
    def _prepare_results_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare results dataframe with all necessary columns."""
        results_df = df.copy()
        
        # Add all V2.0 columns
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
            'Initial_Proof_Length': 0,
            'Final_Proof_Length': 0,
            'Proof_Length_Growth': 0,
            
            # Feedback Summary
            'Final_Feedback_Preview': '',
            
            # Model Information
            'Provider_Used': '',
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
        
        print(f"\nAlgorithm: {algo_text[:70]}...")
        print(f"Assumptions: {assump_text[:70]}...")
        
        # Create algorithm-specific output directory
        algo_dir = self.output_dir / f"algorithm_{index:03d}"
        algo_dir.mkdir(exist_ok=True)
        
        # Track processing time
        start_time = time.time()
        
        # Run workflow
        workflow = ProofWorkflow(
            provider_name=self.provider_name,
            prover_model=self.prover_model,
            evaluator_models=self.evaluator_models,
            api_key=self.api_key,
            use_multi_judge=self.use_multi_judge,
            max_iterations=self.max_iterations,
            temperature=self.temperature,
            timeout=self.timeout,
            max_retries=self.max_retries,
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
            "crs_final": exec_log.get("statistics", {}).get("crs_progression", [None])[-1] if exec_log.get("statistics", {}).get("crs_progression") else None
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
        df.at[index, 'Provider_Used'] = self.provider_name
        df.at[index, 'Model_Used'] = f"{self.prover_model} (Judges: {len(self.evaluator_models)})"
        
        # Iteration Information
        iterations = exec_log.get("iterations", [])
        df.at[index, 'Total_Iterations'] = len(iterations)
        df.at[index, 'Converged'] = exec_log.get("final_results", {}).get("convergence_achieved", False)
        
        # V1 Metrics (First Iteration)
        if len(iterations) >= 1 and "evaluator" in iterations[0].get("nodes", {}):
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
            total_v1_errors = sum(
                len(v1_error_set.get(k, []) if isinstance(v1_error_set.get(k, []), list) else v1_error_set.get(k, set()))
                for k in ["hallucinations", "missing_steps", "operator_errors", "assumption_violations"]
            )
            df.at[index, 'Total_Errors_V1'] = total_v1_errors
        
        # Get initial proof length
        if len(iterations) >= 1 and "prover" in iterations[0].get("nodes", {}):
            df.at[index, 'Initial_Proof_Length'] = iterations[0]["nodes"]["prover"].get("proof_length", 0)
        
        # V2 Metrics (Final Iteration)
        if len(iterations) >= 1:
            final_iter = iterations[-1]
            
            if "evaluator" in final_iter.get("nodes", {}):
                vf_eval = final_iter["nodes"]["evaluator"]
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
                total_vf_errors = sum(
                    len(vf_error_set.get(k, []) if isinstance(vf_error_set.get(k, []), list) else vf_error_set.get(k, set()))
                    for k in ["hallucinations", "missing_steps", "operator_errors", "assumption_violations"]
                )
                df.at[index, 'Total_Errors_Final'] = total_vf_errors
                
                # Feedback preview
                feedback = vf_eval.get("feedback", "")
                df.at[index, 'Final_Feedback_Preview'] = feedback[:200] + "..." if len(feedback) > 200 else feedback
                
                # Multi-judge metrics
                judge_evals = vf_eval.get("judge_evaluations", [])
                judge_reliability = vf_eval.get("judge_reliability", [])
                
                if judge_evals:
                    df.at[index, 'Num_Judges'] = len(judge_evals)
                    df.at[index, 'Weighted_CFRS'] = round(vf_metrics.get("weighted_cfrs", 0.0), 3)
                
                if judge_reliability:
                    df.at[index, 'Mean_JRS'] = round(sum(judge_reliability) / len(judge_reliability), 3)
            
            # Get final proof length
            if "prover" in final_iter.get("nodes", {}):
                df.at[index, 'Final_Proof_Length'] = final_iter["nodes"]["prover"].get("proof_length", 0)
        
        # Proof length growth
        df.at[index, 'Proof_Length_Growth'] = df.at[index, 'Final_Proof_Length'] - df.at[index, 'Initial_Proof_Length']
        
        # CRS Metrics (for each iteration that has them)
        for i, iteration in enumerate(iterations):
            if "evaluator" in iteration.get("nodes", {}):
                corr_metrics = iteration["nodes"]["evaluator"].get("correction_metrics")
                if corr_metrics:
                    crs_value = corr_metrics.get("correction_reasoning_score", 0.0)
                    
                    # Store per-iteration CRS
                    if i < 3:
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
    
    def _save_batch_log(self):
        """Save batch execution log."""
        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(self.batch_log, f, indent=2, default=str)
    
    def _finalize_batch(self, results_df: pd.DataFrame):
        """Finalize batch processing with summary."""
        self.batch_log["metadata"]["end_time"] = datetime.now().isoformat()
        self.batch_log["metadata"]["total_algorithms"] = len(results_df)
        self.batch_log["metadata"]["completed"] = int((results_df['Processing_Status'] == 'COMPLETED').sum())
        self.batch_log["metadata"]["failed"] = int((results_df['Processing_Status'] == 'FAILED').sum())
        
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
        summary_lines.append(f"Provider: {self.provider_name}")
        summary_lines.append(f"Prover Model: {self.prover_model}")
        summary_lines.append(f"Evaluator Models: {', '.join(self.evaluator_models)}")
        summary_lines.append(f"Multi-Judge: {self.use_multi_judge}")
        summary_lines.append(f"Max Iterations: {self.max_iterations}")
        summary_lines.append(f"Timeout: {self.timeout}s")
        summary_lines.append(f"Max Retries: {self.max_retries}")
        
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
            
            if self.use_multi_judge and completed_df['Num_Judges'].max() > 1:
                summary_lines.append(f"\n{'-'*80}")
                summary_lines.append("MULTI-JUDGE STATISTICS")
                summary_lines.append(f"{'-'*80}")
                summary_lines.append(f"Number of Judges: {int(completed_df['Num_Judges'].max())}")
                summary_lines.append(f"Mean Judge Reliability (JRS): {completed_df['Mean_JRS'].mean():.3f}")
                summary_lines.append(f"Mean Weighted CFRS: {completed_df['Weighted_CFRS'].mean():.3f}")
            
            summary_lines.append(f"\n{'-'*80}")
            summary_lines.append("TOP 5 ALGORITHMS BY CRS")
            summary_lines.append(f"{'-'*80}")
            top5 = completed_df.nlargest(5, 'CRS_Final')
            for i, (idx, row) in enumerate(top5.iterrows(), 1):
                algo = str(row['Problem Statement'])[:50] + "..."
                summary_lines.append(f"{i}. CRS={row['CRS_Final']:.3f}, Score={row['Overall_Score']}, Iter={row['Total_Iterations']}")
                summary_lines.append(f"   {algo}")
            
            if len(completed_df) >= 5:
                summary_lines.append(f"\n{'-'*80}")
                summary_lines.append("BOTTOM 5 ALGORITHMS BY CRS")
                summary_lines.append(f"{'-'*80}")
                bottom5 = completed_df.nsmallest(5, 'CRS_Final')
                for i, (idx, row) in enumerate(bottom5.iterrows(), 1):
                    algo = str(row['Problem Statement'])[:50] + "..."
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
    
    # Provider configuration
    parser.add_argument("--provider", default="nvidia",
                       choices=["nvidia", "openrouter", "openai", "anthropic"],
                       help="LLM provider to use")
    parser.add_argument("--api-key", help="API key (overrides environment variable)")
    
    # Model configuration
    parser.add_argument("--model", help="Prover model")
    parser.add_argument("--judges", nargs="+", help="List of judge models")
    parser.add_argument("--multi-judge", action="store_true", help="Use multi-judge")
    
    # Generation parameters
    parser.add_argument("--max-iter", type=int, default=3, help="Max iterations")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retry attempts")
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    
    # Get default model for provider if not specified
    if args.model is None:
        default_models = {
            "nvidia": "openai/gpt-oss-120b",
            "openrouter": "anthropic/claude-3.5-sonnet",
            "openai": "gpt-4",
            "anthropic": "claude-3-5-sonnet-20241022"
        }
        args.model = default_models.get(args.provider)
        print(f"Using default model for {args.provider}: {args.model}")
    
    # Setup evaluator models
    evaluator_models = args.judges if args.judges else [args.model]
    
    # Run batch processor
    processor = BatchProcessor(
        input_csv=args.input,
        output_dir=args.output,
        provider_name=args.provider,
        prover_model=args.model,
        evaluator_models=evaluator_models,
        api_key=args.api_key,
        use_multi_judge=args.multi_judge,
        max_iterations=args.max_iter,
        temperature=args.temperature,
        timeout=args.timeout,
        max_retries=args.max_retries
    )
    
    processor.process()


if __name__ == "__main__":
    main()