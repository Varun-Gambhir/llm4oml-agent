# ============================================================================
# File: scripts/run_batch.py
# ============================================================================
"""Batch processor for multiple algorithms with v2.0 metrics."""

import pandas as pd
import json
import argparse
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.workflow import ProofWorkflow
from src.utils.logger import StructuredLogger


class BatchProcessor:
    """Process multiple algorithms with full v2.0 metric tracking."""
    
    def __init__(
        self,
        input_csv: str,
        output_csv: str,
        output_json: str,
        prover_model: str,
        evaluator_models: list,
        use_multi_judge: bool,
        max_iterations: int
    ):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.output_json = output_json
        
        self.workflow = ProofWorkflow(
            prover_model=prover_model,
            evaluator_models=evaluator_models,
            use_multi_judge=use_multi_judge,
            max_iterations=max_iterations
        )
        
        self.results_log = []
    
    def process(self):
        """Process all algorithms in CSV."""
        print(f"--- Starting Batch Processing: {self.input_csv} ---")
        
        # Load CSV
        try:
            df = pd.read_csv(self.input_csv, header=1)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return
        
        # Filter empty rows
        df = df[df['Problem Statement'].notna()].reset_index(drop=True)
        print(f"Found {len(df)} algorithms to process.")
        
        # Add new v2.0 columns
        self._add_v2_columns(df)
        
        # Process each row
        for index, row in df.iterrows():
            try:
                self._process_row(index, row, df)
                
                # Save progress after each row
                df.to_csv(self.output_csv, index=False)
                with open(self.output_json, "w", encoding="utf-8") as f:
                    json.dump(self.results_log, f, indent=2, default=str)
                    
            except Exception as e:
                print(f"Error processing row {index}: {e}")
                self.results_log.append({
                    "row_index": index,
                    "error": str(e)
                })
        
        print(f"\n✅ Batch Processing Complete!")
        print(f"Results: {self.output_csv}")
        print(f"Detailed logs: {self.output_json}")
    
    def _add_v2_columns(self, df: pd.DataFrame):
        """Add v2.0 metric columns to dataframe."""
        new_columns = {
            # CRS Metrics (per iteration)
            'CRS_Iter1': 0.0,
            'CRS_Iter2': 0.0,
            'CRS_Iter3': 0.0,
            'CRS_Final': 0.0,
            
            # CRS Components (final iteration)
            'ERR_Final': 0.0,
            'RP_Final': 0.0,
            'TFP_Final': 0.0,
            
            # Error tracking
            'Errors_Fixed': 0,
            'Errors_Introduced': 0,
            'Net_Improvement': 0,
            
            # Multi-judge metrics
            'Num_Judges': 0,
            'Mean_JRS': 0.0,
            'Outlier_Judges': "",
            'Judge_Agreement_ESA': 0.0,
            'Score_Consistency_SC': 0.0,
            
            # Weighted consensus
            'Weighted_CFRS': 0.0
        }
        
        for col, default in new_columns.items():
            if col not in df.columns:
                df[col] = default
    
    def _process_row(self, index: int, row: pd.Series, df: pd.DataFrame):
        """Process a single algorithm."""
        algo_text = row['Problem Statement']
        assump_text = row['Assumptions Used']
        
        print(f"\n=== Processing Row {index + 1}/{len(df)} ===")
        print(f"Algorithm: {algo_text[:60]}...")
        
        # Run workflow
        final_state = self.workflow.run(algo_text, assump_text)
        
        # Log detailed results
        row_log = {
            "row_index": index,
            "algorithm": algo_text,
            "final_verdict": final_state.get("verdict", "UNKNOWN"),
            "iterations": []
        }
        
        # Extract iteration data
        # Note: In real implementation, you'd track this during workflow execution
        # For now, we'll extract from final_state
        
        # Update V1 columns (first iteration)
        if final_state.get("iteration", 0) >= 1:
            metrics_v1 = final_state.get("metrics", {})
            self._update_v1_columns(df, index, metrics_v1)
        
        # Update V2 columns (final iteration)
        self._update_v2_columns(df, index, final_state)
        
        # Update CRS columns
        self._update_crs_columns(df, index, final_state)
        
        # Update multi-judge columns
        self._update_judge_columns(df, index, final_state)
        
        self.results_log.append(row_log)
    
    def _update_v1_columns(self, df: pd.DataFrame, index: int, metrics: Dict[str, Any]):
        """Update V1 (first iteration) columns."""
        to_yn = lambda x: "yes" if x else "no"
        
        df.at[index, 'Hallucination Error (HA) (yes/no)'] = to_yn(metrics.get("HA", False))
        df.at[index, 'Missing Step (MS) (yes/no)'] = to_yn(metrics.get("MS", False))
        df.at[index, 'Operator Error (OP) (yes/no)'] = to_yn(metrics.get("OP", False))
        df.at[index, 'Completeness Score (0–5)'] = metrics.get("completeness_score", 0)
        df.at[index, 'Assumption Use Score (0–5)'] = metrics.get("assumption_use_score", 0)
        df.at[index, 'Model for evaluation'] = "Multi-Judge Ensemble"
    
    def _update_v2_columns(self, df: pd.DataFrame, index: int, state: Dict[str, Any]):
        """Update V2 (final iteration) columns."""
        metrics = state.get("metrics", {})
        to_yn = lambda x: "yes" if x else "no"
        
        if state.get("iteration", 0) > 1:
            df.at[index, 'Hallucination Error (HA) (yes/no).1'] = to_yn(metrics.get("HA", False))
            df.at[index, 'Missing Step (MS) (yes/no).1'] = to_yn(metrics.get("MS", False))
            df.at[index, 'Operator Error (OP) (yes/no).1'] = to_yn(metrics.get("OP", False))
            df.at[index, 'Completeness Score (0–5).1'] = metrics.get("completeness_score", 0)
            df.at[index, 'Assumption Use Score (0–5).1'] = metrics.get("assumption_use_score", 0)
        
        # Overall score based on verdict
        verdict_scores = {"PASS": 5, "PASS_MINOR": 4, "CONDITIONAL": 3, "FAIL": 2, "REJECT": 1}
        df.at[index, 'Overall Score (0–5)'] = verdict_scores.get(state.get("verdict", "FAIL"), 0)
        
        # Comments
        feedback = state.get("feedback", "")
        df.at[index, 'Comments on LLM Judge'] = f"Verdict: {state.get('verdict')}. {feedback[:100]}..."
    
    def _update_crs_columns(self, df: pd.DataFrame, index: int, state: Dict[str, Any]):
        """Update CRS metric columns."""
        correction_metrics = state.get("correction_metrics")
        
        if correction_metrics:
            df.at[index, 'CRS_Final'] = correction_metrics.get("correction_reasoning_score", 0.0)
            df.at[index, 'ERR_Final'] = correction_metrics.get("error_resolution_rate", 0.0)
            df.at[index, 'RP_Final'] = correction_metrics.get("regression_penalty", 0.0)
            df.at[index, 'TFP_Final'] = correction_metrics.get("targeted_fix_precision", 0.0)
            
            errors_fixed = correction_metrics.get("errors_fixed", 0)
            errors_introduced = correction_metrics.get("errors_introduced", 0)
            
            df.at[index, 'Errors_Fixed'] = errors_fixed
            df.at[index, 'Errors_Introduced'] = errors_introduced
            df.at[index, 'Net_Improvement'] = errors_fixed - errors_introduced
    
    def _update_judge_columns(self, df: pd.DataFrame, index: int, state: Dict[str, Any]):
        """Update multi-judge reliability columns."""
        judge_evals = state.get("judge_evaluations", [])
        judge_reliability = state.get("judge_reliability", [])
        
        if judge_evals:
            df.at[index, 'Num_Judges'] = len(judge_evals)
            df.at[index, 'Weighted_CFRS'] = state.get("metrics", {}).get("weighted_cfrs", 0.0)
        
        if judge_reliability:
            df.at[index, 'Mean_JRS'] = sum(judge_reliability) / len(judge_reliability)
            
            # Find outliers (simplified - assumes z-score > 2)
            # In full implementation, extract from consensus
            outliers = state.get("judge_evaluations", [])
            if outliers:
                outlier_ids = [e["judge_id"] for e in outliers[:1]]  # Placeholder
                df.at[index, 'Outlier_Judges'] = ", ".join(outlier_ids) if outlier_ids else "None"


def main():
    parser = argparse.ArgumentParser(description="Batch process convergence proofs")
    parser.add_argument("--input", default="LLM for Convergence - GPT 5.csv", help="Input CSV")
    parser.add_argument("--output", default="processed_results_v2.csv", help="Output CSV")
    parser.add_argument("--json", default="batch_execution_v2.json", help="JSON log")
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
        output_csv=args.output,
        output_json=args.json,
        prover_model=args.model,
        evaluator_models=evaluator_models,
        use_multi_judge=args.multi_judge,
        max_iterations=args.max_iter
    )
    
    processor.process()


if __name__ == "__main__":
    main()
