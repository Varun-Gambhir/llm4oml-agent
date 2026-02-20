# ============================================================================
# File: scripts/run_batch.py
# ============================================================================
"""Batch processor with full state tracking, CSV generation, and provider support."""

import pandas as pd
import json
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.workflow import ProofWorkflow
from src.config.temperatures import TemperatureConfig


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
        timeout: int = 600,
        max_retries: int = 3,
        temp_config: TemperatureConfig = None,
    ):
        self.input_csv = input_csv
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.provider_name = provider_name
        self.prover_model = prover_model
        self.evaluator_models = evaluator_models
        self.api_key = api_key
        self.use_multi_judge = use_multi_judge
        self.max_iterations = max_iterations
        self.timeout = timeout
        self.max_retries = max_retries
        self.temp_config = temp_config or TemperatureConfig()

        self.output_csv = self.output_dir / "results.csv"
        self.output_json = self.output_dir / "batch_execution_log.json"
        self.summary_txt = self.output_dir / "summary.txt"

        self.batch_log: Dict[str, Any] = {
            "metadata": {
                "start_time": datetime.now().isoformat(),
                "input_file": str(input_csv),
                "configuration": {
                    "provider": provider_name,
                    "prover_model": prover_model,
                    "evaluator_models": evaluator_models,
                    "use_multi_judge": use_multi_judge,
                    "max_iterations": max_iterations,
                    "timeout": timeout,
                    "max_retries": max_retries,
                    "temperatures": {
                        "generation": self.temp_config.generation,
                        "correction": self.temp_config.correction,
                        "evaluation": self.temp_config.evaluation,
                        "verification": self.temp_config.verification,
                    },
                },
            },
            "algorithms": [],
        }

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def process(self):
        print(f"\n{'='*80}")
        print("BATCH PROCESSING STARTED")
        print(f"{'='*80}")
        print(f"Input           : {self.input_csv}")
        print(f"Output          : {self.output_dir}")
        print(f"Provider        : {self.provider_name}")
        print(f"Model           : {self.prover_model}")
        print(f"Multi-Judge     : {self.use_multi_judge}")
        print(f"Max Iterations  : {self.max_iterations}")
        print(f"Temperatures    : gen={self.temp_config.generation}, "
              f"corr={self.temp_config.correction}, "
              f"eval={self.temp_config.evaluation}")
        print(f"{'='*80}\n")

        try:
            df = pd.read_csv(self.input_csv)
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            return

        df = df[df["Problem Statement"].notna()].reset_index(drop=True)
        total = len(df)
        print(f"📊 Found {total} algorithms to process.\n")

        results_df = self._prepare_results_dataframe(df)
        successful = failed = 0

        for index, row in df.iterrows():
            print(f"\n{'#'*80}")
            print(f"# Algorithm {index + 1}/{total}")
            print(f"{'#'*80}")
            try:
                self._process_single_algorithm(index, row, results_df)
                successful += 1
            except Exception as exc:
                failed += 1
                print(f"❌ Error processing row {index}: {exc}")
                import traceback; traceback.print_exc()
                self.batch_log["algorithms"].append({
                    "row_index": index, "error": str(exc), "status": "failed"
                })
                results_df.at[index, "Processing_Status"] = "FAILED"
                results_df.at[index, "Error_Message"] = str(exc)[:500]
            finally:
                results_df.to_csv(self.output_csv, index=False)
                self._save_batch_log()
                print(f"✅ Progress: {index+1}/{total} (ok={successful}, fail={failed})")

        self._finalize_batch(results_df)

        print(f"\n{'='*80}")
        print("✅ BATCH PROCESSING COMPLETE")
        print(f"   Total={total}  Success={successful}  Failed={failed}")
        print(f"   CSV     : {self.output_csv}")
        print(f"   Summary : {self.summary_txt}")
        print(f"   Log     : {self.output_json}")
        print(f"{'='*80}\n")

    # ------------------------------------------------------------------
    # Per-algorithm processing
    # ------------------------------------------------------------------

    def _process_single_algorithm(
        self, index: int, row: pd.Series, results_df: pd.DataFrame
    ):
        algo_text = str(row["Problem Statement"])
        assump_text = str(row.get("Assumptions Used", ""))

        print(f"Algorithm   : {algo_text[:70]}...")
        print(f"Assumptions : {assump_text[:70]}...")

        algo_dir = self.output_dir / f"algorithm_{index:03d}"
        start = time.time()

        workflow = ProofWorkflow(
            provider_name=self.provider_name,
            prover_model=self.prover_model,
            evaluator_models=self.evaluator_models,
            api_key=self.api_key,
            use_multi_judge=self.use_multi_judge,
            max_iterations=self.max_iterations,
            timeout=self.timeout,
            max_retries=self.max_retries,
            output_dir=str(algo_dir),
            temp_config=self.temp_config,
        )

        final_state, tracker = workflow.run(algo_text, assump_text)
        elapsed = time.time() - start
        exec_log = tracker.get_log()

        self._update_results_row(index, results_df, exec_log, final_state, elapsed)

        self.batch_log["algorithms"].append({
            "row_index": index,
            "algorithm": algo_text,
            "status": "completed",
            "processing_time": elapsed,
            "final_verdict": final_state.get("verdict") if final_state else "UNKNOWN",
            "iterations": exec_log.get("final_results", {}).get("total_iterations", 0),
            "output_directory": str(algo_dir),
            "crs_final": (exec_log.get("statistics", {}).get("crs_progression") or [None])[-1],
        })

    def _update_results_row(
        self,
        index: int,
        df: pd.DataFrame,
        exec_log: Dict,
        final_state: Dict,
        processing_time: float,
    ):
        to_yn = lambda x: "yes" if x else "no"

        df.at[index, "Processing_Status"] = "COMPLETED"
        df.at[index, "Processing_Time_Seconds"] = round(processing_time, 2)
        df.at[index, "Provider_Used"] = self.provider_name
        df.at[index, "Model_Used"] = self.prover_model

        iterations = exec_log.get("iterations", [])
        df.at[index, "Total_Iterations"] = len(iterations)
        df.at[index, "Converged"] = exec_log.get("final_results", {}).get(
            "convergence_achieved", False
        )

        # V1 metrics
        if iterations and "evaluator" in iterations[0].get("nodes", {}):
            ev1 = iterations[0]["nodes"]["evaluator"]
            m1 = ev1.get("metrics", {})
            df.at[index, "V1_Hallucination_Error"] = to_yn(m1.get("HA", False))
            df.at[index, "V1_Missing_Step"] = to_yn(m1.get("MS", False))
            df.at[index, "V1_Operator_Error"] = to_yn(m1.get("OP", False))
            df.at[index, "V1_Completeness_Score"] = m1.get("completeness_score", 0)
            df.at[index, "V1_Assumption_Score"] = m1.get("assumption_use_score", 0)
            df.at[index, "V1_Verdict"] = ev1.get("verdict", "")
            es1 = ev1.get("error_set", {})
            df.at[index, "Total_Errors_V1"] = sum(
                len(v) for v in es1.values() if isinstance(v, list)
            )

        if iterations and "prover" in iterations[0].get("nodes", {}):
            df.at[index, "Initial_Proof_Length"] = iterations[0]["nodes"]["prover"].get(
                "proof_length", 0
            )

        # V2 (final) metrics
        if iterations:
            evf = iterations[-1].get("nodes", {}).get("evaluator", {})
            mf = evf.get("metrics", {})
            df.at[index, "V2_Hallucination_Error"] = to_yn(mf.get("HA", False))
            df.at[index, "V2_Missing_Step"] = to_yn(mf.get("MS", False))
            df.at[index, "V2_Operator_Error"] = to_yn(mf.get("OP", False))
            df.at[index, "V2_Completeness_Score"] = mf.get("completeness_score", 0)
            df.at[index, "V2_Assumption_Score"] = mf.get("assumption_use_score", 0)
            df.at[index, "V2_Verdict"] = evf.get("verdict", "")

            verdict_scores = {"PASS": 5, "PASS_MINOR": 4, "CONDITIONAL": 3, "FAIL": 2, "REJECT": 1}
            df.at[index, "Overall_Score"] = verdict_scores.get(evf.get("verdict", "FAIL"), 0)

            esf = evf.get("error_set", {})
            df.at[index, "Total_Errors_Final"] = sum(
                len(v) for v in esf.values() if isinstance(v, list)
            )

            feedback = evf.get("feedback", "")
            df.at[index, "Final_Feedback_Preview"] = (
                feedback[:200] + "..." if len(feedback) > 200 else feedback
            )

            judge_evals = evf.get("judge_evaluations", [])
            judge_rel = evf.get("judge_reliability", [])
            if judge_evals:
                df.at[index, "Num_Judges"] = len(judge_evals)
                df.at[index, "Weighted_CFRS"] = round(mf.get("weighted_cfrs", 0.0), 3)
            if judge_rel:
                df.at[index, "Mean_JRS"] = round(sum(judge_rel) / len(judge_rel), 3)

            if "prover" in iterations[-1].get("nodes", {}):
                df.at[index, "Final_Proof_Length"] = iterations[-1]["nodes"]["prover"].get(
                    "proof_length", 0
                )

        df.at[index, "Proof_Length_Growth"] = (
            df.at[index, "Final_Proof_Length"] - df.at[index, "Initial_Proof_Length"]
        )

        # CRS per iteration
        for i, it in enumerate(iterations):
            ev = it.get("nodes", {}).get("evaluator", {})
            cm = ev.get("correction_metrics")
            if cm:
                crs_val = cm.get("correction_reasoning_score", 0.0)
                if i < 3:
                    df.at[index, f"CRS_Iter_{i+1}"] = round(crs_val, 3)
                if i == len(iterations) - 1:
                    df.at[index, "CRS_Final"] = round(crs_val, 3)
                    df.at[index, "ERR_Final"] = round(cm.get("error_resolution_rate", 0.0), 3)
                    df.at[index, "RP_Final"] = round(cm.get("regression_penalty", 0.0), 3)
                    df.at[index, "TFP_Final"] = round(cm.get("targeted_fix_precision", 0.0), 3)
                    df.at[index, "Errors_Fixed"] = cm.get("errors_fixed", 0)
                    df.at[index, "Errors_Introduced"] = cm.get("errors_introduced", 0)

        df.at[index, "Net_Improvement"] = (
            df.at[index, "Errors_Fixed"] - df.at[index, "Errors_Introduced"]
        )

    def _prepare_results_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        defaults = {
            "Processing_Status": "PENDING", "Error_Message": "",
            "Processing_Time_Seconds": 0.0,
            "V1_Hallucination_Error": "", "V1_Missing_Step": "", "V1_Operator_Error": "",
            "V1_Completeness_Score": 0, "V1_Assumption_Score": 0, "V1_Verdict": "",
            "V2_Hallucination_Error": "", "V2_Missing_Step": "", "V2_Operator_Error": "",
            "V2_Completeness_Score": 0, "V2_Assumption_Score": 0, "V2_Verdict": "",
            "Overall_Score": 0,
            "Total_Iterations": 0, "Converged": False,
            "CRS_Iter_1": 0.0, "CRS_Iter_2": 0.0, "CRS_Iter_3": 0.0, "CRS_Final": 0.0,
            "ERR_Final": 0.0, "RP_Final": 0.0, "TFP_Final": 0.0,
            "Total_Errors_V1": 0, "Total_Errors_Final": 0,
            "Errors_Fixed": 0, "Errors_Introduced": 0, "Net_Improvement": 0,
            "Num_Judges": 0, "Mean_JRS": 0.0, "Outlier_Judges": "",
            "Judge_Agreement_ESA": 0.0, "Weighted_CFRS": 0.0,
            "Initial_Proof_Length": 0, "Final_Proof_Length": 0, "Proof_Length_Growth": 0,
            "Final_Feedback_Preview": "",
            "Provider_Used": "", "Model_Used": "",
        }
        for col, val in defaults.items():
            if col not in result.columns:
                result[col] = val
        return result

    def _save_batch_log(self):
        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(self.batch_log, f, indent=2, default=str)

    def _finalize_batch(self, results_df: pd.DataFrame):
        self.batch_log["metadata"]["end_time"] = datetime.now().isoformat()
        self.batch_log["metadata"]["total"] = len(results_df)
        self.batch_log["metadata"]["completed"] = int(
            (results_df["Processing_Status"] == "COMPLETED").sum()
        )
        self.batch_log["metadata"]["failed"] = int(
            (results_df["Processing_Status"] == "FAILED").sum()
        )
        self._save_batch_log()
        self._generate_summary(results_df)

    def _generate_summary(self, df: pd.DataFrame):
        lines = ["=" * 80, "BATCH PROCESSING SUMMARY", "=" * 80]
        lines += [
            f"Generated    : {datetime.now():%Y-%m-%d %H:%M:%S}",
            f"Input        : {self.input_csv}",
            f"Output       : {self.output_dir}",
            "",
            f"Provider     : {self.provider_name}",
            f"Prover Model : {self.prover_model}",
            f"Evaluators   : {', '.join(self.evaluator_models)}",
            f"Multi-Judge  : {self.use_multi_judge}",
            f"Max Iter     : {self.max_iterations}",
            f"Temperatures : gen={self.temp_config.generation}, "
            f"corr={self.temp_config.correction}, eval={self.temp_config.evaluation}",
            "",
            f"Total        : {len(df)}",
            f"Completed    : {(df['Processing_Status'] == 'COMPLETED').sum()}",
            f"Failed       : {(df['Processing_Status'] == 'FAILED').sum()}",
            f"Total Time   : {df['Processing_Time_Seconds'].sum():.2f}s",
            f"Avg Time     : {df['Processing_Time_Seconds'].mean():.2f}s",
        ]

        comp = df[df["Processing_Status"] == "COMPLETED"]
        if len(comp) > 0:
            lines += ["", "-" * 40, "VERDICT DISTRIBUTION", "-" * 40]
            for v, c in comp["V2_Verdict"].value_counts().items():
                lines.append(f"  {v:15s}: {c:3d} ({c/len(comp)*100:.1f}%)")

            lines += ["", "-" * 40, "CRS STATISTICS", "-" * 40]
            lines += [
                f"  Mean CRS : {comp['CRS_Final'].mean():.3f}",
                f"  Median   : {comp['CRS_Final'].median():.3f}",
                f"  Std Dev  : {comp['CRS_Final'].std():.3f}",
                f"  Min      : {comp['CRS_Final'].min():.3f}",
                f"  Max      : {comp['CRS_Final'].max():.3f}",
                "",
                f"  Mean ERR : {comp['ERR_Final'].mean():.3f}",
                f"  Mean RP  : {comp['RP_Final'].mean():.3f}",
                f"  Mean TFP : {comp['TFP_Final'].mean():.3f}",
            ]

            crs_strong = (comp["CRS_Final"] >= 0.8).sum()
            crs_good = ((comp["CRS_Final"] >= 0.6) & (comp["CRS_Final"] < 0.8)).sum()
            crs_mid = ((comp["CRS_Final"] >= 0.4) & (comp["CRS_Final"] < 0.6)).sum()
            crs_weak = (comp["CRS_Final"] < 0.4).sum()
            lines += [
                "",
                f"  ≥ 0.80 (Strong)  : {crs_strong:3d} ({crs_strong/len(comp)*100:.1f}%)",
                f"  ≥ 0.60 (Good)    : {crs_good:3d} ({crs_good/len(comp)*100:.1f}%)",
                f"  ≥ 0.40 (Moderate): {crs_mid:3d} ({crs_mid/len(comp)*100:.1f}%)",
                f"  < 0.40 (Weak)    : {crs_weak:3d} ({crs_weak/len(comp)*100:.1f}%)",
            ]

            lines += ["", "-" * 40, "CRS PER ITERATION (mean over completed)", "-" * 40]
            for col in ["CRS_Iter_1", "CRS_Iter_2", "CRS_Iter_3"]:
                non_zero = comp[comp[col] > 0][col]
                if len(non_zero) > 0:
                    lines.append(f"  {col}: {non_zero.mean():.3f} (n={len(non_zero)})")

        lines += ["", "=" * 80, "END OF SUMMARY", "=" * 80]
        text = "\n".join(lines)
        self.summary_txt.write_text(text, encoding="utf-8")
        print("\n" + text)


def main():
    parser = argparse.ArgumentParser(description="Batch process convergence proofs")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="batch_results")
    parser.add_argument(
        "--provider", default="nvidia",
        choices=["nvidia", "openrouter", "openai", "anthropic"],
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--judges", nargs="+", default=None)
    parser.add_argument("--multi-judge", action="store_true")
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--temp-generation", type=float, default=0.7)
    parser.add_argument("--temp-correction", type=float, default=0.4)
    parser.add_argument("--temp-evaluation", type=float, default=0.05)
    parser.add_argument("--temp-verification", type=float, default=0.05)

    args = parser.parse_args()
    load_dotenv()

    if args.model is None:
        defaults = {
            "nvidia": "openai/gpt-oss-120b",
            "openrouter": "anthropic/claude-3.5-sonnet",
            "openai": "gpt-4",
            "anthropic": "claude-3-5-sonnet-20241022",
        }
        args.model = defaults[args.provider]
        print(f"Using default model: {args.model}")

    evaluator_models = args.judges or [args.model]
    temp_config = TemperatureConfig(
        generation=args.temp_generation,
        correction=args.temp_correction,
        evaluation=args.temp_evaluation,
        verification=args.temp_verification,
    )

    BatchProcessor(
        input_csv=args.input,
        output_dir=args.output,
        provider_name=args.provider,
        prover_model=args.model,
        evaluator_models=evaluator_models,
        api_key=args.api_key,
        use_multi_judge=args.multi_judge,
        max_iterations=args.max_iter,
        timeout=args.timeout,
        max_retries=args.max_retries,
        temp_config=temp_config,
    ).process()


if __name__ == "__main__":
    main()