# ============================================================================
# File: scripts/run_single.py (UPDATED)
# ============================================================================
"""Script to run single proof generation with complete tracking."""

import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.workflow import ProofWorkflow


def main():
    parser = argparse.ArgumentParser(description="Generate and evaluate convergence proof")
    parser.add_argument("--algorithm", required=True, help="Algorithm description")
    parser.add_argument("--assumptions", required=True, help="Assumptions")
    parser.add_argument("--model", default="openai/gpt-oss-120b", help="Model to use")
    parser.add_argument("--multi-judge", action="store_true", help="Use multi-judge evaluation")
    parser.add_argument("--judges", nargs="+", help="List of judge models")
    parser.add_argument("--max-iter", type=int, default=3, help="Max iterations")
    parser.add_argument("--output", default="output", help="Output directory")
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    
    # Setup workflow
    evaluator_models = args.judges if args.judges else [args.model]
    
    workflow = ProofWorkflow(
        prover_model=args.model,
        evaluator_models=evaluator_models,
        use_multi_judge=args.multi_judge,
        max_iterations=args.max_iter,
        output_dir=args.output
    )
    
    # Run with tracking
    final_state, tracker = workflow.run(args.algorithm, args.assumptions)
    
    # Print summary
    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    
    log = tracker.get_log()
    
    print(f"\nAlgorithm: {log['algorithm_description'][:60]}...")
    print(f"Total Iterations: {log['final_results']['total_iterations']}")
    print(f"Final Verdict: {log['final_results']['verdict']}")
    print(f"Convergence: {'Yes' if log['final_results']['convergence_achieved'] else 'No'}")
    
    if "statistics" in log:
        stats = log["statistics"]
        print(f"\nTotal Errors Identified: {stats['total_errors_identified']}")
        print(f"Error Breakdown:")
        for error_type, count in stats["errors_by_type"].items():
            if count > 0:
                print(f"  - {error_type}: {count}")
        
        if stats["crs_progression"]:
            print(f"\nCRS Progression: {' → '.join([f'{x:.2f}' for x in stats['crs_progression']])}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()