# ============================================================================
# File: scripts/run_single.py
# ============================================================================
"""Script to run single proof generation and evaluation."""

import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.workflow import ProofWorkflow
from src.utils.logger import StructuredLogger


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
    
    # Setup logger
    logger = StructuredLogger(log_dir=args.output)
    
    # Setup workflow
    evaluator_models = args.judges if args.judges else [args.model]
    
    workflow = ProofWorkflow(
        prover_model=args.model,
        evaluator_models=evaluator_models,
        use_multi_judge=args.multi_judge,
        max_iterations=args.max_iter
    )
    
    # Run
    print("Starting proof generation and evaluation...")
    final_state = workflow.run(args.algorithm, args.assumptions)
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Save final proof
    if final_state and "current_proof" in final_state:
        proof_path = output_dir / "final_proof.tex"
        with open(proof_path, "w") as f:
            f.write(final_state["current_proof"])
        print(f"Proof saved to {proof_path}")
    
    # Save full state
    state_path = output_dir / "final_state.json"
    with open(state_path, "w") as f:
        json.dump(final_state, f, indent=2, default=str)
    print(f"State saved to {state_path}")
    
    logger.save()
    print(f"\n✅ Complete! Verdict: {final_state.get('verdict', 'UNKNOWN')}")


if __name__ == "__main__":
    main()