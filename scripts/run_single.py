# ============================================================================
# File: scripts/run_single.py (UPDATED)
# ============================================================================
"""Script to run single proof generation with provider support."""

import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.workflow import ProofWorkflow


def main():
    parser = argparse.ArgumentParser(description="Generate and evaluate convergence proof")
    parser.add_argument("--algorithm", required=True, help="Algorithm description")
    parser.add_argument("--assumptions", required=True, help="Assumptions")
    
    # Provider configuration
    parser.add_argument("--provider", default="nvidia", 
                       choices=["nvidia", "openrouter", "openai", "anthropic"],
                       help="LLM provider to use")
    parser.add_argument("--api-key", help="API key (overrides environment variable)")
    
    # Model configuration
    parser.add_argument("--model", help="Model to use (provider-specific)")
    parser.add_argument("--judges", nargs="+", help="List of judge models")
    parser.add_argument("--multi-judge", action="store_true", help="Use multi-judge evaluation")
    
    # Generation parameters
    parser.add_argument("--max-iter", type=int, default=3, help="Max iterations")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retry attempts")
    
    # Output
    parser.add_argument("--output", default="output", help="Output directory")
    
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
    
    # Create workflow
    workflow = ProofWorkflow(
        provider_name=args.provider,
        prover_model=args.model,
        evaluator_models=evaluator_models,
        api_key=args.api_key,
        use_multi_judge=args.multi_judge,
        max_iterations=args.max_iter,
        temperature=args.temperature,
        timeout=args.timeout,
        max_retries=args.max_retries,
        output_dir=args.output
    )
    
    # Run with tracking
    print(f"\nProvider: {args.provider}")
    print(f"Model: {args.model}")
    print(f"Timeout: {args.timeout}s")
    print(f"Max Retries: {args.max_retries}\n")
    
    final_state, tracker = workflow.run(args.algorithm, args.assumptions)
    
    # Print summary
    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    
    log = tracker.get_log()
    
    print(f"\nProvider: {args.provider}")
    print(f"Algorithm: {log['algorithm_description'][:60]}...")
    print(f"Total Iterations: {log['final_results']['total_iterations']}")
    print(f"Final Verdict: {log['final_results']['verdict']}")
    print(f"Convergence: {'Yes' if log['final_results']['convergence_achieved'] else 'No'}")
    
    if "statistics" in log:
        stats = log["statistics"]
        print(f"\nTotal Errors Identified: {stats['total_errors_identified']}")
        
        if stats["crs_progression"]:
            print(f"CRS Progression: {' → '.join([f'{x:.2f}' for x in stats['crs_progression']])}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()