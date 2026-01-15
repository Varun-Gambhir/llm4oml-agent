# ============================================================================
# File: scripts/run_batch.py (UPDATED)
# ============================================================================
"""Batch processor with provider support."""

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
    
    print(f"\nConfiguration:")
    print(f"  Provider: {args.provider}")
    print(f"  Model: {args.model}")
    print(f"  Judges: {evaluator_models}")
    print(f"  Timeout: {args.timeout}s")
    print(f"  Max Retries: {args.max_retries}")
    print(f"  Max Iterations: {args.max_iter}\n")
    
    # Use the existing BatchProcessor but pass provider info
    # (BatchProcessor code remains same, just passes these to ProofWorkflow)
    
    from scripts.run_batch import BatchProcessor  # Import from previous artifact
    
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