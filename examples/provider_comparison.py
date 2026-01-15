# ============================================================================
# File: examples/provider_comparison.py
# ============================================================================
"""Compare different providers on the same algorithm."""

import subprocess
import json
from pathlib import Path

algorithm = "SGD: w_{t+1} = w_t - η∇f(w_t)"
assumptions = "Convex, L-smooth"

providers = [
    {
        "name": "nvidia",
        "model": "openai/gpt-oss-120b",
        "key_env": "NVIDIA_API_KEY"
    },
    {
        "name": "openrouter",
        "model": "anthropic/claude-3.5-sonnet",
        "key_env": "OPENROUTER_API_KEY"
    },
    {
        "name": "openai",
        "model": "gpt-4",
        "key_env": "OPENAI_API_KEY"
    }
]

results = []

for provider in providers:
    print(f"\n{'='*60}")
    print(f"Testing provider: {provider['name']}")
    print(f"{'='*60}\n")
    
    output_dir = f"output/comparison_{provider['name']}"
    
    cmd = [
        "python", "scripts/run_single.py",
        "--provider", provider["name"],
        "--model", provider["model"],
        "--algorithm", algorithm,
        "--assumptions", assumptions,
        "--output", output_dir,
        "--max-iter", "2",
        "--timeout", "600",
        "--max-retries", "3"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        
        # Load results
        log_files = list(Path(output_dir).glob("execution_log_*.json"))
        if log_files:
            with open(log_files[0]) as f:
                log = json.load(f)
                
            results.append({
                "provider": provider["name"],
                "model": provider["model"],
                "verdict": log["final_results"]["verdict"],
                "iterations": log["final_results"]["total_iterations"],
                "crs": log.get("statistics", {}).get("crs_progression", [])
            })
    
    except Exception as e:
        print(f"Error with {provider['name']}: {e}")
        results.append({
            "provider": provider["name"],
            "error": str(e)
        })

# Print comparison
print("\n" + "="*60)
print("PROVIDER COMPARISON RESULTS")
print("="*60)

for result in results:
    print(f"\nProvider: {result['provider']}")
    if "error" in result:
        print(f"  Status: ERROR - {result['error']}")
    else:
        print(f"  Model: {result['model']}")
        print(f"  Verdict: {result['verdict']}")
        print(f"  Iterations: {result['iterations']}")
        if result['crs']:
            print(f"  CRS: {' → '.join([f'{x:.2f}' for x in result['crs']])}")