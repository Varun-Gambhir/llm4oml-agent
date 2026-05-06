#!/usr/bin/env python3
"""Iterative auto-prover: LLM generation -> Lean verification -> LLM repair loop.

This script is designed to automate future proof-building runs:
1) Generate Lean from a natural-language claim
2) Run strict Lean verification
3) Feed diagnostics back to the LLM for repair
4) Repeat until pass or max attempts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.formal.lean_probe import LeanProbe
from src.formal.prompts import AUTO_PROOF_PROMPT, AUTO_PROOF_REPAIR_PROMPT


def _strip_lean_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```lean"):
        text = text[len("```lean"):]
    elif text.startswith("```"):
        text = text[len("```"):]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _load_claim(claim: str, claim_file: str | None) -> str:
    if claim_file:
        return Path(claim_file).read_text(encoding="utf-8").strip()
    return claim.strip()


def _stub_llm_response(attempt: int) -> str:
    # Intentionally fail first attempt (contains sorry), then repair to a full proof.
    if attempt == 1:
        return (
            "import ProofChecks.Lemmas\n\n"
            "theorem auto_abs_nonneg (x : ℝ) : 0 ≤ |x| := by\n"
            "  sorry\n"
        )
    return (
        "import ProofChecks.Lemmas\n\n"
        "theorem auto_abs_nonneg (x : ℝ) : 0 ≤ |x| := by\n"
        "  exact abs_nonneg x\n"
    )


def _build_diagnostics(errors: list[str]) -> str:
    if not errors:
        return "No errors reported by Lean."
    return "\n".join(f"- {e}" for e in errors)


def run_auto_prove(
    claim: str,
    context: str,
    max_attempts: int,
    out_name: str,
    use_stub: bool,
    provider: str,
    model: str,
    api_key: str | None,
    temperature: float,
) -> dict[str, Any]:
    lean_dir = ROOT / "lean"
    output_dir = ROOT / "output" / "auto_prove"
    output_dir.mkdir(parents=True, exist_ok=True)

    llm = None
    if not use_stub:
        from src.llm.provider_factory import LLMProviderFactory
        llm = LLMProviderFactory.create(
            provider_name=provider,
            model_name=model,
            api_key=api_key,
            temperature=temperature,
        )

    # Cache: map claim_hash -> {status, lean_code, timestamp}
    cache_file = ROOT / "output" / "auto_prove" / "cache.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as cf:
                cache = json.load(cf)
        except Exception:
            cache = {}
    else:
        cache = {}

    def _save_cache():
        with open(cache_file, "w", encoding="utf-8") as cf:
            json.dump(cache, cf, indent=2)

    def _claim_hash(s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    # Rate limiting: minimum seconds between LLM calls
    last_call = {"t": 0.0}
    min_delay = float(1.0)

    def _invoke_llm_with_rate_limit(prompt: list[dict[str, str]], temperature: float, max_attempts: int = 3) -> str:
        # simple rate limiter
        now = time.time()
        elapsed = now - last_call["t"]
        if elapsed < min_delay:
            time.sleep(min_delay - elapsed)
        last_call["t"] = time.time()

        # wrapper for provider invoke with extra retry/backoff
        last_exc = None
        for attempt in range(max_attempts):
            try:
                return llm.invoke(prompt, temperature=temperature)
            except Exception as exc:
                last_exc = exc
                backoff = 2 ** attempt
                time.sleep(backoff)
        raise RuntimeError(f"LLM invoke failed after {max_attempts} attempts: {last_exc}")

    probe = LeanProbe(str(lean_dir), llm or SimpleNamespace(invoke=lambda *a, **k: ""))
    attempts: list[dict[str, Any]] = []
    prev_lean = ""
    prev_errors: list[str] = []

    claim_hash = _claim_hash(claim)

    # If cached success exists, restore and exit early
    cached = cache.get(claim_hash)
    if cached and cached.get("status") == "pass":
        final_file = lean_dir / out_name
        final_file.write_text(cached.get("lean_code", ""), encoding="utf-8")
        return {
            "status": "pass",
            "attempts": [],
            "final_file": str(final_file),
            "cached": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    for attempt in range(1, max_attempts + 1):
        if attempt == 1:
            prompt = AUTO_PROOF_PROMPT.format(claim=claim, context=context or "(none)")
        else:
            prompt = AUTO_PROOF_REPAIR_PROMPT.format(
                claim=claim,
                previous_lean=prev_lean,
                diagnostics=_build_diagnostics(prev_errors),
            )

        if use_stub:
            raw = _stub_llm_response(attempt)
        else:
            raw = _invoke_llm_with_rate_limit([{"role": "user", "content": prompt}], temperature)

        lean_code = _strip_lean_fences(raw)
        attempt_file = lean_dir / f"AutoProof_attempt_{attempt}.lean"
        attempt_file.write_text(lean_code, encoding="utf-8")

        verdict = probe._run_lean(lean_code, [claim])
        attempts.append(
            {
                "attempt": attempt,
                "file": str(attempt_file),
                "status": verdict.status,
                "errors": verdict.errors,
                "elapsed_seconds": verdict.elapsed_seconds,
            }
        )

        if verdict.status == "pass":
            final_file = lean_dir / out_name
            final_file.write_text(lean_code, encoding="utf-8")
            # update cache
            cache[claim_hash] = {
                "status": "pass",
                "lean_code": lean_code,
                "final_file": str(final_file),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            _save_cache()
            result = {
                "status": "pass",
                "attempts": attempts,
                "final_file": str(final_file),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            log_path = output_dir / "auto_prove_last.json"
            log_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            return result

        prev_lean = lean_code
        prev_errors = verdict.errors

    result = {
        "status": "fail",
        "attempts": attempts,
        "final_file": str(lean_dir / out_name),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    log_path = output_dir / "auto_prove_last.json"
    log_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Iterative LLM-to-Lean auto-prover")
    parser.add_argument("--claim", type=str, default="", help="Claim text")
    parser.add_argument("--claim-file", type=str, default=None, help="Path to claim text file")
    parser.add_argument("--context", type=str, default="", help="Optional context for claim")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--out", type=str, default="AutoProof_final.lean")

    parser.add_argument("--provider", type=str, default="openai", choices=["nvidia", "openrouter", "openai", "anthropic"])
    parser.add_argument("--model", type=str, default="gpt-4")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--min-delay", type=float, default=1.0, help="Minimum seconds between LLM calls (rate limit)")
    parser.add_argument("--cache-file", type=str, default=None, help="Path to cache json (optional)")
    parser.add_argument("--stub", action="store_true", help="Run without API calls using deterministic stub responses")

    args = parser.parse_args()

    claim_text = _load_claim(args.claim, args.claim_file)
    if not claim_text:
        raise ValueError("Provide `--claim` or `--claim-file`.")

    result = run_auto_prove(
        claim=claim_text,
        context=args.context,
        max_attempts=args.max_attempts,
        out_name=args.out,
        use_stub=args.stub,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
