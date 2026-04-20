# src/formal/lean_probe.py
import subprocess
import time
import logging
import os
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LeanVerdict:
    status: str          # "pass" | "sorry_pass" | "fail" | "timeout" | "unavailable"
    errors: list[str] = field(default_factory=list)
    lean_code: str = ""
    elapsed_seconds: float = 0.0
    claims_extracted: list[str] = field(default_factory=list)

    @property
    def is_useful(self) -> bool:
        """True when Lean ran and produced actionable output."""
        return self.status not in ("unavailable", "timeout")

    def to_feedback_string(self) -> str:
        if self.status == "pass":
            return "LEAN: All theorem statements type-check with complete proofs."
        if self.status == "sorry_pass":
            return (
                "LEAN: Theorem statements are well-typed. "
                "Proofs use sorry placeholders — logical structure accepted, "
                "but correctness is not formally verified."
            )
        if self.status == "fail":
            err_block = "\n".join(f"  - {e}" for e in self.errors[:5])
            return (
                f"LEAN: Type-checking FAILED. The following claims could not be "
                f"expressed as valid Lean 4 statements:\n{err_block}\n"
                f"This often indicates a missing assumption, an incorrect quantifier, "
                f"or a claim that doesn't follow from stated hypotheses."
            )
        return "LEAN: Could not run (Lean not installed or timed out)."


class LeanProbe:
    """
    Non-destructive Lean 4 probe.

    Runs ALONGSIDE the existing LLM evaluator. Never blocks the main pipeline.
    Returns LeanVerdict(status="unavailable") if Lean is not installed.
    """

    LEAN_TIMEOUT = 120  # seconds — mathlib builds are slow

    def __init__(self, lean_project_dir: str, llm_provider):
        self.lean_dir = Path(lean_project_dir)
        self.llm = llm_provider
        # Resolve elan binary paths (lean/lake may not be on system PATH)
        elan_bin = Path.home() / ".elan" / "bin"
        self._lean_bin = str(elan_bin / "lean") if (elan_bin / "lean").exists() else "lean"
        self._lake_bin = str(elan_bin / "lake") if (elan_bin / "lake").exists() else "lake"
        self._lean_available = self._check_lean()

    def _check_lean(self) -> bool:
        try:
            result = subprocess.run(
                [self._lean_bin, "--version"],
                capture_output=True, timeout=10
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("[LeanProbe] Lean not found — probe will be skipped.")
            return False

    def probe(self, latex_proof: str, algorithm: str = "") -> LeanVerdict:
        if not self._lean_available:
            return LeanVerdict(status="unavailable")

        # Step 1: extract claims via LLM
        claims = self._extract_claims(latex_proof, algorithm)

        # Step 2: translate to Lean 4 via LLM
        lean_code = self._translate_to_lean(claims, algorithm)

        # Step 3: write to file and run Lean
        return self._run_lean(lean_code, claims)

    def _extract_claims(self, proof: str, algorithm: str) -> list[str]:
        from .prompts import CLAIM_EXTRACTION_PROMPT
        response = self.llm.invoke(
            [{"role": "user", "content": CLAIM_EXTRACTION_PROMPT.format(
                proof=proof, algorithm=algorithm
            )}],
            temperature=0.05,
        )
        # Parse numbered list "1. ..." from response
        lines = [
            line.lstrip("0123456789. ").strip()
            for line in response.splitlines()
            if line.strip() and line.strip()[0].isdigit()
        ]
        return lines[:6]  # cap at 6 claims

    def _translate_to_lean(self, claims: list[str], algorithm: str) -> str:
        from .prompts import LEAN_TRANSLATION_PROMPT
        claims_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
        return self.llm.invoke(
            [{"role": "user", "content": LEAN_TRANSLATION_PROMPT.format(
                claims=claims_text, algorithm=algorithm
            )}],
            temperature=0.05,
        )

    def _run_lean(self, lean_code: str, claims: list[str]) -> LeanVerdict:
        # Strip markdown fences if LLM wrapped the code
        import re
        lean_code = re.sub(r"```lean\s*", "", lean_code)
        lean_code = re.sub(r"```\s*$", "", lean_code, flags=re.MULTILINE).strip()

        target = self.lean_dir / "ProofChecks.lean"
        target.write_text(lean_code, encoding="utf-8")

        start = time.time()
        try:
            result = subprocess.run(
                [self._lake_bin, "build"],
                cwd=self.lean_dir,
                capture_output=True,
                text=True,
                timeout=self.LEAN_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return LeanVerdict(
                status="timeout",
                lean_code=lean_code,
                elapsed_seconds=self.LEAN_TIMEOUT,
                claims_extracted=claims,
            )

        elapsed = time.time() - start
        has_sorry = "sorry" in lean_code
        errors = [
            line.strip()
            for line in (result.stdout + result.stderr).splitlines()
            if "error:" in line.lower()
        ]

        if result.returncode == 0:
            status = "sorry_pass" if has_sorry else "pass"
        else:
            status = "fail"

        return LeanVerdict(
            status=status,
            errors=errors[:10],
            lean_code=lean_code,
            elapsed_seconds=round(elapsed, 2),
            claims_extracted=claims,
        )
