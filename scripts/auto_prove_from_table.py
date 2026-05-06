#!/usr/bin/env python3
"""Run auto_prove over each row in `table.csv`.

Creates a claim from `Problem Statement` and `Assumptions Used` and calls
`scripts/auto_prove.py` for each entry. Saves per-row logs under
`output/auto_prove/csv_runs/`.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "table.csv"
OUT_DIR = ROOT / "output" / "auto_prove" / "csv_runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def row_to_claim(row: dict[str, str]) -> str:
    tid = row.get("Task_ID", "")
    stmt = (row.get("Problem Statement") or "").strip()
    assump = (row.get("Assumptions Used") or "").strip()
    claim = f"Task {tid}: {stmt}\nAssumptions: {assump}"
    return claim


def run_for_csv(csv_path: Path, start: int | None, end: int | None, stub: bool, provider: str, model: str, api_key: str | None, max_attempts: int):
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    selection = rows[slice(start - 1 if start else None, end if end else None)]

    for i, row in enumerate(selection, start=(start or 1)):
        claim = row_to_claim(row)
        print(f"Running auto_prove for row {i}: Task_ID={row.get('Task_ID')}")

        # write claim to temp file to pass via --claim-file
        with NamedTemporaryFile(mode="w+", suffix=".txt", delete=False, encoding="utf-8") as t:
            t.write(claim)
            t.flush()

            cmd = [
                "python3", "scripts/auto_prove.py",
                "--claim-file", t.name,
                "--max-attempts", str(max_attempts),
                "--out", f"AutoProof_row_{i}.lean",
            ]
            if stub:
                cmd.append("--stub")
            else:
                cmd.extend(["--provider", provider, "--model", model])
                if api_key:
                    cmd.extend(["--api-key", api_key])

            # ensure PYTHONPATH includes repo root so scripts import correctly
            result = subprocess.run(cmd, capture_output=True, text=True)

        # Save stdout/stderr and the auto_prove log
        log_file = OUT_DIR / f"row_{i}_log.json"
        summary = {
            "task_id": row.get("Task_ID"),
            "claim": claim,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        log_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved run log to {log_file}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default=str(CSV_PATH))
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--stub", action="store_true")
    p.add_argument("--provider", type=str, default="openai")
    p.add_argument("--model", type=str, default="gpt-4")
    p.add_argument("--api-key", type=str, default=None)
    p.add_argument("--max-attempts", type=int, default=3)
    args = p.parse_args()

    run_for_csv(Path(args.csv), args.start, args.end, args.stub, args.provider, args.model, args.api_key, args.max_attempts)


if __name__ == "__main__":
    main()
