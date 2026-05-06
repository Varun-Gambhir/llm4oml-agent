import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.formal.lean_probe import LeanProbe


def test_run_lean_rejects_sorry_and_uses_no_sorry(monkeypatch, tmp_path):
    monkeypatch.setattr(LeanProbe, "_check_lean", lambda self: True)

    calls = []

    def fake_run(args, cwd=None, capture_output=None, text=None, timeout=None):
        calls.append({"args": args, "cwd": cwd, "timeout": timeout})
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("src.formal.lean_probe.subprocess.run", fake_run)

    probe = LeanProbe(str(tmp_path), SimpleNamespace(invoke=lambda *args, **kwargs: ""))
    verdict = probe._run_lean("theorem t : True := by sorry", ["True"])

    assert verdict.status == "fail"
    assert any("strict verification requires complete proofs" in error for error in verdict.errors)
    assert calls[0]["args"] == [probe._lake_bin, "build"]


def test_run_lean_passes_without_sorry(monkeypatch, tmp_path):
    monkeypatch.setattr(LeanProbe, "_check_lean", lambda self: True)

    def fake_run(args, cwd=None, capture_output=None, text=None, timeout=None):
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("src.formal.lean_probe.subprocess.run", fake_run)

    probe = LeanProbe(str(tmp_path), SimpleNamespace(invoke=lambda *args, **kwargs: ""))
    verdict = probe._run_lean("theorem t : True := by trivial", ["True"])

    assert verdict.status == "pass"
    assert verdict.errors == []