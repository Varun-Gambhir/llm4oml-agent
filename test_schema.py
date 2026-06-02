import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from src.models.schemas import EvaluationMetrics

data = {
   "hallucination_error": False,
   "missing_step": False,
   "operator_error": False,
   "completeness_score": 5,
   "assumption_use_score": 5,
   "overall_verdict": "PASS",
   "detailed_feedback": "Looks good."
}

m = EvaluationMetrics(**data)
print("Success:", m)
