import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from src.evaluators.single_judge import SingleJudge
import logging
logging.basicConfig(level=logging.DEBUG)

class DummySingleJudge(SingleJudge):
    def __init__(self):
        self.judge_id = "test"
        self.model_name = "test"

json_str = """{
   "hallucination_error": false,
   "missing_step": false,
   "operator_error": false,
   "completeness_score": 5,
   "assumption_use_score": 5,
   "overall_verdict": "PASS",
   "detailed_feedback": "Looks good."
}"""

judge = DummySingleJudge()
res = judge._parse_response(json_str)
print("Result:", res)
