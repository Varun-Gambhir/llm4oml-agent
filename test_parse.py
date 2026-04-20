import re
import json

def _parse_response(text: str):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    
    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        print("matched json block")
        json_str = match.group(1)
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
        try:
            data = json.loads(json_str)
            return data
        except json.JSONDecodeError as e:
            print("JSON decode error:", e)
            pass

    key_idx = text.find('"hallucination_error"')
    if key_idx != -1:
        start_idx = text.rfind('{', 0, key_idx)
    else:
        start_idx = text.find('{')
        
    if start_idx == -1:
        return None
        
    end_idx = len(text)
    data = None
    while True:
        end_idx = text.rfind('}', start_idx, end_idx)
        if end_idx == -1:
            break
            
        json_str = text[start_idx:end_idx+1]
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
        try:
            data = json.loads(json_str)
            break
        except json.JSONDecodeError:
            end_idx -= 1

    return data

text = """
```json
{
  "hallucination_error": false,
  "missing_step": true,
  "operator_error": false,
  "completeness_score": 3,
  "assumption_use_score": 4,
  "overall_verdict": "FAIL",
  "detailed_feedback": "blabla",
  "hallucination_steps": [],
  "missing_step_indices": [5],
  "operator_error_steps": [],
  "assumption_violation_steps": [],
  "flagged_steps": [5]
}
```
"""
print(_parse_response(text))
