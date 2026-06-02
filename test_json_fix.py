import json
import re

bad_json = r"""{
  "hallucination_error": false,
  "detailed_feedback": "Proof claims \eta is correct, and \frac{1}{2} too. Also \"quote\", \\backslash, \n newline."
}"""

def fix_json_escapes(json_str: str) -> str:
    # replace backslashes that are followed by something invalid with double backslashes
    return re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', json_str)

fixed = fix_json_escapes(bad_json)
print("Fixed:", fixed)
try:
    print(json.loads(fixed))
except Exception as e:
    print("Error:", e)
