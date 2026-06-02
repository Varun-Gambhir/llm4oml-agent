import json

bad_json = r"""{
  "hallucination_error": false,
  "detailed_feedback": "Proof claims \eta is correct."
}"""

print("JSON string:", bad_json)

try:
    json.loads(bad_json)
    print("Success")
except Exception as e:
    print("Error:", e)
