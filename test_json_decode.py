import json

bad_json = """{
  "hallucination_error": false,
  "detailed_feedback": "Proof claims \\frac{1}{T} which is correct."
}"""

try:
    json.loads(bad_json)
    print("Success")
except Exception as e:
    print("Error:", e)
