import json

# Suppose the LLM outputs exactly \ f r a c with one slash.
# In Python string literal, we use raw string or double escape to represent it.
bad_json = r"""{
  "hallucination_error": false,
  "detailed_feedback": "Proof claims \frac{1}{T} which is correct."
}"""

print("JSON string:", bad_json)

try:
    json.loads(bad_json)
    print("Success")
except Exception as e:
    print("Error:", e)
