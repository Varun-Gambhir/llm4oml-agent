import json

bad_json = r"""{
  "hallucination_error": false,
  "detailed_feedback": "Proof claims \eta is correct, and \frac{1}{2} too. Also \"quote\", \\backslash, \n newline."
}"""

# Try parsing with strict=False
try:
    print(json.loads(bad_json, strict=False)) # strict=False is for control chars, won't fix escapes
except Exception as e:
    print("Normal failed.", e)
    
bad_json_escaped = bad_json.replace('\\', '\\\\')
print("Replaced:", bad_json_escaped)
try:
    # This might fail on \"quote\" because it becomes \\"quote\" -> invalid json quote
    print(json.loads(bad_json_escaped))
except Exception as e:
    print("Replaced failed.", e)

