import json
from unittest.mock import MagicMock
# Just print the prompt that single_judge generates to see if it's too long
from src.utils.prompts import PromptManager

pm = PromptManager()
proof = "DUMMY PROOF " * 3000
prompt = pm.get_initial_evaluation_prompt(proof)
print(f"Prompt length: {len(prompt)}")
