import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage

try:
    client = ChatNVIDIA(model="openai/gpt-oss-120b", temperature=0.05, max_tokens=1500)
    messages = [HumanMessage(content="Evaluate this JSON")]
    full = []
    for chunk in client.stream(messages):
        full.append(str(chunk.content))
    print("Full response:", "".join(full))
except Exception as e:
    print("Error:", e)
