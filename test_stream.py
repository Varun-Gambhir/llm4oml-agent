import sys
import logging
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage

client = ChatNVIDIA(model="qwen/qwq-32b", temperature=0.05, top_p=1)
messages = [HumanMessage(content="Please reason heavily about pi starting now...")]
import time
start = time.time()
print("Starting stream...")
try:
    for chunk in client.stream(messages):
        sys.stdout.write(".")
        sys.stdout.flush()
    print("\nFinished in", time.time() - start)
except Exception as e:
    print("Error:", e)
