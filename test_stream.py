import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage

client = ChatNVIDIA(model="qwen/qwq-32b", temperature=0.05, top_p=1)
response = client.invoke([HumanMessage(content="Please write a 1000 word essay about the history of mathematics. Do it slowly if possible, I just want to test.")])
print(response.content[:100])
