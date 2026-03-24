import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage

client = ChatNVIDIA(model="openai/gpt-oss-20b")
print("Client created. Invoking...")
try:
    res = client.invoke([HumanMessage(content="Hello")])
    print(res)
except Exception as e:
    print("Error:", type(e), e)
