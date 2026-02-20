# ============================================================================
# File: src/llm/nvidia_provider.py
# ============================================================================
"""NVIDIA AI Endpoints provider."""

from typing import Optional, List, Dict
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, SystemMessage
from .base_provider import BaseLLMProvider


class NVIDIAProvider(BaseLLMProvider):
    """NVIDIA NIM / AI Endpoints LLM provider."""

    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        # Client is created fresh per request when temperature changes
        self._base_client = self._build_client(self.temperature)

    def _build_client(self, temperature: float) -> ChatNVIDIA:
        return ChatNVIDIA(
            model=self.model_name,
            api_key=self.api_key,
            temperature=temperature,
            max_tokens=self.max_tokens,
            top_p=1,
            timeout=self.timeout,
        )

    def _make_request(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        client = (
            self._build_client(temperature)
            if temperature is not None and temperature != self.temperature
            else self._base_client
        )

        lc_messages = []
        for msg in messages:
            if msg["role"] == "system":
                lc_messages.append(SystemMessage(content=msg["content"]))
            else:
                lc_messages.append(HumanMessage(content=msg["content"]))

        response = client.invoke(lc_messages)
        return response.content