# ============================================================================
# File: src/llm/google_ai_studio_provider.py
# ============================================================================
"""Google AI Studio / Gemini API provider."""

from typing import Optional, List, Dict
from urllib.parse import quote

import requests

from .base_provider import BaseLLMProvider


class GoogleAIStudioProvider(BaseLLMProvider):
    """Google AI Studio Gemini API provider."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(model_name, api_key, **kwargs)

    def _make_request(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        system_text = None
        contents = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_text = content if system_text is None else f"{system_text}\n\n{content}"
                continue

            gemini_role = "model" if role == "assistant" else "user"
            contents.append(
                {
                    "role": gemini_role,
                    "parts": [{"text": content}],
                }
            )

        payload: Dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature if temperature is not None else self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
        }
        if system_text:
            payload["systemInstruction"] = {"parts": [{"text": system_text}]}

        model = self.model_name.removeprefix("models/")
        model_path = quote(model, safe="")
        url = f"{self.BASE_URL}/models/{model_path}:generateContent"
        response = requests.post(
            url,
            params={"key": self.api_key},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        data = response.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(part.get("text", "") for part in parts).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Gemini API response: {data}") from exc
