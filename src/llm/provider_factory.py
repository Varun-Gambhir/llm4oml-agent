# ============================================================================
# File: src/llm/provider_factory.py
# ============================================================================
"""Factory for creating LLM providers."""

import os
from typing import Optional
from .base_provider import BaseLLMProvider
from .nvidia_provider import NVIDIAProvider
from .openrouter_provider import OpenRouterProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .google_ai_studio_provider import GoogleAIStudioProvider

_ENV_VARS = {
    "nvidia": ("NVIDIA_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "google": ("GOOGLE_AI_STUDIO_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "google_ai_studio": ("GOOGLE_AI_STUDIO_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"),
}

_PROVIDERS = {
    "nvidia": NVIDIAProvider,
    "openrouter": OpenRouterProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleAIStudioProvider,
    "google_ai_studio": GoogleAIStudioProvider,
}


class LLMProviderFactory:
    """Factory for creating LLM providers based on configuration."""

    @classmethod
    def create(
        cls,
        provider_name: str,
        model_name: str,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> BaseLLMProvider:
        """
        Create an LLM provider instance.

        Parameters
        ----------
        provider_name : "nvidia" | "openrouter" | "openai" | "anthropic" |
                        "google" | "google_ai_studio"
        model_name    : provider-specific model identifier
        api_key       : API key; falls back to environment variable if None
        **kwargs      : forwarded to provider constructor (temperature, timeout, etc.)
        """
        key = provider_name.lower()
        if key not in _PROVIDERS:
            raise ValueError(
                f"Unknown provider: {provider_name!r}. Available: {list(_PROVIDERS)}"
            )

        if api_key is None:
            env_vars = _ENV_VARS[key]
            api_key = next((os.getenv(env_var) for env_var in env_vars if os.getenv(env_var)), None)
            if not api_key:
                raise ValueError(
                    f"API key not found for provider {provider_name!r}. "
                    f"Set one of {', '.join(env_vars)} or pass api_key."
                )

        return _PROVIDERS[key](model_name, api_key, **kwargs)

    @classmethod
    def from_config(cls, config: dict) -> BaseLLMProvider:
        """Create provider from a configuration dictionary."""
        config = dict(config)  # copy so we don't mutate caller's dict
        provider_name = config.pop("provider")
        model_name = config.pop("model")
        api_key = config.pop("api_key", None)
        return cls.create(provider_name, model_name, api_key, **config)
