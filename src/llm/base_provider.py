# ============================================================================
# File: src/llm/base_provider.py
# ============================================================================
"""Base LLM provider interface for easy switching between APIs."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import time


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(
        self,
        model_name: str,
        api_key: str,
        temperature: float = 0.5,
        max_tokens: int = 8192,
        timeout: int = 600,  # 10 minutes
        max_retries: int = 3,
        retry_delay: int = 5
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    @abstractmethod
    def _make_request(self, messages: list, **kwargs) -> str:
        """Make the actual API request. Implemented by subclasses."""
        pass
    
    def invoke(self, messages: list, **kwargs) -> str:
        """
        Invoke the LLM with retry logic.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Response text
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    print(f"[{self.__class__.__name__}] Retry attempt {attempt + 1}/{self.max_retries}")
                    time.sleep(self.retry_delay * attempt)  # Exponential backoff
                
                response = self._make_request(messages, **kwargs)
                return response
                
            except Exception as e:
                last_error = e
                error_msg = str(e)
                
                # Check if it's a timeout
                if "504" in error_msg or "timeout" in error_msg.lower():
                    print(f"[{self.__class__.__name__}] Timeout error (attempt {attempt + 1})")
                    if attempt < self.max_retries - 1:
                        print(f"[{self.__class__.__name__}] Waiting {self.retry_delay * (attempt + 1)}s before retry...")
                        continue
                
                # Check if it's a rate limit
                elif "429" in error_msg or "rate limit" in error_msg.lower():
                    print(f"[{self.__class__.__name__}] Rate limit hit (attempt {attempt + 1})")
                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                        print(f"[{self.__class__.__name__}] Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue
                
                # Other errors
                else:
                    print(f"[{self.__class__.__name__}] Error: {error_msg}")
                    if attempt < self.max_retries - 1:
                        continue
        
        # All retries exhausted
        raise Exception(f"All {self.max_retries} retry attempts failed. Last error: {last_error}")
