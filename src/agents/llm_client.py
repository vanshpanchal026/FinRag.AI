"""Unified LLM Client module for FinRAG-Eval.

Supports Groq Cloud API (https://api.groq.com/openai/v1) and custom
OpenAI-compatible endpoints with automatic exponential backoff retry
for HTTP 429/503 rate limits and enforced JSON schema generation.

Sources:
- OpenAI Python SDK: https://github.com/openai/openai-python
- Techstack.md Section 4
"""

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Type, Union
from openai import APIConnectionError, InternalServerError, OpenAI, RateLimitError
from pydantic import BaseModel

from src.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified OpenAI-compatible LLM client supporting Groq and custom providers."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        backoff_factor: float = 2.0,
    ):
        # Resolve API key: Groq first, then OpenAI/custom, then fallback
        self.api_key = (
            api_key
            or settings.groq_api_key
            or settings.openai_api_key
            or os.environ.get("GROQ_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "dummy_key_for_testing"
        )

        # Resolve Base URL
        if base_url:
            self.base_url = base_url
        elif settings.groq_api_key or os.environ.get("GROQ_API_KEY"):
            self.base_url = "https://api.groq.com/openai/v1"
        elif settings.openai_base_url:
            self.base_url = settings.openai_base_url
        else:
            self.base_url = "https://api.groq.com/openai/v1"

        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call LLM chat completions with automatic exponential backoff retry.

        Args:
            messages: List of message dictionaries [{'role': ..., 'content': ...}].
            model: Target model name (default: settings.main_agent_model).
            temperature: Sampling temperature (default 0.1 for high precision).
            max_tokens: Maximum tokens in response.

        Returns:
            String response content.
        """
        target_model = model or settings.main_agent_model
        params: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            params["max_tokens"] = max_tokens

        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**params)
                return response.choices[0].message.content or ""
            except (RateLimitError, InternalServerError, APIConnectionError) as e:
                last_exception = e
                delay = self.base_delay * (self.backoff_factor**attempt)
                logger.warning(
                    f"LLM API rate limit or transient error on attempt {attempt + 1}: {e}. Retrying in {delay:.2f}s..."
                )
                time.sleep(delay)
            except Exception as e:
                # Non-retryable error
                raise e

        if last_exception:
            raise last_exception
        return ""

    def generate_json(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        response_model: Optional[Type[BaseModel]] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Generate structured JSON response, optionally validated against Pydantic schema.

        Args:
            messages: List of chat messages.
            model: Target LLM model name.
            response_model: Optional Pydantic model for schema validation.
            temperature: Sampling temperature (default 0.0 for strict JSON).

        Returns:
            Parsed JSON dictionary conforming to response schema.
        """
        target_model = model or settings.eval_agent_model
        params: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }

        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**params)
                content = response.choices[0].message.content or "{}"
                # Clean up any potential markdown code blocks
                clean_content = content.strip()
                if clean_content.startswith("```json"):
                    clean_content = clean_content[7:]
                if clean_content.startswith("```"):
                    clean_content = clean_content[3:]
                if clean_content.endswith("```"):
                    clean_content = clean_content[:-3]
                clean_content = clean_content.strip()

                parsed_json = json.loads(clean_content)

                if response_model:
                    validated = response_model.model_validate(parsed_json)
                    return validated.model_dump()

                return parsed_json
            except (RateLimitError, InternalServerError, APIConnectionError) as e:
                last_exception = e
                delay = self.base_delay * (self.backoff_factor**attempt)
                time.sleep(delay)
            except json.JSONDecodeError as e:
                last_exception = e
                logger.warning(f"Invalid JSON returned by LLM on attempt {attempt + 1}: {e}")
                time.sleep(self.base_delay)
            except Exception as e:
                raise e

        if last_exception:
            raise last_exception
        return {}
