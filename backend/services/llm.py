"""Generic LLM transport service wrapping LiteLLM with structured schema support, timeouts, and rate-limit backoff."""

import json
import time
from typing import List, Dict, Any, Type, Optional, TypeVar
from pydantic import BaseModel
import litellm
from backend.config import settings

T = TypeVar("T", bound=BaseModel)

class LLMService:
    """Generic LLM provider transport service interfacing with LiteLLM."""

    def __init__(
        self,
        api_key: str = settings.GEMINI_API_KEY,
        model: str = settings.LLM_MODEL,
        temperature: float = settings.LLM_TEMPERATURE,
        timeout: float = settings.LLM_TIMEOUT_SECONDS,
        max_retries: int = 3,
        retry_delay_seconds: float = 5.0,
        mock_handler: Optional[Any] = None
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.mock_handler = mock_handler

    def generate_structured_completion(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[T]
    ) -> T:
        """
        Executes an LLM completion returning a strictly validated Pydantic model instance.
        If a mock handler is configured, delegates directly to mock without network calls.
        Implements automatic backoff retries for rate-limit exceptions.
        """
        if self.mock_handler is not None:
            return self.mock_handler(messages, response_model)

        json_schema = response_model.model_json_schema()
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": json_schema,
                "strict": True
            }
        }

        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = litellm.completion(
                    model=self.model,
                    messages=messages,
                    api_key=self.api_key if self.api_key else None,
                    temperature=self.temperature,
                    timeout=self.timeout,
                    response_format=response_format
                )
                content = response.choices[0].message.content
                if isinstance(content, str):
                    data = json.loads(content)
                else:
                    data = content
                return response_model.model_validate(data)
            except (litellm.exceptions.RateLimitError, litellm.exceptions.APIConnectionError, litellm.exceptions.Timeout) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay_seconds * (attempt + 1))
                else:
                    raise RuntimeError(f"LiteLLM structured completion failed after {self.max_retries} attempts: {e}") from e
            except Exception as e:
                # Other exceptions fail fast
                raise RuntimeError(f"LiteLLM structured completion failed: {e}") from e

        if last_exception:
            raise RuntimeError(f"LiteLLM structured completion failed: {last_exception}") from last_exception
