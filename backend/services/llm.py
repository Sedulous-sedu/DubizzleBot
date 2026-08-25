"""LiteLLM integration service wrapper for model calls and guardrails."""

import litellm
from backend.config import settings

class LLMService:
    """Service interfacing with LiteLLM for intent recognition, guardrails, and agent responses."""

    def __init__(self, api_key: str = settings.GEMINI_API_KEY):
        self.api_key = api_key

    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        """Skeleton method for generating LLM completion."""
        # Functionality will be implemented in next phase
        raise NotImplementedError("LLM response generation will be implemented in subsequent task.")
