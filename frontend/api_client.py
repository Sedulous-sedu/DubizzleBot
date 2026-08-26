"""HTTP API Client for interacting with DubizzleBot FastAPI backend."""

import logging
from contextlib import nullcontext
from typing import Optional, List, Dict, Any
import httpx
from pydantic import BaseModel, Field, ValidationError

from frontend.config import BACKEND_URL

logger = logging.getLogger(__name__)

class DubizzleAPIError(Exception):
    """User-facing exception for API client communication errors."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

class FrontendCarListing(BaseModel):
    """Pydantic model representing a verified inventory car listing matching backend contract."""
    listing_id: int
    year: int
    make: str
    model: str
    trim: Optional[str] = None
    title: str
    description: Optional[str] = None
    photo_url: Optional[str] = None
    price_aed: Optional[float] = None
    monthly_payment_aed: Optional[float] = None
    mileage_km: Optional[float] = None
    regional_specs: Optional[str] = None
    has_positive_warranty: Optional[bool] = None
    warranty_status: Optional[str] = None
    body_type: Optional[str] = None
    provenance: Optional[Dict[str, Any]] = None

class FrontendChatResponse(BaseModel):
    """Pydantic model representing the backend /chat response payload."""
    user_id: str
    session_id: str
    response: str
    matched_cars: Optional[List[FrontendCarListing]] = None
    intent: str
    total_matches: int = 0
    requires_clarification: bool = False

class DubizzleAPIClient:
    """Pure HTTP client encapsulation for communicating with FastAPI backend."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        transport: Optional[httpx.BaseTransport] = None,
        client: Optional[httpx.Client] = None,
    ):
        self.base_url = (base_url or BACKEND_URL).rstrip("/")
        self.transport = transport
        self.client = client

    def _get_client_context(self, timeout: float):
        if self.client is not None:
            return nullcontext(self.client)
        return httpx.Client(transport=self.transport, timeout=timeout)

    def health_check(self, timeout: float = 3.0) -> bool:
        """
        Checks backend availability via GET /health.
        Returns True if server returns 200 OK with status='ok', False otherwise.
        """
        url = f"{self.base_url}/health"
        try:
            with self._get_client_context(timeout=timeout) as client:
                res = client.get(url)
                if res.status_code == 200:
                    data = res.json()
                    return data.get("status") == "ok"
                return False
        except Exception as e:
            logger.debug(f"Health check failed for {url}: {e}")
            return False

    def send_chat(
        self,
        user_id: str,
        session_id: str,
        message: str,
        timeout: float = 30.0,
    ) -> FrontendChatResponse:
        """
        Sends a conversational message to the backend via POST /chat.
        Enforces single-shot execution with zero automatic retries to prevent duplicate stateful mutations.
        """
        clean_user_id = str(user_id).strip()
        clean_session_id = str(session_id).strip()
        clean_message = str(message).strip()

        if not clean_user_id:
            raise DubizzleAPIError("User ID cannot be empty.")
        if not clean_session_id:
            raise DubizzleAPIError("Session ID cannot be empty.")
        if not clean_message:
            raise DubizzleAPIError("Message cannot be empty.")

        url = f"{self.base_url}/chat"
        payload = {
            "user_id": clean_user_id,
            "session_id": clean_session_id,
            "message": clean_message,
        }

        try:
            with self._get_client_context(timeout=timeout) as client:
                response = client.post(url, json=payload)
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            logger.warning(f"Connection failure to backend at {url}: {e}")
            raise DubizzleAPIError(
                f"Could not connect to DubizzleBot backend at {self.base_url}. "
                "Please ensure the backend server is running."
            )
        except httpx.ReadTimeout as e:
            logger.warning(f"Read timeout from backend at {url}: {e}")
            raise DubizzleAPIError(
                "DubizzleBot couldn't confirm whether that request completed in time. "
                "Please check the conversation state before submitting again."
            )
        except httpx.RequestError as e:
            logger.warning(f"HTTP request error for {url}: {e}")
            raise DubizzleAPIError(f"Network error communicating with backend: {e}")

        if response.status_code != 200:
            logger.warning(f"Non-200 response from backend ({response.status_code}): {response.text}")
            raise DubizzleAPIError(
                f"Backend service returned an error ({response.status_code}). Please try again.",
                status_code=response.status_code,
            )

        try:
            raw_data = response.json()
        except Exception as e:
            logger.error(f"Malformed JSON response from backend: {e}")
            raise DubizzleAPIError("Received malformed response from backend service.")

        try:
            parsed = FrontendChatResponse.model_validate(raw_data)
        except ValidationError as e:
            logger.error(f"Response validation error against schema: {e}")
            raise DubizzleAPIError("Backend response schema mismatch.")

        # Identity consistency validation
        if parsed.user_id != clean_user_id or parsed.session_id != clean_session_id:
            logger.error(
                f"Identity mismatch in response. Expected ({clean_user_id}, {clean_session_id}), "
                f"got ({parsed.user_id}, {parsed.session_id})"
            )
            raise DubizzleAPIError("Inconsistent session identity returned from backend.")

        return parsed
