"""Chat request and response schemas."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class ChatRequest(BaseModel):
    """Payload schema for user chat requests."""
    user_id: str = Field(..., description="Unique identifier for user session / user profile")
    message: str = Field(..., description="User prompt or query message")
    session_id: Optional[str] = Field(None, description="Optional session identifier for short-term memory")

class ChatResponse(BaseModel):
    """Payload schema for agent chat responses."""
    user_id: str
    session_id: str
    response: str
    matched_cars: Optional[List[Dict[str, Any]]] = None
    intent: Optional[str] = None
