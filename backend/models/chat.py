"""Chat request and response schemas for DubizzleBot."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from backend.models.car import CarListing
from backend.models.intent import UserIntentEnum

class ChatRequest(BaseModel):
    """Payload schema for user chat requests."""
    user_id: str = Field(..., min_length=1, description="Unique identifier for user session / user profile")
    message: str = Field(..., min_length=1, description="User prompt or query message")
    session_id: Optional[str] = Field(None, description="Optional session identifier for short-term memory")

    @field_validator("user_id", "message", mode="after")
    @classmethod
    def validate_non_empty_strings(cls, v: str) -> str:
        """Reject empty or whitespace-only inputs."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or whitespace only")
        return v.strip()

class ChatResponse(BaseModel):
    """Payload schema for agent chat responses."""
    user_id: str = Field(..., description="Echoed user identifier")
    session_id: str = Field(..., description="Supplied or newly generated session UUID")
    response: str = Field(..., description="Factual, grounded user-facing prose response")
    matched_cars: Optional[List[CarListing]] = Field(None, description="Exact matching car listing objects")
    intent: Optional[UserIntentEnum] = Field(None, description="Classified user intent")
    total_matches: int = Field(0, description="Total count of verifiable matching vehicles")
    requires_clarification: bool = Field(False, description="Flag indicating if user input was ambiguous")
