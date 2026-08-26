"""Data models for short-term conversation session memory and contextual reference resolution."""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from backend.models.car import CarListing
from backend.models.intent import (
    ParsedInventoryQuery,
    UnsupportedConstraint,
    UserIntentEnum,
)
from backend.models.booking import BookingDraft
from backend.models.lead import LeadDraft

class ResolutionStatus(str, Enum):
    """Status indicating how a user message was processed by ContextResolver."""
    NOT_CONTEXTUAL = "not_contextual"
    RESOLVED = "resolved"
    CLARIFICATION_REQUIRED = "clarification_required"

class TargetAttribute(str, Enum):
    """Supported vehicle attributes for follow-up inquiries."""
    MILEAGE = "mileage"
    WARRANTY = "warranty"
    PRICE = "price"
    MONTHLY_PAYMENT = "monthly_payment"
    REGIONAL_SPECS = "regional_specs"
    YEAR = "year"
    MAKE_MODEL_TRIM = "make_model_trim"
    BODY_TYPE = "body_type"
    ALL_DETAILS = "all_details"

class ContextResolutionResult(BaseModel):
    """Structured outcome produced by ContextResolver."""
    status: ResolutionStatus
    resolved_car: Optional[CarListing] = None
    target_attribute: Optional[TargetAttribute] = None
    clarification_message: Optional[str] = None
    raw_query: str

class ConversationTurn(BaseModel):
    """Represents a single conversational request-response exchange."""
    turn_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    user_message: str
    assistant_response: str
    intent: UserIntentEnum
    matched_listing_ids: List[int] = Field(default_factory=list)
    referenced_listing_id: Optional[int] = None

class PendingSupportedSearch(BaseModel):
    """Supported filters awaiting consent after unsupported constraints were rejected."""
    query_filters: ParsedInventoryQuery
    unsupported_constraints: List[UnsupportedConstraint] = Field(default_factory=list)

class SessionState(BaseModel):
    """Maintains in-memory short-term conversational context for an active session."""
    session_id: str
    user_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    turns: List[ConversationTurn] = Field(default_factory=list)
    current_result_set: List[CarListing] = Field(default_factory=list)
    active_listing_id: Optional[int] = None
    pending_supported_search: Optional[PendingSupportedSearch] = None
    pending_booking: Optional[BookingDraft] = None
    pending_lead: Optional[LeadDraft] = None
