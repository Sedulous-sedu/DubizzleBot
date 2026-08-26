"""Pydantic data models for Phase 5 test-drive and viewing booking workflows."""

import uuid
from datetime import datetime, date, time, timezone
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict

from backend.models.car import CarListing

class WorkflowStatus(str, Enum):
    """Lifecycle status for interactive multi-turn workflows."""
    COLLECTING = "collecting"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"

class BookingStatus(str, Enum):
    """Database status for persisted vehicle viewings."""
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"

class BookingDraft(BaseModel):
    """In-memory draft for an in-progress test-drive / viewing booking."""
    model_config = ConfigDict(frozen=False)

    booking_id: str = Field(default_factory=lambda: f"BK-{uuid.uuid4().hex[:6].upper()}")
    user_id: str
    session_id: str
    listing_id: Optional[int] = None
    target_car: Optional[CarListing] = None
    requested_date: Optional[date] = None
    requested_time: Optional[time] = None
    requested_date_str: Optional[str] = None
    requested_time_str: Optional[str] = None
    appointment_at: Optional[datetime] = None
    status: WorkflowStatus = WorkflowStatus.COLLECTING
    customer_name: Optional[str] = None
    customer_phone: Optional[str] = None
    customer_email: Optional[str] = None
    missing_fields: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ConfirmedBooking(BaseModel):
    """Persistent confirmed booking record stored in SQLite."""
    model_config = ConfigDict(frozen=True)

    booking_id: str
    user_id: str
    listing_id: int
    appointment_at: datetime
    created_at: datetime
    status: BookingStatus = BookingStatus.CONFIRMED
    customer_name: Optional[str] = None
    customer_phone: Optional[str] = None
    customer_email: Optional[str] = None

class BookingValidationResult(BaseModel):
    """Result of validating a requested booking appointment against business hours."""
    model_config = ConfigDict(frozen=True)

    is_valid: bool
    error_message: Optional[str] = None
    appointment_at: Optional[datetime] = None
