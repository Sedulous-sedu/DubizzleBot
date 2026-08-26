"""Pydantic data models for Phase 5 lead qualification and CSV capture."""

import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict
from pydantic import BaseModel, Field, ConfigDict

from backend.models.booking import WorkflowStatus

class LeadDraft(BaseModel):
    """In-memory draft of lead information collected across conversational turns."""
    model_config = ConfigDict(frozen=False)

    lead_id: str = Field(default_factory=lambda: f"LEAD-{uuid.uuid4().hex[:6].upper()}")
    user_id: str
    session_id: str
    name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    min_budget_aed: Optional[float] = None
    max_budget_aed: Optional[float] = None
    interested_make: Optional[str] = None
    interested_model: Optional[str] = None
    interested_listing_id: Optional[int] = None
    requirements: Optional[str] = None
    booking_reference: Optional[str] = None
    status: WorkflowStatus = WorkflowStatus.COLLECTING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def has_contact(self) -> bool:
        """Returns True if at least one valid phone or email contact method is present."""
        return bool((self.phone and self.phone.strip()) or (self.email and self.email.strip()))

    def has_budget(self) -> bool:
        """Returns True if at least one budget boundary is defined."""
        return self.min_budget_aed is not None or self.max_budget_aed is not None

    def has_automotive_need(self) -> bool:
        """Returns True if a specific listing ID, make/model, or explicit requirement is set."""
        return bool(
            self.interested_listing_id is not None
            or (self.interested_make and self.interested_make.strip())
            or (self.interested_model and self.interested_model.strip())
            or (self.requirements and self.requirements.strip())
        )

    def is_fully_qualified(self) -> bool:
        """Enforces assessment qualification criteria: Contact + Budget + Automotive Need."""
        return self.has_contact() and self.has_budget() and self.has_automotive_need()

class QualifiedLead(BaseModel):
    """Finalized qualified lead record matching CSV columns."""
    model_config = ConfigDict(frozen=True)

    lead_id: str
    created_at: str
    user_id: str
    session_id: str
    name: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    min_budget_aed: Optional[float] = None
    max_budget_aed: Optional[float] = None
    interested_make: Optional[str] = None
    interested_model: Optional[str] = None
    interested_listing_id: Optional[int] = None
    requirements: Optional[str] = None
    booking_reference: Optional[str] = None

    def to_csv_dict(self) -> Dict[str, str]:
        """Serializes qualified lead to a flat string dictionary matching CSV header columns."""
        return {
            "lead_id": self.lead_id,
            "created_at": self.created_at,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "name": self.name or "",
            "phone": self.phone or "",
            "email": self.email or "",
            "min_budget_aed": f"{self.min_budget_aed:,.0f}" if self.min_budget_aed is not None else "",
            "max_budget_aed": f"{self.max_budget_aed:,.0f}" if self.max_budget_aed is not None else "",
            "interested_make": self.interested_make or "",
            "interested_model": self.interested_model or "",
            "interested_listing_id": str(self.interested_listing_id) if self.interested_listing_id is not None else "",
            "requirements": self.requirements or "",
            "booking_reference": self.booking_reference or "",
        }
