"""Pydantic schemas and enums for natural-language intent interpretation."""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, model_validator
from backend.models.car import CarFilter

class UserIntentEnum(str, Enum):
    """Primary intent categories for user queries."""
    INVENTORY_SEARCH = "inventory_search"
    VIEWING_OR_LEAD_REQUEST = "viewing_or_lead_request"
    GENERAL_CHAT = "general_chat"
    UNKNOWN = "unknown"

class RegionalSpecEnum(str, Enum):
    """Constrained regional vehicle specifications matching Phase 2 taxonomy."""
    GCC = "GCC"
    USA = "USA"
    JAPANESE = "Japanese"
    KOREAN = "Korean"
    EUROPEAN = "European"
    CANADIAN = "Canadian"
    UK = "UK"
    RUSSIAN = "Russian"
    SINGAPORE = "Singapore"
    OTHER = "Other"
    CUSTOM = "Custom"

class SearchReadinessState(str, Enum):
    """Deterministic readiness state indicating how Phase 3B should handle search execution."""
    READY = "ready"
    CLARIFICATION_REQUIRED = "clarification_required"
    UNSUPPORTED_CONSTRAINTS_PRESENT = "unsupported_constraints_present"
    NON_INVENTORY_INTENT = "non_inventory_intent"

class UnsupportedConstraint(BaseModel):
    """Structured representation of requested user constraints not supported by deterministic inventory filtering."""
    model_config = ConfigDict(extra="forbid")

    field: str = Field(..., description="Target attribute name (e.g. color, transmission, fuel_type, ranking, seats)")
    requested_value: str = Field(..., description="Value or condition requested by user (e.g. red, automatic, cheapest)")
    reason: str = Field(..., description="Explanation why constraint cannot be deterministically filtered")

class ParsedInventoryQuery(BaseModel):
    """Canonical Phase 2 inventory search filters extracted from natural language."""
    model_config = ConfigDict(extra="forbid")

    make: Optional[str] = Field(None, description="Vehicle manufacturer brand (e.g. Bentley, Ford)")
    model: Optional[str] = Field(None, description="Vehicle model name (e.g. Bentayga, Explorer)")
    min_year: Optional[int] = Field(None, ge=1900, le=2100, description="Minimum manufacturing year")
    max_year: Optional[int] = Field(None, ge=1900, le=2100, description="Maximum manufacturing year")
    min_price_aed: Optional[float] = Field(None, ge=0.0, description="Explicit minimum cash price in AED")
    max_price_aed: Optional[float] = Field(None, ge=0.0, description="Explicit maximum cash price in AED")
    min_mileage_km: Optional[int] = Field(None, ge=0, description="Explicit minimum odometer reading in KM")
    max_mileage_km: Optional[int] = Field(None, ge=0, description="Explicit maximum odometer reading in KM")
    min_monthly_aed: Optional[float] = Field(None, ge=0.0, description="Explicit minimum monthly installment in AED")
    max_monthly_aed: Optional[float] = Field(None, ge=0.0, description="Explicit maximum monthly installment in AED")
    regional_specs: Optional[RegionalSpecEnum] = Field(None, description="Constrained regional specification tag")
    warranty: Optional[bool] = Field(None, description="True for active positive warranty, False for no/expired warranty")
    keywords: Optional[str] = Field(None, description="Free-text keywords matching title or description")
    limit: Optional[int] = Field(None, ge=1, le=100, description="Safe maximum result count")

    @model_validator(mode="after")
    def validate_ranges(self) -> 'ParsedInventoryQuery':
        """Ensure range minimums do not exceed range maximums."""
        if self.min_year is not None and self.max_year is not None and self.min_year > self.max_year:
            raise ValueError(f"min_year ({self.min_year}) cannot exceed max_year ({self.max_year})")
        if self.min_price_aed is not None and self.max_price_aed is not None and self.min_price_aed > self.max_price_aed:
            raise ValueError(f"min_price_aed ({self.min_price_aed}) cannot exceed max_price_aed ({self.max_price_aed})")
        if self.min_mileage_km is not None and self.max_mileage_km is not None and self.min_mileage_km > self.max_mileage_km:
            raise ValueError(f"min_mileage_km ({self.min_mileage_km}) cannot exceed max_mileage_km ({self.max_mileage_km})")
        if self.min_monthly_aed is not None and self.max_monthly_aed is not None and self.min_monthly_aed > self.max_monthly_aed:
            raise ValueError(f"min_monthly_aed ({self.min_monthly_aed}) cannot exceed max_monthly_aed ({self.max_monthly_aed})")
        return self

    def to_car_filter(self) -> CarFilter:
        """Converts validated query to canonical Phase 2 CarFilter."""
        return CarFilter(
            make=self.make,
            model=self.model,
            min_year=self.min_year,
            max_year=self.max_year,
            min_price_aed=self.min_price_aed,
            max_price_aed=self.max_price_aed,
            min_mileage_km=self.min_mileage_km,
            max_mileage_km=self.max_mileage_km,
            min_monthly_aed=self.min_monthly_aed,
            max_monthly_aed=self.max_monthly_aed,
            regional_specs=self.regional_specs.value if self.regional_specs else None,
            warranty=self.warranty,
            keywords=self.keywords,
            limit=self.limit
        )

class LLMIntentPayload(BaseModel):
    """Raw semantic JSON payload emitted by the LLM (readiness state is strictly not authorable by the model)."""
    model_config = ConfigDict(extra="forbid")

    intent: UserIntentEnum = Field(..., description="Primary classification of user intent")
    query_filters: Optional[ParsedInventoryQuery] = Field(None, description="Extracted canonical inventory query filters")
    requires_clarification: bool = Field(False, description="True if prompt is vague and requires clarification")
    clarification_question: Optional[str] = Field(None, description="Targeted clarification question")
    unsupported_constraints: List[UnsupportedConstraint] = Field(default_factory=list, description="Structured unhandled criteria")

class ParsedUserIntent(BaseModel):
    """Complete domain intent object with Python-derived deterministic search readiness state."""
    model_config = ConfigDict(extra="forbid")

    intent: UserIntentEnum = Field(..., description="Primary classification of user intent")
    query_filters: Optional[ParsedInventoryQuery] = Field(None, description="Extracted canonical inventory query filters")
    requires_clarification: bool = Field(False, description="True if prompt is vague and requires clarification")
    clarification_question: Optional[str] = Field(None, description="Targeted clarification question")
    unsupported_constraints: List[UnsupportedConstraint] = Field(default_factory=list, description="Structured unhandled criteria")
    readiness_state: SearchReadinessState = Field(..., description="Deterministic search readiness state computed by Python engine")

    @classmethod
    def from_payload(cls, payload: LLMIntentPayload) -> 'ParsedUserIntent':
        """Constructs a ParsedUserIntent and deterministically computes the readiness state in Python."""
        readiness = cls.derive_readiness_state(payload)
        return cls(
            intent=payload.intent,
            query_filters=payload.query_filters,
            requires_clarification=payload.requires_clarification,
            clarification_question=payload.clarification_question,
            unsupported_constraints=payload.unsupported_constraints,
            readiness_state=readiness
        )

    @staticmethod
    def derive_readiness_state(payload: LLMIntentPayload) -> SearchReadinessState:
        """
        Deterministically derive the search readiness state in Python based on validated semantic payload.
        The LLM has zero control over execution readiness.
        """
        if payload.intent not in (UserIntentEnum.INVENTORY_SEARCH, UserIntentEnum.VIEWING_OR_LEAD_REQUEST):
            return SearchReadinessState.NON_INVENTORY_INTENT
        if payload.requires_clarification:
            return SearchReadinessState.CLARIFICATION_REQUIRED
        if len(payload.unsupported_constraints) > 0:
            return SearchReadinessState.UNSUPPORTED_CONSTRAINTS_PRESENT
        return SearchReadinessState.READY
