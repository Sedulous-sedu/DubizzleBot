"""Pydantic data models for Phase 4B long-term persistent user memory."""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Set, List, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict

from backend.models.car import CarFilter, CarListing
from backend.models.intent import ParsedInventoryQuery

class UserProfile(BaseModel):
    """Persistent user profile metadata model."""
    model_config = ConfigDict(frozen=True)

    user_id: str
    created_at: datetime
    updated_at: datetime
    last_seen_at: datetime

class PreferencePatch(BaseModel):
    """Patch-based model for granular, non-destructive user preference updates."""
    model_config = ConfigDict(frozen=True)

    preferred_make: Optional[str] = None
    preferred_model: Optional[str] = None
    min_year: Optional[int] = None
    max_year: Optional[int] = None
    min_price_aed: Optional[float] = None
    max_price_aed: Optional[float] = None
    min_mileage_km: Optional[int] = None
    max_mileage_km: Optional[int] = None
    min_monthly_payment: Optional[float] = None
    max_monthly_payment: Optional[float] = None
    regional_specs: Optional[str] = None
    warranty_preference: Optional[bool] = None
    keywords: Optional[str] = None
    clear_fields: Set[str] = Field(default_factory=set)

    def is_empty(self) -> bool:
        """Returns True if patch specifies no field updates or field clears."""
        return (
            self.preferred_make is None
            and self.preferred_model is None
            and self.min_year is None
            and self.max_year is None
            and self.min_price_aed is None
            and self.max_price_aed is None
            and self.min_mileage_km is None
            and self.max_mileage_km is None
            and self.min_monthly_payment is None
            and self.max_monthly_payment is None
            and self.regional_specs is None
            and self.warranty_preference is None
            and self.keywords is None
            and len(self.clear_fields) == 0
        )

class UserPreferences(BaseModel):
    """Persistent user search preferences and last search metadata."""
    model_config = ConfigDict(frozen=False)

    user_id: str
    preferred_make: Optional[str] = None
    preferred_model: Optional[str] = None
    min_year: Optional[int] = None
    max_year: Optional[int] = None
    min_price_aed: Optional[float] = None
    max_price_aed: Optional[float] = None
    min_mileage_km: Optional[int] = None
    max_mileage_km: Optional[int] = None
    min_monthly_payment: Optional[float] = None
    max_monthly_payment: Optional[float] = None
    regional_specs: Optional[str] = None
    warranty_preference: Optional[bool] = None
    keywords: Optional[str] = None
    last_search_filters: Optional[ParsedInventoryQuery] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def apply_patch(self, patch: PreferencePatch) -> "UserPreferences":
        """Applies a patch non-destructively: updates specified fields, clears marked fields, preserves others."""
        field_map = {
            "preferred_make": patch.preferred_make,
            "preferred_model": patch.preferred_model,
            "min_year": patch.min_year,
            "max_year": patch.max_year,
            "min_price_aed": patch.min_price_aed,
            "max_price_aed": patch.max_price_aed,
            "min_mileage_km": patch.min_mileage_km,
            "max_mileage_km": patch.max_mileage_km,
            "min_monthly_payment": patch.min_monthly_payment,
            "max_monthly_payment": patch.max_monthly_payment,
            "regional_specs": patch.regional_specs,
            "warranty_preference": patch.warranty_preference,
            "keywords": patch.keywords,
        }

        # Apply positive field updates
        for field_name, new_val in field_map.items():
            if new_val is not None:
                setattr(self, field_name, new_val)

        # Apply explicit field clears
        for clear_field in patch.clear_fields:
            if hasattr(self, clear_field):
                setattr(self, clear_field, None)

        self.updated_at = datetime.now(timezone.utc)
        return self

    def has_explicit_preferences(self) -> bool:
        """Returns True if any persistent explicit preference filter is set."""
        return any([
            self.preferred_make is not None,
            self.preferred_model is not None,
            self.min_year is not None,
            self.max_year is not None,
            self.min_price_aed is not None,
            self.max_price_aed is not None,
            self.min_mileage_km is not None,
            self.max_mileage_km is not None,
            self.min_monthly_payment is not None,
            self.max_monthly_payment is not None,
            self.regional_specs is not None,
            self.warranty_preference is not None,
            self.keywords is not None,
        ])

    def to_car_filter(self) -> Optional[CarFilter]:
        """Converts explicit non-null preferences into a deterministic CarFilter. Excludes last_search_filters."""
        if not self.has_explicit_preferences():
            return None

        # Build keywords list if keywords string is set
        kw_list = [k.strip() for k in self.keywords.split()] if self.keywords else None

        return CarFilter(
            make=self.preferred_make,
            model=self.preferred_model,
            min_year=self.min_year,
            max_year=self.max_year,
            min_price_aed=self.min_price_aed,
            max_price_aed=self.max_price_aed,
            min_mileage_km=self.min_mileage_km,
            max_mileage_km=self.max_mileage_km,
            min_monthly_payment=self.min_monthly_payment,
            max_monthly_payment=self.max_monthly_payment,
            regional_specs=self.regional_specs,
            warranty=self.warranty_preference,
            keywords=kw_list,
        )

class LikedCarRecord(BaseModel):
    """Record of a vehicle explicitly liked/saved by a user."""
    model_config = ConfigDict(frozen=True)

    user_id: str
    listing_id: int
    liked_at: datetime

class LongTermMemoryAction(str, Enum):
    """Categorized actions for long-term memory operations."""
    NOT_MEMORY_ACTION = "not_memory_action"
    SAVE_LIKED_CAR = "save_liked_car"
    RECALL_LIKED_CARS = "recall_liked_cars"
    REMOVE_LIKED_CAR = "remove_liked_car"
    SAVE_PREFERENCE = "save_preference"
    RECALL_MEMORY = "recall_memory"
    CLEAR_PREFERENCES = "clear_preferences"
    CLEAR_LIKED_CARS = "clear_liked_cars"
    CLEAR_ALL_MEMORY = "clear_all_memory"
    SEARCH_SAVED_PREFERENCES = "search_saved_preferences"

class LongTermMemoryResolution(BaseModel):
    """Result of evaluating a message for long-term memory operations."""
    model_config = ConfigDict(frozen=True)

    action: LongTermMemoryAction
    target_car: Optional[CarListing] = None
    target_listing_id: Optional[int] = None
    preference_patch: Optional[PreferencePatch] = None
    clarification_message: Optional[str] = None
    search_query_override: Optional[Union[str, ParsedInventoryQuery]] = None
