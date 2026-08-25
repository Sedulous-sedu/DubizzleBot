"""Car listing filter and representation schemas for DubizzleBot."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class CarFilter(BaseModel):
    """Filter parameters for deterministic car inventory queries."""
    make: Optional[str] = Field(None, description="Vehicle make (case-insensitive)")
    model: Optional[str] = Field(None, description="Vehicle model (case-insensitive)")
    min_year: Optional[int] = Field(None, description="Minimum manufacturing year")
    max_year: Optional[int] = Field(None, description="Maximum manufacturing year")
    min_price_aed: Optional[float] = Field(None, description="Minimum cash price limit in AED")
    max_price_aed: Optional[float] = Field(None, description="Maximum cash price limit in AED")
    min_price: Optional[float] = Field(None, description="Alias for min_price_aed")
    max_price: Optional[float] = Field(None, description="Alias for max_price_aed")
    min_mileage_km: Optional[int] = Field(None, description="Minimum odometer reading in KM")
    max_mileage_km: Optional[int] = Field(None, description="Maximum odometer reading in KM")
    min_mileage: Optional[int] = Field(None, description="Alias for min_mileage_km")
    max_mileage: Optional[int] = Field(None, description="Alias for max_mileage_km")
    min_monthly_aed: Optional[float] = Field(None, description="Minimum monthly payment in AED")
    max_monthly_aed: Optional[float] = Field(None, description="Maximum monthly payment in AED")
    min_monthly_payment: Optional[float] = Field(None, description="Alias for min_monthly_aed")
    max_monthly_payment: Optional[float] = Field(None, description="Alias for max_monthly_aed")
    regional_specs: Optional[str] = Field(None, description="Regional specification tag (e.g. GCC, USA, Japanese, Korean)")
    warranty: Optional[bool] = Field(None, description="True for positive active warranty, False for no/expired warranty")
    keywords: Optional[str] = Field(None, description="Free-text keywords matching title, description, make, model, trim")
    limit: Optional[int] = Field(None, description="Maximum number of listings to return")

class CarListing(BaseModel):
    """Structured model representing a car listing with original display fields and derived attributes."""
    # Original source fields (unmodified display values preserved)
    listing_id: int = Field(..., description="Unique listing identifier from dataset")
    year: int = Field(..., description="Manufacturing year")
    make: str = Field(..., description="Original vehicle make string")
    model: str = Field(..., description="Original vehicle model string")
    trim: Optional[str] = Field(None, description="Original trim or variant string")
    title: str = Field(..., description="Original listing title string")
    description: str = Field(..., description="Original listing description text")
    photo_url: Optional[str] = Field(None, description="Image CDN URL")

    # Normalized / Derived attributes (Missing facts are None)
    price_aed: Optional[float] = Field(None, description="Extracted cash price in AED")
    monthly_payment_aed: Optional[float] = Field(None, description="Extracted monthly payment rate in AED")
    mileage_km: Optional[int] = Field(None, description="Extracted odometer reading in KM")
    regional_specs: Optional[str] = Field(None, description="Standardized regional spec tag (GCC, USA, Japanese, Korean, European, etc.)")
    has_positive_warranty: Optional[bool] = Field(None, description="True if explicit active warranty exists")
    warranty_status: Optional[str] = Field(None, description="Categorized warranty status string")
    body_type: Optional[str] = Field(None, description="Explicitly grounded body type")

    # Provenance tracing
    provenance: Optional[Dict[str, Any]] = Field(None, description="Source text snippets for derived values")
