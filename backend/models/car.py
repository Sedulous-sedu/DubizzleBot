"""Car listing filter and representation schemas."""

from pydantic import BaseModel, Field
from typing import Optional

class CarFilter(BaseModel):
    """Filter parameter parameters for querying car inventory dataset."""
    make: Optional[str] = Field(None, description="Vehicle make (e.g. Honda, Toyota)")
    model: Optional[str] = Field(None, description="Vehicle model")
    min_year: Optional[int] = Field(None, description="Minimum manufacturing year")
    max_year: Optional[int] = Field(None, description="Maximum manufacturing year")
    max_price: Optional[float] = Field(None, description="Maximum price limit")
    min_price: Optional[float] = Field(None, description="Minimum price limit")
    body_type: Optional[str] = Field(None, description="Vehicle body type (e.g. SUV, Sedan)")

class CarListing(BaseModel):
    """Structured model representing a car listing from dataset."""
    id: Optional[str] = None
    make: str
    model: str
    trim: Optional[str] = None
    year: int
    price: float
    description: Optional[str] = None
