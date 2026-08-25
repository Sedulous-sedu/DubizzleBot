"""Lead qualification and test drive slot booking schemas."""

from pydantic import BaseModel, Field
from typing import Optional

class LeadQualification(BaseModel):
    """Schema for lead qualification and viewing slot recording."""
    user_id: str
    name: Optional[str] = None
    phone: Optional[str] = None
    price_range: Optional[str] = None
    preferred_car_id: Optional[str] = None
    slot_datetime: Optional[str] = Field(None, description="Booked viewing slot (Mon-Sat 8am-8pm)")
    notes: Optional[str] = None
