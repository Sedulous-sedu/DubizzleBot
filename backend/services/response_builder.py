"""Deterministic grounded response builder for DubizzleBot chat responses."""

import re
from typing import List, Optional, Union, Dict, Any
from backend.models.car import CarListing
from backend.models.intent import (
    ParsedInventoryQuery,
    UnsupportedConstraint,
)

KNOWN_COMPETITORS = [
    "yallamotor", "carswitch", "kavak", "cars24", "opensooq", "seez",
    "autotrader", "cargurus", "carwale", "mobile.de", "olx"
]

class GroundedResponseBuilder:
    """Builder generating factual, grounded natural language responses directly from inventory records and intent state."""

    @classmethod
    def _to_car_listing(cls, item: Union[CarListing, Dict[str, Any]]) -> CarListing:
        """Converts raw dict or CarListing into validated CarListing instance."""
        if isinstance(item, CarListing):
            return item
        return CarListing.model_validate(item)

    @classmethod
    def _format_listing_line(cls, item: Union[CarListing, Dict[str, Any]]) -> str:
        """Formats a single car listing with verified attributes."""
        listing = cls._to_car_listing(item)
        title = f"• Listing #{listing.listing_id}: {listing.year} {listing.make} {listing.model}"
        if listing.trim and listing.trim.lower() != "nan":
            title += f" {listing.trim}"

        details = []
        if listing.price_aed is not None:
            details.append(f"Price: AED {listing.price_aed:,.0f}")
        else:
            details.append("Price: Not stated")

        if listing.monthly_payment_aed is not None:
            details.append(f"Monthly: AED {listing.monthly_payment_aed:,.0f}/mo")

        if listing.mileage_km is not None:
            details.append(f"Mileage: {listing.mileage_km:,} km")
        else:
            details.append("Mileage: Not stated")

        if listing.regional_specs:
            details.append(f"Specs: {listing.regional_specs}")
        else:
            details.append("Specs: Not stated")

        # Warranty formatting preserving Phase 2 exact semantics
        if listing.warranty_status:
            details.append(f"Warranty: {listing.warranty_status}")
        elif listing.has_positive_warranty is True:
            details.append("Warranty: Yes (Active Warranty)")
        elif listing.has_positive_warranty is False:
            details.append("Warranty: No / Not Active")

        if listing.body_type:
            details.append(f"Body: {listing.body_type}")

        details_str = " | ".join(details)
        return f"{title}\n  {details_str}"

    @classmethod
    def format_inventory_search_response(
        cls,
        listings: List[Union[CarListing, Dict[str, Any]]],
        total_count: int,
        query_filters: Optional[ParsedInventoryQuery] = None
    ) -> str:
        """Formats deterministic inventory search results."""
        if total_count == 0 or not listings:
            return (
                "I couldn't find any listings matching those exact criteria in our verified inventory. "
                "You might want to adjust your budget, year, or model preferences."
            )

        shown_count = min(len(listings), 5)
        if total_count == 1:
            header = "Found 1 verified matching vehicle in our inventory:"
        elif total_count <= shown_count:
            header = f"Found {total_count} verified matching vehicles in our inventory:"
        else:
            header = f"Found {total_count} verified matching vehicles in our inventory. Here are the first {shown_count} options:"

        lines = [header]
        for listing in listings[:shown_count]:
            lines.append(cls._format_listing_line(listing))

        return "\n\n".join(lines)

    @staticmethod
    def format_clarification_response(clarification_question: Optional[str]) -> str:
        """Returns the targeted clarification question without querying inventory."""
        if clarification_question and clarification_question.strip():
            return clarification_question.strip()
        return "Could you please specify your preferred budget, year, or vehicle make so I can find matching cars for you?"

    @staticmethod
    def format_unsupported_constraints_response(
        unsupported_constraints: List[UnsupportedConstraint],
        query_filters: Optional[ParsedInventoryQuery] = None
    ) -> str:
        """Explains unsupported constraints without exposing internal implementation details."""
        subject = "vehicles"
        if query_filters:
            parts = []
            if query_filters.make:
                parts.append(query_filters.make)
            if query_filters.model:
                parts.append(query_filters.model)
            if parts:
                subject = " ".join(parts)

        # Check if ranking is the unsupported constraint
        ranking_constraint = next((c for c in unsupported_constraints if c.field.lower() == "ranking"), None)
        if ranking_constraint:
            return (
                f"I can search for {subject}, but I can't reliably rank them by '{ranking_constraint.requested_value}' "
                f"with the current inventory data. Would you like me to search using the supported criteria only?"
            )

        # Other unsupported constraints
        reasons = [f"'{c.requested_value}' ({c.field})" for c in unsupported_constraints]
        reasons_str = ", ".join(reasons)
        return (
            f"I cannot deterministically filter by {reasons_str} in the inventory. "
            f"Would you like me to search for {subject} using the supported criteria only?"
        )

    @classmethod
    def format_viewing_response(
        cls,
        listings: Optional[List[Union[CarListing, Dict[str, Any]]]],
        total_count: int,
        query_filters: Optional[ParsedInventoryQuery] = None
    ) -> str:
        """Formats viewing / test-drive responses."""
        if listings is not None and len(listings) > 0:
            shown_count = min(len(listings), 5)
            if total_count == 1:
                header = (
                    "I can help arrange a viewing or test drive. Found 1 matching candidate vehicle in our inventory. "
                    "Please let me know which specific Listing ID you would like to view:"
                )
            elif total_count <= shown_count:
                header = (
                    f"I can help arrange a viewing or test drive. Found {total_count} matching candidate vehicles in our inventory. "
                    f"Please let me know which specific Listing ID you would like to view:"
                )
            else:
                header = (
                    f"I can help arrange a viewing or test drive. Found {total_count} matching candidate vehicles in our inventory. "
                    f"Here are the first {shown_count} options. Please let me know which specific Listing ID you would like to view:"
                )

            lines = [header]
            for listing in listings[:shown_count]:
                lines.append(cls._format_listing_line(listing))
            return "\n\n".join(lines)

        if listings is not None and total_count == 0:
            return (
                "I can help arrange a viewing or test drive, but I couldn't find any listings matching those exact criteria. "
                "Please let me know if you would like to view a different vehicle from our inventory."
            )

        return "I would be happy to help arrange a viewing or test drive. Which vehicle from our inventory are you interested in viewing?"

    @staticmethod
    def format_general_chat_response(message: str) -> str:
        """Handles greetings, capability questions, and competitor redirections."""
        msg_lower = message.lower()
        if any(re.search(rf"\b{re.escape(comp)}\b", msg_lower) for comp in KNOWN_COMPETITORS):
            return (
                "I specialize in dubizzle verified car inventory across the UAE. "
                "Let me know what make, model, or budget you're looking for and I'll find matching listings for you!"
            )

        return (
            "Hello! I am DubizzleBot, your assistant for searching verified used cars in the UAE and arranging viewings. "
            "How can I help you today? You can ask me to search by make, model, year, price, mileage, regional specs, or warranty!"
        )

    @staticmethod
    def format_unknown_response() -> str:
        """Polite domain guardrail refusal for non-automotive requests."""
        return (
            "I am DubizzleBot, specialized in helping you find cars and arrange viewings from our verified UAE inventory. "
            "I cannot assist with non-automotive topics, but I would be glad to help you find a car!"
        )
