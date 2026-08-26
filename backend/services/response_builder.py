"""Deterministic grounded response builder for DubizzleBot chat responses."""

import re
from typing import List, Optional, Union, Dict, Any
from backend.models.car import CarListing
from backend.models.intent import (
    ParsedInventoryQuery,
    UnsupportedConstraint,
)
from backend.models.memory import TargetAttribute

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

    @classmethod
    def format_vehicle_attribute_response(
        cls,
        item: Union[CarListing, Dict[str, Any]],
        target_attribute: TargetAttribute
    ) -> str:
        """Deterministically formats an answer for a specific attribute of a resolved vehicle."""
        car = cls._to_car_listing(item)
        car_name = f"{car.year} {car.make} {car.model}"
        car_ref = f"{car_name} (Listing #{car.listing_id})"

        if target_attribute == TargetAttribute.MILEAGE:
            if car.mileage_km is not None:
                return f"The {car_ref} has {car.mileage_km:,} km on the odometer."
            return f"The mileage is not stated in the listing for the {car_ref}."

        elif target_attribute == TargetAttribute.WARRANTY:
            if car.warranty_status:
                if car.warranty_status.lower() == "under warranty" or car.has_positive_warranty is True:
                    return f"Yes, the {car_ref} is listed with warranty status: {car.warranty_status}."
                return f"The {car_ref} has warranty status: {car.warranty_status}."
            elif car.has_positive_warranty is False:
                return f"The {car_ref} does not have an active warranty."
            return f"The warranty status is not stated in the listing for the {car_ref}."

        elif target_attribute == TargetAttribute.PRICE:
            if car.price_aed is not None:
                return f"The cash price for the {car_ref} is AED {car.price_aed:,.0f}."
            return f"The price is not stated in the listing for the {car_ref}."

        elif target_attribute == TargetAttribute.MONTHLY_PAYMENT:
            if car.monthly_payment_aed is not None:
                return f"The estimated installment for the {car_ref} is AED {car.monthly_payment_aed:,.0f} per month."
            return f"A monthly installment estimate is not stated in the listing for the {car_ref}."

        elif target_attribute == TargetAttribute.REGIONAL_SPECS:
            if car.regional_specs:
                return f"The regional specification for the {car_ref} is {car.regional_specs}."
            return f"The regional specification is not stated in the listing for the {car_ref}."

        elif target_attribute == TargetAttribute.YEAR:
            return f"The {car.make} {car.model} (Listing #{car.listing_id}) is a {car.year} model."

        elif target_attribute == TargetAttribute.BODY_TYPE:
            if car.body_type:
                return f"The body type for the {car_ref} is {car.body_type}."
            return f"The body type is not stated in the listing for the {car_ref}."

        elif target_attribute == TargetAttribute.MAKE_MODEL_TRIM:
            trim_str = f" {car.trim}" if car.trim and car.trim.lower() != "nan" else ""
            return f"This vehicle is a {car.year} {car.make} {car.model}{trim_str} (Listing #{car.listing_id})."

        else:  # ALL_DETAILS
            return f"Here are the details for the {car_ref}:\n\n{cls._format_listing_line(car)}"

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

    # =========================================================================
    # PHASE 4B: LONG-TERM PERSISTENT MEMORY FORMATTERS
    # =========================================================================

    @classmethod
    def format_saved_car_confirmation(cls, item: Union[CarListing, Dict[str, Any]]) -> str:
        """Formats factual confirmation when a car is saved to favorites."""
        listing = cls._to_car_listing(item)
        model_str = f" {listing.model}" if listing.model else ""
        return f"I've saved the {listing.year} {listing.make}{model_str} (Listing #{listing.listing_id}) to your favorites."

    @classmethod
    def format_removed_car_confirmation(cls, listing_id: int, item: Optional[Union[CarListing, Dict[str, Any]]] = None) -> str:
        """Formats confirmation when a car is removed from favorites."""
        if item:
            listing = cls._to_car_listing(item)
            model_str = f" {listing.model}" if listing.model else ""
            return f"I've removed the {listing.year} {listing.make}{model_str} (Listing #{listing_id}) from your favorites."
        return f"I've removed Listing #{listing_id} from your favorites."

    @classmethod
    def format_liked_cars_response(
        cls,
        saved_cars: List[Union[CarListing, Dict[str, Any]]],
        missing_ids: Optional[List[int]] = None
    ) -> str:
        """Formats verified saved vehicles from inventory rehydration."""
        missing = missing_ids or []
        if not saved_cars and not missing:
            return (
                "You don't have any saved cars in your favorites yet. "
                "You can save any car from your search results by saying 'Save this car' or 'I like the first one'."
            )

        lines = []
        if saved_cars:
            count = len(saved_cars)
            if count == 1:
                lines.append("You have 1 saved vehicle in your favorites:")
            else:
                lines.append(f"You have {count} saved vehicles in your favorites:")

            for car in saved_cars:
                lines.append(cls._format_listing_line(car))

        if missing:
            missing_str = ", ".join(f"#{i}" for i in missing)
            lines.append(f"Note: Saved Listing {missing_str} is no longer available in our active inventory.")

        return "\n\n".join(lines)

    @classmethod
    def format_preferences_summary_response(
        cls,
        prefs: Optional[Any],
        liked_count: int = 0
    ) -> str:
        """Formats a transparent summary clearly distinguishing explicit preferences from last search criteria."""
        has_prefs = prefs is not None and prefs.has_explicit_preferences()
        has_last_search = prefs is not None and prefs.last_search_filters is not None
        has_likes = liked_count > 0

        if not has_prefs and not has_last_search and not has_likes:
            return (
                "I don't have any saved preferences or search history for you yet. "
                "You can tell me your preferences like 'I prefer GCC cars' or 'My budget is AED 100,000', "
                "or save vehicles by saying 'Save this car'."
            )

        sections = ["Here is what I remember about you:"]

        # 1. Explicit Preferences
        if has_prefs:
            pref_items = []
            if prefs.preferred_make:
                pref_items.append(f"Make: {prefs.preferred_make}")
            if prefs.preferred_model:
                pref_items.append(f"Model: {prefs.preferred_model}")
            if prefs.min_year and prefs.max_year:
                pref_items.append(f"Year: {prefs.min_year} - {prefs.max_year}")
            elif prefs.min_year:
                pref_items.append(f"Year: from {prefs.min_year}")
            elif prefs.max_year:
                pref_items.append(f"Year: up to {prefs.max_year}")

            if prefs.min_price_aed and prefs.max_price_aed:
                pref_items.append(f"Budget: AED {prefs.min_price_aed:,.0f} - AED {prefs.max_price_aed:,.0f}")
            elif prefs.max_price_aed:
                pref_items.append(f"Budget: up to AED {prefs.max_price_aed:,.0f}")
            elif prefs.min_price_aed:
                pref_items.append(f"Budget: from AED {prefs.min_price_aed:,.0f}")

            if prefs.max_mileage_km:
                pref_items.append(f"Mileage: under {prefs.max_mileage_km:,} km")
            if prefs.regional_specs:
                pref_items.append(f"Regional Specs: {prefs.regional_specs.upper()}")
            if prefs.warranty_preference is True:
                pref_items.append("Warranty: Preferred / Required")
            elif prefs.warranty_preference is False:
                pref_items.append("Warranty: Not required")
            if prefs.keywords:
                pref_items.append(f"Keywords: {prefs.keywords}")

            pref_str = "\n".join(f"  • {item}" for item in pref_items)
            sections.append(f"Saved Preferences:\n{pref_str}")

        # 2. Most Recent Search (Clearly labeled as history)
        if has_last_search:
            ls = prefs.last_search_filters
            ls_parts = []
            if ls.make:
                ls_parts.append(ls.make)
            if ls.model:
                ls_parts.append(ls.model)
            if ls.min_year and ls.max_year:
                ls_parts.append(f"({ls.min_year}-{ls.max_year})")
            elif ls.min_year:
                ls_parts.append(f"(from {ls.min_year})")
            elif ls.max_year:
                ls_parts.append(f"(up to {ls.max_year})")
            if ls.max_price_aed:
                ls_parts.append(f"under AED {ls.max_price_aed:,.0f}")
            if ls.regional_specs:
                ls_parts.append(f"{ls.regional_specs} specs")
            if ls.warranty is True:
                ls_parts.append("with warranty")

            summary_query = " ".join(ls_parts) if ls_parts else "general search"
            sections.append(f"Most Recent Search:\n  • {summary_query}")

        # 3. Saved Favorites Count
        if has_likes:
            car_word = "vehicle" if liked_count == 1 else "vehicles"
            sections.append(f"Saved Favorites:\n  • {liked_count} {car_word} in your favorites")

        return "\n\n".join(sections)

    @classmethod
    def format_preference_saved_confirmation(cls, prefs: Any) -> str:
        """Formats confirmation when user preferences are updated."""
        pref_items = []
        if prefs.preferred_make:
            pref_items.append(f"Make: {prefs.preferred_make}")
        if prefs.preferred_model:
            pref_items.append(f"Model: {prefs.preferred_model}")
        if prefs.min_year and prefs.max_year:
            pref_items.append(f"Year: {prefs.min_year} - {prefs.max_year}")
        elif prefs.min_year:
            pref_items.append(f"Year: from {prefs.min_year}")
        elif prefs.max_year:
            pref_items.append(f"Year: up to {prefs.max_year}")

        if prefs.min_price_aed and prefs.max_price_aed:
            pref_items.append(f"Budget: AED {prefs.min_price_aed:,.0f} - AED {prefs.max_price_aed:,.0f}")
        elif prefs.max_price_aed:
            pref_items.append(f"Budget: up to AED {prefs.max_price_aed:,.0f}")
        elif prefs.min_price_aed:
            pref_items.append(f"Budget: from AED {prefs.min_price_aed:,.0f}")

        if prefs.max_mileage_km:
            pref_items.append(f"Mileage: under {prefs.max_mileage_km:,} km")
        if prefs.regional_specs:
            pref_items.append(f"Regional Specs: {prefs.regional_specs.upper()}")
        if prefs.warranty_preference is True:
            pref_items.append("Warranty: Preferred")
        elif prefs.warranty_preference is False:
            pref_items.append("Warranty: Not required")
        if prefs.keywords:
            pref_items.append(f"Keywords: {prefs.keywords}")

        if pref_items:
            pref_str = "\n".join(f"• {item}" for item in pref_items)
            return f"I've updated your saved preferences:\n\n{pref_str}"
        return "I've updated your saved preferences."

    @staticmethod
    def format_clear_confirmation(target: str) -> str:
        """Formats confirmation when memory or preferences are cleared."""
        if target == "preferences":
            return "I've cleared your saved preferences."
        elif target == "liked_cars":
            return "I've cleared all your saved favorites."
        elif target == "all":
            return "I've cleared all your profile data, saved preferences, and favorites."
        return "I've updated your saved memory."

