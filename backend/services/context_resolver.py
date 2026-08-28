"""Deterministic Context & Reference Resolver for Short-Term Conversational Follow-Ups."""

import re
from typing import Optional, List, Tuple
from backend.models.car import CarListing
from backend.models.memory import (
    SessionState,
    ContextResolutionResult,
    ResolutionStatus,
    ResultSetComparisonType,
    TargetAttribute,
)

class ContextResolver:
    """
    Evaluates incoming user messages against active SessionState to determine
    if a message is a deterministic follow-up inquiring about a previously verified vehicle.
    """

    # Ordinal words to 0-based index (or -1 for last)
    ORDINAL_MAP = {
        "first": 0,
        "1st": 0,
        "top": 0,
        "second": 1,
        "2nd": 1,
        "third": 2,
        "3rd": 2,
        "fourth": 3,
        "4th": 3,
        "fifth": 4,
        "5th": 4,
        "last": -1,
    }

    # Attribute detection patterns
    ATTRIBUTE_PATTERNS = [
        (TargetAttribute.MILEAGE, r"\b(mileage|km|kms|kilometers?|odometer|how far|how many km|driven)\b"),
        (TargetAttribute.WARRANTY, r"\b(warranty|guarantee|warranties)\b"),
        (TargetAttribute.MONTHLY_PAYMENT, r"\b(monthly\s+payment|per\s+month|monthly|installment|emi)\b"),
        (TargetAttribute.PRICE, r"\b(price|cost|how\s+much|cash\s+price|aed)\b"),
        (TargetAttribute.REGIONAL_SPECS, r"\b(specs?|specifications?|regional(\s+specs?)?|gcc|imported)\b"),
        (TargetAttribute.YEAR, r"\b(year|model\s+year|how\s+old|what\s+year)\b"),
        (TargetAttribute.BODY_TYPE, r"\b(body(\s+type)?|suv|sedan|coupe|hatchback|convertible)\b"),
        (TargetAttribute.MAKE_MODEL_TRIM, r"\b(trim|version|exact\s+model)\b"),
        (TargetAttribute.ALL_DETAILS, r"\b(tell\s+me\s+more|details|more\s+info|summary|specs\s+and\s+price)\b"),
    ]

    # Patterns indicating a fresh search (imperative search phrases for multiple/plural cars)
    FRESH_SEARCH_INDICATORS = [
        r"^(show|find|search|look\s+for|list|give\s+me|get\s+me|i\s+want|i\s+need|i'm\s+looking\s+for)\s+(me\s+)?(all\s+)?(cars|vehicles|suvs|sedans|options|automobiles)\b",
        r"^(show|find|search|look\s+for|list|give\s+me|get\s+me|i\s+want|i\s+need|i'm\s+looking\s+for)\s+([a-zA-Z0-9_-]+\s+)?(cars|vehicles|suvs|sedans|options)\b",
        r"^(show|find|search|look\s+for|list|give\s+me|get\s+me|i\s+want|i\s+need)\s+(me\s+)?(any\s+)?[a-zA-Z0-9_-]+\s+(under|above|below|from|between|with|around)\b",
        r"^(show|find|search|look\s+for|list|give\s+me|get\s+me)\s+(me\s+)?(the\s+)?(\d+\s+)?(newest|latest|oldest|earliest|most\s+recent)\b",
        r"\b(under|below|less\s+than)\s+aed\s+\d+",
        r"\b(from\s+\d{4}\s+to\s+\d{4}|\d{4}\s+or\s+newer|\d{4}\s+or\s+older)\b",
    ]

    @classmethod
    def resolve(cls, message: str, session: SessionState) -> ContextResolutionResult:
        """
        Main entry point to evaluate if message is a contextual follow-up.
        Enforces strict routing precedence:
        A. Unmistakable fresh-search detection FIRST
        B. Contextual result-set comparison detection
        C. Ordinal / specific-car attribute handling
        D. Make/model reference handling
        E. Pronoun / deictic handling
        F. Fallback (NOT_CONTEXTUAL)
        """
        raw_msg = message.strip()
        msg_lower = raw_msg.lower()

        # A. Check if the query is an unmistakable fresh search
        if cls._is_unmistakable_fresh_search(msg_lower):
            return ContextResolutionResult(status=ResolutionStatus.NOT_CONTEXTUAL, raw_query=raw_msg)

        # B. Check for whole result-set comparison (e.g. "Which is the latest year model?", "Which is the oldest?")
        comparison_res = cls._match_result_set_comparison(msg_lower, session, raw_msg)
        if comparison_res is not None:
            return comparison_res

        # Detect requested attribute
        target_attr = cls._extract_target_attribute(msg_lower)

        # C. Check for Ordinal or Qualified Ordinal reference (e.g. "that first Honda", "the second one")
        ordinal_match = cls._match_ordinal_reference(msg_lower, session.current_result_set)
        if ordinal_match is not None:
            status, car, clarify_msg = ordinal_match
            if status == ResolutionStatus.RESOLVED:
                # If no attribute specified, default to ALL_DETAILS
                attr = target_attr or TargetAttribute.ALL_DETAILS
                return ContextResolutionResult(
                    status=ResolutionStatus.RESOLVED,
                    resolved_car=car,
                    target_attribute=attr,
                    raw_query=raw_msg
                )
            elif status == ResolutionStatus.CLARIFICATION_REQUIRED:
                return ContextResolutionResult(
                    status=ResolutionStatus.CLARIFICATION_REQUIRED,
                    clarification_message=clarify_msg,
                    raw_query=raw_msg
                )

        # D. Check for specific Make/Model reference without ordinal in current results (e.g. "What's the mileage on the Honda?")
        make_match = cls._match_make_reference(msg_lower, session.current_result_set)
        if make_match is not None:
            status, car, clarify_msg = make_match
            if status == ResolutionStatus.RESOLVED:
                attr = target_attr or TargetAttribute.ALL_DETAILS
                return ContextResolutionResult(
                    status=ResolutionStatus.RESOLVED,
                    resolved_car=car,
                    target_attribute=attr,
                    raw_query=raw_msg
                )
            elif status == ResolutionStatus.CLARIFICATION_REQUIRED:
                return ContextResolutionResult(
                    status=ResolutionStatus.CLARIFICATION_REQUIRED,
                    clarification_message=clarify_msg,
                    raw_query=raw_msg
                )

        # E. Check for Pronoun / Deictic reference ("it", "its", "that car", "this car", "this one", etc.)
        pronoun_match = cls._match_pronoun_reference(msg_lower, session)
        if pronoun_match is not None:
            status, car, clarify_msg = pronoun_match
            # Treat as contextual if asking about an attribute or general details/questions
            if target_attr is not None or any(w in msg_lower for w in ["tell me more", "about", "details", "info", "what is", "how is", "does it", "is it", "is there", "what are", "how much"]):
                if status == ResolutionStatus.RESOLVED:
                    attr = target_attr or TargetAttribute.ALL_DETAILS
                    return ContextResolutionResult(
                        status=ResolutionStatus.RESOLVED,
                        resolved_car=car,
                        target_attribute=attr,
                        raw_query=raw_msg
                    )
                elif status == ResolutionStatus.CLARIFICATION_REQUIRED:
                    return ContextResolutionResult(
                        status=ResolutionStatus.CLARIFICATION_REQUIRED,
                        clarification_message=clarify_msg,
                        raw_query=raw_msg
                    )

        # F. Fallback: Not a contextual follow-up
        return ContextResolutionResult(status=ResolutionStatus.NOT_CONTEXTUAL, raw_query=raw_msg)

    @classmethod
    def _match_result_set_comparison(
        cls, msg_lower: str, session: SessionState, raw_msg: str
    ) -> Optional[ContextResolutionResult]:
        """
        Evaluates whether the query is a whole-result-set model-year comparison
        (e.g., 'Which is the latest year model?' or 'Which is the oldest?').
        Strictly whole-result-set only.
        """
        is_latest = bool(re.search(r"\b(latest|newest|most\s+recent)\b", msg_lower))
        is_oldest = bool(re.search(r"\b(oldest|earliest)\b", msg_lower))

        if not is_latest and not is_oldest:
            return None

        # Exclude specific-car ordinal references (e.g. 'Which is the latest on the second car?')
        ordinal_words_pattern = r"\b(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|last|top)\b"
        if re.search(ordinal_words_pattern, msg_lower):
            return None

        # Check comparison phrasing patterns
        comparison_patterns = [
            # "which [car/one/vehicle/model] is [the] latest/oldest [model/car/year]?"
            r"\bwhich\s+(one|car|vehicle|model)?\s*(is|are)?\s*(the\s+)?(latest|newest|most\s+recent|oldest|earliest)\b",
            # "which [one/car/vehicle/model] has [the] newest/latest/oldest/earliest [model] [year]?"
            r"\bwhich\s+(one|car|vehicle|model)?\s*(has|have)\s*(the\s+)?(newest|latest|most\s+recent|oldest|earliest)\b",
            # "what is / what's the latest/newest/oldest/earliest [one/car/vehicle/model] [year]?"
            r"\b(what\s+is|what's)\s+(the\s+)?(latest|newest|most\s+recent|oldest|earliest)\b",
            # "which of these / which of them / which of the cars is [the] latest/oldest?"
            r"\bwhich\s+of\s+(these|them|the\s+cars|the\s+vehicles|the\s+results|the\s+listings)\s+(is|are|has)?\s*(the\s+)?(latest|newest|most\s+recent|oldest|earliest)\b",
            # "which car/vehicle is latest/oldest?"
            r"\bwhich\s+(car|vehicle|model|one)\s+(is|are)\s+(the\s+)?(latest|newest|most\s+recent|oldest|earliest)\b",
            # "which is latest/oldest?"
            r"^(which|what|what's)\s+(is\s+|has\s+)?(the\s+)?(latest|newest|most\s+recent|oldest|earliest)\b",
        ]

        if not any(re.search(p, msg_lower) for p in comparison_patterns):
            return None

        comp_type = ResultSetComparisonType.LATEST_YEAR if is_latest else ResultSetComparisonType.OLDEST_YEAR

        # Handle empty current_result_set
        if not session.current_result_set:
            label = "latest" if comp_type == ResultSetComparisonType.LATEST_YEAR else "oldest"
            return ContextResolutionResult(
                status=ResolutionStatus.CLARIFICATION_REQUIRED,
                comparison_type=comp_type,
                clarification_message=(
                    f"I don't have a current set of vehicle results to compare. "
                    f"Search for some cars first, then I can tell you which has the {label} model year."
                ),
                raw_query=raw_msg
            )

        # Non-empty result set: determine max/min year over session.current_result_set
        valid_years = [c.year for c in session.current_result_set if c.year is not None]
        if not valid_years:
            return ContextResolutionResult(
                status=ResolutionStatus.CLARIFICATION_REQUIRED,
                comparison_type=comp_type,
                clarification_message="None of the vehicles in your current results have a valid model year recorded.",
                raw_query=raw_msg
            )

        target_year = max(valid_years) if comp_type == ResultSetComparisonType.LATEST_YEAR else min(valid_years)
        matching_cars = [c for c in session.current_result_set if c.year == target_year]

        return ContextResolutionResult(
            status=ResolutionStatus.RESULT_SET_COMPARISON,
            comparison_type=comp_type,
            comparison_year=target_year,
            resolved_cars=matching_cars,
            resolved_car=matching_cars[0] if len(matching_cars) == 1 else None,
            raw_query=raw_msg
        )

    @classmethod
    def _is_unmistakable_fresh_search(cls, msg_lower: str) -> bool:
        """Conservative check to avoid intercepting fresh search queries."""
        # e.g., "show me gcc cars", "find cars under warranty", "show me low mileage bentleys"
        for pattern in cls.FRESH_SEARCH_INDICATORS:
            if re.search(pattern, msg_lower):
                return True
        return False

    @classmethod
    def _extract_target_attribute(cls, msg_lower: str) -> Optional[TargetAttribute]:
        """Extracts the vehicle attribute being inquired about."""
        for attr, pattern in cls.ATTRIBUTE_PATTERNS:
            if re.search(pattern, msg_lower):
                return attr
        return None

    @classmethod
    def _match_ordinal_reference(
        cls, msg_lower: str, result_set: List[CarListing]
    ) -> Optional[Tuple[ResolutionStatus, Optional[CarListing], Optional[str]]]:
        """
        Matches expressions like:
        - "the first one", "first car", "that second car", "last one"
        - "that first Honda", "second Bentley", "the last Land Rover"
        """
        ordinal_regex = r"\b(that\s+|the\s+)?(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|last|top)\s+([a-zA-Z0-9_-]+(\s+[a-zA-Z0-9_-]+)?)"
        match = re.search(ordinal_regex, msg_lower)
        if not match:
            # Check standalone ordinal like "how much is the first?"
            standalone_regex = r"\b(that\s+|the\s+)?(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|last|top)\b"
            sm = re.search(standalone_regex, msg_lower)
            if not sm:
                return None
            ord_word = sm.group(2)
            qualifier = "car"
        else:
            ord_word = match.group(2)
            qualifier = match.group(3).strip()

        ordinal_idx = cls.ORDINAL_MAP.get(ord_word)
        if ordinal_idx is None:
            return None

        # Filter result set by qualifier if qualifier is not generic
        generic_qualifiers = {"one", "car", "vehicle", "option", "result", "listing", "choice"}
        if qualifier in generic_qualifiers or qualifier.startswith("one") or qualifier.startswith("car"):
            filtered_subset = result_set
            qualifier_name = "vehicle"
        else:
            # Qualifier is a specific make or model (e.g. "honda", "bentley", "land rover")
            filtered_subset = [
                car for car in result_set
                if qualifier in car.make.lower() or (car.model and qualifier in car.model.lower())
            ]
            qualifier_name = qualifier.title()

            if not filtered_subset:
                return (
                    ResolutionStatus.CLARIFICATION_REQUIRED,
                    None,
                    f"There are no {qualifier_name} vehicles in your current search results. Would you like to search the full inventory for {qualifier_name}?"
                )

        # Apply ordinal index
        if not filtered_subset:
            return (
                ResolutionStatus.CLARIFICATION_REQUIRED,
                None,
                "There are no matching vehicles in your current search results."
            )

        if ordinal_idx == -1:  # "last"
            return (ResolutionStatus.RESOLVED, filtered_subset[-1], None)

        if ordinal_idx < len(filtered_subset):
            return (ResolutionStatus.RESOLVED, filtered_subset[ordinal_idx], None)
        else:
            # Out of range ordinal
            count = len(filtered_subset)
            suffix = f"{qualifier_name} " if qualifier_name != "vehicle" else ""
            return (
                ResolutionStatus.CLARIFICATION_REQUIRED,
                None,
                f"There are only {count} {suffix}vehicles in your current results. Please specify a number between 1 and {count}."
            )

    @classmethod
    def _match_make_reference(
        cls, msg_lower: str, result_set: List[CarListing]
    ) -> Optional[Tuple[ResolutionStatus, Optional[CarListing], Optional[str]]]:
        """
        Matches make/model references without ordinals like:
        - "What's the mileage on the Honda?"
        - "How much is the Bentley?"
        """
        # Look for phrases like "the {make}" or "that {make}"
        match = re.search(r"\b(on|about|for|is|is\s+there\s+a|how\s+much\s+is)\s+(the|that|this)\s+([a-zA-Z0-9_-]+)\b", msg_lower)
        if not match:
            return None

        candidate_make = match.group(3).strip()
        generic_words = {"car", "one", "vehicle", "option", "price", "warranty", "mileage", "spec", "specs"}
        if candidate_make in generic_words:
            return None

        matching_cars = [
            car for car in result_set
            if candidate_make in car.make.lower() or (car.model and candidate_make in car.model.lower())
        ]

        if not matching_cars:
            return None

        if len(matching_cars) == 1:
            return (ResolutionStatus.RESOLVED, matching_cars[0], None)
        else:
            # Multiple cars match this make in current results -> Ambiguity
            car_listings = ", ".join([f"Listing #{c.listing_id} ({c.year} {c.make} {c.model})" for c in matching_cars])
            return (
                ResolutionStatus.CLARIFICATION_REQUIRED,
                None,
                f"You have {len(matching_cars)} {candidate_make.title()} vehicles in your search results ({car_listings}). Which one would you like to know about?"
            )

    @classmethod
    def _match_pronoun_reference(
        cls, msg_lower: str, session: SessionState
    ) -> Optional[Tuple[ResolutionStatus, Optional[CarListing], Optional[str]]]:
        """
        Matches pronoun/deictic expressions like:
        - "Does it have a warranty?", "Is it GCC?", "What's its price?"
        - "How much is that car?", "Tell me more about this car"
        """
        pronoun_pattern = r"\b(it|its|that\s+car|this\s+car|the\s+car|that\s+vehicle|this\s+vehicle|this\s+one|that\s+one)\b"
        has_qualified_reference = re.search(pronoun_pattern, msg_lower) is not None
        if not has_qualified_reference and not cls._is_bare_deictic_follow_up(msg_lower):
            return None

        # 1. If active_listing_id exists, look it up in current_result_set
        if session.active_listing_id is not None:
            for car in session.current_result_set:
                if car.listing_id == session.active_listing_id:
                    return (ResolutionStatus.RESOLVED, car, None)

        # 2. Single-visible-car fallback: If exactly one car in current_result_set
        if len(session.current_result_set) == 1:
            return (ResolutionStatus.RESOLVED, session.current_result_set[0], None)

        # 3. Multiple cars in result set without active listing -> Ambiguity
        if len(session.current_result_set) > 1:
            return (
                ResolutionStatus.CLARIFICATION_REQUIRED,
                None,
                "Which vehicle from your search results are you referring to? You can say, for example, 'the first one' or 'the second car'."
            )

        # 4. Zero cars in result set and no active vehicle -> Clarification
        return (
            ResolutionStatus.CLARIFICATION_REQUIRED,
            None,
            "Which vehicle are you referring to? Please search for a vehicle first or specify the vehicle you would like to know about."
        )

    @staticmethod
    def _is_bare_deictic_follow_up(msg_lower: str) -> bool:
        """Recognizes bare this/that only in unambiguous follow-up question shapes."""
        patterns = [
            r"\b(?:is|on|about|for)\s+(?:that|this)\s*[?.!]*$",
            r"\bdoes\s+(?:that|this)\s+have\b",
            r"^(?:is|does)\s+(?:that|this)\b",
        ]
        return any(re.search(pattern, msg_lower) for pattern in patterns)
