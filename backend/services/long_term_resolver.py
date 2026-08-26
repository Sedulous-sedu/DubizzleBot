"""Deterministic resolver for Phase 4B long-term persistent memory actions."""

import re
from typing import Optional, Tuple, Set

from backend.models.memory import SessionState
from backend.models.persistent_memory import (
    LongTermMemoryAction,
    LongTermMemoryResolution,
    PreferencePatch,
)
from backend.models.intent import RegionalSpecEnum, ParsedInventoryQuery
from backend.services.context_resolver import ContextResolver, ResolutionStatus

class LongTermMemoryResolver:
    """Evaluates incoming messages for persistent memory operations."""

    @staticmethod
    def evaluate(message: str, session: SessionState) -> LongTermMemoryResolution:
        """Evaluates user query for long-term memory intents and resolves targets."""
        msg_clean = message.strip()
        msg_lower = msg_clean.lower()

        # 1. Clear / Forget memory commands
        if re.search(r"\b(?:forget|delete|clear)\s+(?:everything\s+about\s+me|all\s+my\s+data|all\s+(?:my\s+)?memory)\b", msg_lower):
            return LongTermMemoryResolution(action=LongTermMemoryAction.CLEAR_ALL_MEMORY)

        if re.search(r"\b(?:forget|clear|delete|reset)\s+(?:all\s+)?(?:my\s+)?(?:saved\s+)?preferences\b", msg_lower):
            return LongTermMemoryResolution(action=LongTermMemoryAction.CLEAR_PREFERENCES)

        if re.search(r"\b(?:clear|delete|remove)\s+(?:all\s+)?(?:my\s+)?(?:saved|liked)\s+cars\b|\bclear\s+(?:all\s+)?(?:my\s+)?favorites\b", msg_lower):
            return LongTermMemoryResolution(action=LongTermMemoryAction.CLEAR_LIKED_CARS)

        # 2. Recall memory / Memory transparency
        if re.search(r"\bwhat\s+(?:do\s+you\s+remember|do\s+you\s+know)\s+about\s+me\b|\b(?:what\s+are|show)\s+(?:all\s+)?(?:my\s+)?(?:saved\s+)?preferences\b|\bshow\s+my\s+(?:saved\s+)?profile\b|\bwhat\s+preferences\s+do\s+you\s+have\b", msg_lower):
            return LongTermMemoryResolution(action=LongTermMemoryAction.RECALL_MEMORY)

        # 3. Recall liked / saved cars
        if re.search(r"\bwhat\s+cars\s+did\s+i\s+like\b|\bshow\s+(?:all\s+)?(?:my\s+)?(?:saved|liked)\s+cars\b|\bwhat\s+did\s+i\s+save\b|\bdo\s+i\s+have\s+any\s+favorites\b|\bshow\s+(?:all\s+)?(?:my\s+)?favorites\b|\blist\s+(?:all\s+)?(?:my\s+)?(?:saved|liked)\s+cars\b|\bmy\s+liked\s+cars\b|\bwhat\s+cars\s+have\s+i\s+saved\b", msg_lower):
            return LongTermMemoryResolution(action=LongTermMemoryAction.RECALL_LIKED_CARS)

        # 4. Search using saved preferences
        pref_search_match = re.search(
            r"\b(?:show|find|search)\s+(?:me\s+)?([a-z0-9\s]+?)\s*(?:matching|based\s+on)\s+(?:my\s+)?(?:saved\s+)?preferences\b|\bsearch\s+using\s+(?:my\s+)?(?:saved\s+)?preferences\b|\bfind\s+cars\s+for\s+me\s+based\s+on\s+my\s+preferences\b",
            msg_lower
        )
        if pref_search_match:
            override_query: Optional[ParsedInventoryQuery] = None
            if pref_search_match.group(1):
                subject = pref_search_match.group(1).strip()
                if subject not in ("cars", "vehicles", "any cars", "some cars", "options", ""):
                    clean_subject = re.sub(r"s$", "", subject) if subject.endswith("s") and not subject.endswith("ss") else subject
                    override_query = ParsedInventoryQuery(make=clean_subject.title())
            return LongTermMemoryResolution(
                action=LongTermMemoryAction.SEARCH_SAVED_PREFERENCES,
                search_query_override=override_query
            )

        # 5. Remove liked car
        if re.search(r"\b(?:remove|delete|unlike)\s+(?:this|that|the)\s+car\s+from\s+(?:my\s+)?favorites\b|\bunlike\s+(?:this|that)\s+car\b|\bdelete\s+from\s+saved\b|\bremove\s+(?:the\s+)?(?:first|second|third|1st|2nd|3rd)\s+(?:one|car)?\s+from\s+(?:my\s+)?favorites\b", msg_lower):
            ctx_res = ContextResolver.resolve(message, session)
            if ctx_res.status == ResolutionStatus.RESOLVED and ctx_res.resolved_car:
                return LongTermMemoryResolution(
                    action=LongTermMemoryAction.REMOVE_LIKED_CAR,
                    target_car=ctx_res.resolved_car,
                    target_listing_id=ctx_res.resolved_car.listing_id
                )
            elif ctx_res.status == ResolutionStatus.CLARIFICATION_REQUIRED:
                return LongTermMemoryResolution(
                    action=LongTermMemoryAction.REMOVE_LIKED_CAR,
                    clarification_message=ctx_res.clarification_message
                )
            else:
                return LongTermMemoryResolution(
                    action=LongTermMemoryAction.REMOVE_LIKED_CAR,
                    clarification_message="Which vehicle would you like to remove from your favorites?"
                )

        # 6. Save liked car
        # Must detect expressions like "I like the second one", "Save this car", "Add to favorites", "Save the first Bentley", etc.
        # But exclude broad search queries like "I like Bentleys, show me some"
        is_search_query = bool(re.search(r"\b(?:show|find|search|give)\s+me\b", msg_lower))
        is_like_phrase = bool(re.search(
            r"\b(?:save\s+(?:this|that|the|listing\s*#?\d+)|add\s+(?:this|that|the)?\s*(?:car|vehicle|one)?\s*(?:to\s+(?:my\s+)?favorites|to\s+saved)|remember\s+(?:this|that|the)\s+(?:car|vehicle|one)|save\s+to\s+favorites|i\s+(?:really\s+)?(?:like|love)\s+(?:the|this|that))\b",
            msg_lower
        ))

        if is_like_phrase and not is_search_query:
            # Resolve target vehicle through ContextResolver
            ctx_res = ContextResolver.resolve(message, session)
            if ctx_res.status == ResolutionStatus.RESOLVED and ctx_res.resolved_car:
                return LongTermMemoryResolution(
                    action=LongTermMemoryAction.SAVE_LIKED_CAR,
                    target_car=ctx_res.resolved_car,
                    target_listing_id=ctx_res.resolved_car.listing_id
                )
            elif ctx_res.status == ResolutionStatus.CLARIFICATION_REQUIRED:
                return LongTermMemoryResolution(
                    action=LongTermMemoryAction.SAVE_LIKED_CAR,
                    clarification_message=ctx_res.clarification_message
                )
            else:
                # Target could not be resolved from session context
                return LongTermMemoryResolution(
                    action=LongTermMemoryAction.SAVE_LIKED_CAR,
                    clarification_message="Which vehicle would you like to save? Please search for a vehicle first or specify which car from your search results to save."
                )

        # 7. Explicit preference setting / updating
        # Detect explicit preference language
        is_pref_phrase = bool(re.search(
            r"\b(?:i\s+prefer\b|my\s+budget\s+is\b|remember\s+that\s+i\b|save\s+my\s+preference\b|i\s+want\s+you\s+to\s+remember\b|i\s+don't\s+care\s+about\s+warranty\b|i\s+do\s+not\s+care\s+about\s+warranty\b)",
            msg_lower
        ))

        if is_pref_phrase:
            patch = LongTermMemoryResolver._extract_preference_patch(msg_clean)
            if patch and not patch.is_empty():
                return LongTermMemoryResolution(
                    action=LongTermMemoryAction.SAVE_PREFERENCE,
                    preference_patch=patch
                )

        # Not a long-term memory action
        return LongTermMemoryResolution(action=LongTermMemoryAction.NOT_MEMORY_ACTION)

    @staticmethod
    def _extract_preference_patch(message: str) -> Optional[PreferencePatch]:
        """Extracts typed PreferencePatch from explicit preference language."""
        msg_lower = message.lower()
        clear_fields: Set[str] = set()

        preferred_make: Optional[str] = None
        preferred_model: Optional[str] = None
        min_year: Optional[int] = None
        max_year: Optional[int] = None
        min_price_aed: Optional[float] = None
        max_price_aed: Optional[float] = None
        min_mileage_km: Optional[int] = None
        max_mileage_km: Optional[int] = None
        regional_specs: Optional[str] = None
        warranty_preference: Optional[bool] = None
        keywords: Optional[str] = None

        # 1. Warranty clearing vs setting
        if re.search(r"\b(?:don't|do not)\s+(?:care|mind)\s+about\s+warranty\b|\bno\s+warranty\s+preference\b", msg_lower):
            clear_fields.add("warranty_preference")
        elif re.search(r"\b(?:prefer|want|need)\s+(?:cars\s+under\s+)?warranty\b|\bwarranty\s+is\s+required\b", msg_lower):
            warranty_preference = True

        # 2. Regional specs
        if "gcc" in msg_lower:
            regional_specs = RegionalSpecEnum.GCC.value
        elif "american" in msg_lower:
            regional_specs = RegionalSpecEnum.AMERICAN.value
        elif "japanese" in msg_lower:
            regional_specs = RegionalSpecEnum.JAPANESE.value
        elif "european" in msg_lower:
            regional_specs = RegionalSpecEnum.EUROPEAN.value

        # 3. Budget / Max Price
        budget_match = re.search(
            r"\b(?:budget\s+(?:is\s+)?(?:now\s+)?(?:under\s+)?|max(?:imum)?\s+price\s+(?:is\s+)?(?:now\s+)?)(?:aed\s+)?(\d[\d,]*(?:\.\d+)?)\b",
            msg_lower
        )
        if budget_match:
            try:
                max_price_aed = float(budget_match.group(1).replace(",", ""))
            except ValueError:
                pass

        # 4. Preferred Make
        known_makes = [
            "land rover", "aston martin", "rolls-royce", "mercedes-benz", "alfa romeo",
            "bentley", "ferrari", "porsche", "ford", "honda", "toyota", "nissan",
            "bmw", "audi", "chevrolet", "hyundai", "kia", "volkswagen", "lamborghini",
            "maserati", "mclaren", "jaguar", "jeep", "dodge", "cadillac", "lexus"
        ]
        for make in known_makes:
            if re.search(rf"\b(?:prefer|like|want)\s+{re.escape(make)}\b|\b{re.escape(make)}\s+cars\b", msg_lower):
                preferred_make = make.title()
                break

        # 5. Mileage
        mileage_match = re.search(r"\b(?:mileage\s+(?:under|max|less than)|under)\s+(\d[\d,]*)\s*(?:km|kms|k)\b", msg_lower)
        if mileage_match:
            try:
                raw_km = mileage_match.group(1).replace(",", "")
                # If followed by 'k' and value is small (e.g. 50k -> 50000)
                if "k" in mileage_match.group(0) and float(raw_km) < 1000:
                    max_mileage_km = int(float(raw_km) * 1000)
                else:
                    max_mileage_km = int(raw_km)
            except ValueError:
                pass

        # 6. Year
        year_match = re.search(r"\b(?:year\s+(?:after|from|minimum)|from\s+year)\s+(19\d\d|20\d\d)\b", msg_lower)
        if year_match:
            try:
                min_year = int(year_match.group(1))
            except ValueError:
                pass

        return PreferencePatch(
            preferred_make=preferred_make,
            preferred_model=preferred_model,
            min_year=min_year,
            max_year=max_year,
            min_price_aed=min_price_aed,
            max_price_aed=max_price_aed,
            min_mileage_km=min_mileage_km,
            max_mileage_km=max_mileage_km,
            regional_specs=regional_specs,
            warranty_preference=warranty_preference,
            keywords=keywords,
            clear_fields=clear_fields,
        )
