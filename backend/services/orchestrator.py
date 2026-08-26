"""Chat orchestrator connecting MemoryService, PersistentMemoryService, ContextResolver, LongTermMemoryResolver, QueryInterpreter, InventoryService, and GroundedResponseBuilder."""

import uuid
import logging
from typing import Optional, List
from backend.models.car import CarListing
from backend.models.chat import ChatRequest, ChatResponse
from backend.models.intent import (
    UserIntentEnum,
    SearchReadinessState,
    ParsedUserIntent,
)
from backend.models.memory import (
    ResolutionStatus,
    ContextResolutionResult,
)
from backend.models.persistent_memory import (
    LongTermMemoryAction,
    LongTermMemoryResolution,
)
from backend.services.query_interpreter import QueryInterpreter
from backend.services.inventory import InventoryService
from backend.services.response_builder import GroundedResponseBuilder
from backend.services.memory import MemoryService
from backend.services.persistent_memory import PersistentMemoryService
from backend.services.context_resolver import ContextResolver
from backend.services.long_term_resolver import LongTermMemoryResolver

logger = logging.getLogger(__name__)

class ChatOrchestrator:
    """Core domain orchestrator coordinating session memory, persistent memory, contextual resolution, NLP interpretation, and inventory retrieval."""

    def __init__(
        self,
        query_interpreter: Optional[QueryInterpreter] = None,
        inventory_service: Optional[InventoryService] = None,
        memory_service: Optional[MemoryService] = None,
        persistent_memory: Optional[PersistentMemoryService] = None,
    ):
        self.query_interpreter = query_interpreter or QueryInterpreter()
        self.inventory_service = inventory_service or InventoryService()
        self.memory_service = memory_service or MemoryService()
        self.persistent_memory = persistent_memory or PersistentMemoryService()

    def process_chat(self, request: ChatRequest) -> ChatResponse:
        """
        Executes end-to-end processing for incoming user chat request.
        Preserves or generates session_id, evaluates long-term memory actions,
        evaluates short-term contextual references, and routes deterministically.
        """
        session_id = request.session_id or str(uuid.uuid4())
        session = self.memory_service.get_or_create_session(request.user_id, session_id)
        self.persistent_memory.record_activity(request.user_id)

        try:
            # 1. Evaluate deterministic LongTermMemoryResolver first
            lt_res: LongTermMemoryResolution = LongTermMemoryResolver.evaluate(request.message, session)
            if lt_res.action != LongTermMemoryAction.NOT_MEMORY_ACTION:
                return self._handle_long_term_memory(request, session_id, lt_res)

            # 2. Evaluate deterministic ContextResolver on active session state (Phase 4A)
            context_result: ContextResolutionResult = ContextResolver.resolve(request.message, session)

            if context_result.status == ResolutionStatus.RESOLVED:
                target_car = context_result.resolved_car
                response_text = GroundedResponseBuilder.format_vehicle_attribute_response(
                    target_car, context_result.target_attribute
                )
                self.memory_service.record_turn(
                    user_id=request.user_id,
                    session_id=session_id,
                    user_message=request.message,
                    assistant_response=response_text,
                    intent=UserIntentEnum.INVENTORY_SEARCH,
                    matched_cars=[target_car],
                    referenced_listing_id=target_car.listing_id,
                    replace_result_set=False,
                    active_listing_id=target_car.listing_id
                )
                return ChatResponse(
                    user_id=request.user_id,
                    session_id=session_id,
                    response=response_text,
                    matched_cars=[target_car],
                    intent=UserIntentEnum.INVENTORY_SEARCH,
                    total_matches=1,
                    requires_clarification=False
                )

            elif context_result.status == ResolutionStatus.CLARIFICATION_REQUIRED:
                response_text = context_result.clarification_message or GroundedResponseBuilder.format_clarification_response(None)
                self.memory_service.record_turn(
                    user_id=request.user_id,
                    session_id=session_id,
                    user_message=request.message,
                    assistant_response=response_text,
                    intent=UserIntentEnum.INVENTORY_SEARCH,
                    matched_cars=None,
                    referenced_listing_id=None,
                    replace_result_set=False
                )
                return ChatResponse(
                    user_id=request.user_id,
                    session_id=session_id,
                    response=response_text,
                    matched_cars=None,
                    intent=UserIntentEnum.INVENTORY_SEARCH,
                    total_matches=0,
                    requires_clarification=True
                )

            # 3. Fresh query: Route via QueryInterpreter and Phase 3B logic
            parsed_intent: ParsedUserIntent = self.query_interpreter.interpret(request.message)


            if parsed_intent.intent == UserIntentEnum.INVENTORY_SEARCH:
                return self._handle_inventory_search(request.user_id, session_id, request.message, parsed_intent)

            elif parsed_intent.intent == UserIntentEnum.VIEWING_OR_LEAD_REQUEST:
                return self._handle_viewing_request(request.user_id, session_id, request.message, parsed_intent)

            elif parsed_intent.intent == UserIntentEnum.GENERAL_CHAT:
                response_text = GroundedResponseBuilder.format_general_chat_response(request.message)
                self.memory_service.record_turn(
                    user_id=request.user_id,
                    session_id=session_id,
                    user_message=request.message,
                    assistant_response=response_text,
                    intent=parsed_intent.intent,
                    matched_cars=None,
                    referenced_listing_id=None,
                    replace_result_set=False
                )
                return ChatResponse(
                    user_id=request.user_id,
                    session_id=session_id,
                    response=response_text,
                    matched_cars=None,
                    intent=parsed_intent.intent,
                    total_matches=0,
                    requires_clarification=False
                )

            else:  # UNKNOWN / Non-automotive
                response_text = GroundedResponseBuilder.format_unknown_response()
                self.memory_service.record_turn(
                    user_id=request.user_id,
                    session_id=session_id,
                    user_message=request.message,
                    assistant_response=response_text,
                    intent=UserIntentEnum.UNKNOWN,
                    matched_cars=None,
                    referenced_listing_id=None,
                    replace_result_set=False
                )
                return ChatResponse(
                    user_id=request.user_id,
                    session_id=session_id,
                    response=response_text,
                    matched_cars=None,
                    intent=UserIntentEnum.UNKNOWN,
                    total_matches=0,
                    requires_clarification=False
                )

        except Exception as e:
            logger.error(f"Error during chat orchestration: {e}", exc_info=True)
            fallback_text = (
                "I apologize, but I encountered an issue processing your request. "
                "Please try again or rephrase your search criteria."
            )
            self.memory_service.record_turn(
                user_id=request.user_id,
                session_id=session_id,
                user_message=request.message,
                assistant_response=fallback_text,
                intent=UserIntentEnum.UNKNOWN,
                matched_cars=None,
                referenced_listing_id=None,
                replace_result_set=False
            )
            return ChatResponse(
                user_id=request.user_id,
                session_id=session_id,
                response=fallback_text,
                matched_cars=None,
                intent=UserIntentEnum.UNKNOWN,
                total_matches=0,
                requires_clarification=False
            )

    def _handle_inventory_search(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        parsed_intent: ParsedUserIntent
    ) -> ChatResponse:
        """Handles inventory search according to readiness state and updates session state."""
        if parsed_intent.readiness_state == SearchReadinessState.READY:
            if parsed_intent.query_filters:
                car_filter = parsed_intent.query_filters.to_car_filter()
                self.persistent_memory.update_last_search(user_id, parsed_intent.query_filters)
                raw_results = self.inventory_service.search(car_filter)
                matched_cars: List[CarListing] = [
                    CarListing.model_validate(r) if isinstance(r, dict) else r
                    for r in raw_results
                ]
                total_matches = len(matched_cars)
                response_text = GroundedResponseBuilder.format_inventory_search_response(
                    matched_cars, total_matches, parsed_intent.query_filters
                )

                # Update current result set in session and reset active vehicle
                self.memory_service.record_turn(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=user_message,
                    assistant_response=response_text,
                    intent=parsed_intent.intent,
                    matched_cars=matched_cars,
                    referenced_listing_id=None,
                    replace_result_set=True,
                    active_listing_id=None
                )

                return ChatResponse(
                    user_id=user_id,
                    session_id=session_id,
                    response=response_text,
                    matched_cars=matched_cars,
                    intent=parsed_intent.intent,
                    total_matches=total_matches,
                    requires_clarification=False
                )
            else:
                response_text = GroundedResponseBuilder.format_clarification_response(None)
                self.memory_service.record_turn(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=user_message,
                    assistant_response=response_text,
                    intent=parsed_intent.intent,
                    matched_cars=None,
                    referenced_listing_id=None,
                    replace_result_set=False
                )
                return ChatResponse(
                    user_id=user_id,
                    session_id=session_id,
                    response=response_text,
                    matched_cars=None,
                    intent=parsed_intent.intent,
                    total_matches=0,
                    requires_clarification=True
                )

        elif parsed_intent.readiness_state == SearchReadinessState.CLARIFICATION_REQUIRED:
            response_text = GroundedResponseBuilder.format_clarification_response(
                parsed_intent.clarification_question
            )
            self.memory_service.record_turn(
                user_id=user_id,
                session_id=session_id,
                user_message=user_message,
                assistant_response=response_text,
                intent=parsed_intent.intent,
                matched_cars=None,
                referenced_listing_id=None,
                replace_result_set=False
            )
            return ChatResponse(
                user_id=user_id,
                session_id=session_id,
                response=response_text,
                matched_cars=None,
                intent=parsed_intent.intent,
                total_matches=0,
                requires_clarification=True
            )

        elif parsed_intent.readiness_state == SearchReadinessState.UNSUPPORTED_CONSTRAINTS_PRESENT:
            response_text = GroundedResponseBuilder.format_unsupported_constraints_response(
                parsed_intent.unsupported_constraints,
                parsed_intent.query_filters
            )
            self.memory_service.record_turn(
                user_id=user_id,
                session_id=session_id,
                user_message=user_message,
                assistant_response=response_text,
                intent=parsed_intent.intent,
                matched_cars=None,
                referenced_listing_id=None,
                replace_result_set=False
            )
            return ChatResponse(
                user_id=user_id,
                session_id=session_id,
                response=response_text,
                matched_cars=None,
                intent=parsed_intent.intent,
                total_matches=0,
                requires_clarification=False
            )

        else:
            response_text = GroundedResponseBuilder.format_unknown_response()
            self.memory_service.record_turn(
                user_id=user_id,
                session_id=session_id,
                user_message=user_message,
                assistant_response=response_text,
                intent=parsed_intent.intent,
                matched_cars=None,
                referenced_listing_id=None,
                replace_result_set=False
            )
            return ChatResponse(
                user_id=user_id,
                session_id=session_id,
                response=response_text,
                matched_cars=None,
                intent=parsed_intent.intent,
                total_matches=0,
                requires_clarification=False
            )

    def _handle_viewing_request(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        parsed_intent: ParsedUserIntent
    ) -> ChatResponse:
        """Handles viewing / lead requests according to readiness state and filter presence."""
        if parsed_intent.readiness_state == SearchReadinessState.CLARIFICATION_REQUIRED:
            response_text = GroundedResponseBuilder.format_clarification_response(
                parsed_intent.clarification_question
            )
            self.memory_service.record_turn(
                user_id=user_id,
                session_id=session_id,
                user_message=user_message,
                assistant_response=response_text,
                intent=parsed_intent.intent,
                matched_cars=None,
                referenced_listing_id=None,
                replace_result_set=False
            )
            return ChatResponse(
                user_id=user_id,
                session_id=session_id,
                response=response_text,
                matched_cars=None,
                intent=parsed_intent.intent,
                total_matches=0,
                requires_clarification=True
            )

        elif parsed_intent.readiness_state == SearchReadinessState.UNSUPPORTED_CONSTRAINTS_PRESENT:
            response_text = GroundedResponseBuilder.format_unsupported_constraints_response(
                parsed_intent.unsupported_constraints,
                parsed_intent.query_filters
            )
            self.memory_service.record_turn(
                user_id=user_id,
                session_id=session_id,
                user_message=user_message,
                assistant_response=response_text,
                intent=parsed_intent.intent,
                matched_cars=None,
                referenced_listing_id=None,
                replace_result_set=False
            )
            return ChatResponse(
                user_id=user_id,
                session_id=session_id,
                response=response_text,
                matched_cars=None,
                intent=parsed_intent.intent,
                total_matches=0,
                requires_clarification=False
            )

        elif parsed_intent.readiness_state == SearchReadinessState.READY:
            if parsed_intent.query_filters and self._has_searchable_criteria(parsed_intent.query_filters):
                car_filter = parsed_intent.query_filters.to_car_filter()
                self.persistent_memory.update_last_search(user_id, parsed_intent.query_filters)
                raw_results = self.inventory_service.search(car_filter)
                matched_cars: List[CarListing] = [
                    CarListing.model_validate(r) if isinstance(r, dict) else r
                    for r in raw_results
                ]
                total_matches = len(matched_cars)
                response_text = GroundedResponseBuilder.format_viewing_response(
                    matched_cars, total_matches, parsed_intent.query_filters
                )

                # Store viewing candidates as current result set
                self.memory_service.record_turn(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=user_message,
                    assistant_response=response_text,
                    intent=parsed_intent.intent,
                    matched_cars=matched_cars,
                    referenced_listing_id=None,
                    replace_result_set=True,
                    active_listing_id=None
                )

                return ChatResponse(
                    user_id=user_id,
                    session_id=session_id,
                    response=response_text,
                    matched_cars=matched_cars,
                    intent=parsed_intent.intent,
                    total_matches=total_matches,
                    requires_clarification=False
                )
            else:
                response_text = GroundedResponseBuilder.format_viewing_response(None, 0, None)
                self.memory_service.record_turn(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=user_message,
                    assistant_response=response_text,
                    intent=parsed_intent.intent,
                    matched_cars=None,
                    referenced_listing_id=None,
                    replace_result_set=False
                )
                return ChatResponse(
                    user_id=user_id,
                    session_id=session_id,
                    response=response_text,
                    matched_cars=None,
                    intent=parsed_intent.intent,
                    total_matches=0,
                    requires_clarification=False
                )

        else:
            response_text = GroundedResponseBuilder.format_viewing_response(None, 0, None)
            self.memory_service.record_turn(
                user_id=user_id,
                session_id=session_id,
                user_message=user_message,
                assistant_response=response_text,
                intent=parsed_intent.intent,
                matched_cars=None,
                referenced_listing_id=None,
                replace_result_set=False
            )
            return ChatResponse(
                user_id=user_id,
                session_id=session_id,
                response=response_text,
                matched_cars=None,
                intent=parsed_intent.intent,
                total_matches=0,
                requires_clarification=False
            )

    def _handle_long_term_memory(
        self,
        request: ChatRequest,
        session_id: str,
        lt_res: LongTermMemoryResolution
    ) -> ChatResponse:
        """Handles Phase 4B long-term memory actions deterministically."""
        user_id = request.user_id

        # 1. SAVE_LIKED_CAR
        if lt_res.action == LongTermMemoryAction.SAVE_LIKED_CAR:
            if lt_res.target_car:
                target = lt_res.target_car
                self.persistent_memory.save_liked_car(user_id, target.listing_id)
                response_text = GroundedResponseBuilder.format_saved_car_confirmation(target)
                self.memory_service.record_turn(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=request.message,
                    assistant_response=response_text,
                    intent=UserIntentEnum.INVENTORY_SEARCH,
                    matched_cars=[target],
                    referenced_listing_id=target.listing_id,
                    replace_result_set=False,
                    active_listing_id=target.listing_id
                )
                return ChatResponse(
                    user_id=user_id,
                    session_id=session_id,
                    response=response_text,
                    matched_cars=[target],
                    intent=UserIntentEnum.INVENTORY_SEARCH,
                    total_matches=1,
                    requires_clarification=False
                )
            else:
                response_text = lt_res.clarification_message or "Which vehicle would you like to save to your favorites?"
                self.memory_service.record_turn(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=request.message,
                    assistant_response=response_text,
                    intent=UserIntentEnum.INVENTORY_SEARCH,
                    matched_cars=None,
                    referenced_listing_id=None,
                    replace_result_set=False
                )
                return ChatResponse(
                    user_id=user_id,
                    session_id=session_id,
                    response=response_text,
                    matched_cars=None,
                    intent=UserIntentEnum.INVENTORY_SEARCH,
                    total_matches=0,
                    requires_clarification=True
                )

        # 2. RECALL_LIKED_CARS
        elif lt_res.action == LongTermMemoryAction.RECALL_LIKED_CARS:
            liked_ids = self.persistent_memory.get_liked_listing_ids(user_id)
            saved_cars: List[CarListing] = []
            missing_ids: List[int] = []

            for lid in liked_ids:
                car = self.inventory_service.get_by_listing_id(lid)
                if car:
                    saved_cars.append(car)
                else:
                    missing_ids.append(lid)

            response_text = GroundedResponseBuilder.format_liked_cars_response(saved_cars, missing_ids)
            total_matches = len(saved_cars)

            if saved_cars:
                self.memory_service.record_turn(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=request.message,
                    assistant_response=response_text,
                    intent=UserIntentEnum.INVENTORY_SEARCH,
                    matched_cars=saved_cars,
                    referenced_listing_id=None,
                    replace_result_set=True,
                    active_listing_id=None
                )
            else:
                self.memory_service.record_turn(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=request.message,
                    assistant_response=response_text,
                    intent=UserIntentEnum.INVENTORY_SEARCH,
                    matched_cars=None,
                    referenced_listing_id=None,
                    replace_result_set=False
                )

            return ChatResponse(
                user_id=user_id,
                session_id=session_id,
                response=response_text,
                matched_cars=saved_cars if saved_cars else None,
                intent=UserIntentEnum.INVENTORY_SEARCH,
                total_matches=total_matches,
                requires_clarification=False
            )

        # 3. REMOVE_LIKED_CAR
        elif lt_res.action == LongTermMemoryAction.REMOVE_LIKED_CAR:
            if lt_res.target_listing_id:
                self.persistent_memory.remove_liked_car(user_id, lt_res.target_listing_id)
                response_text = GroundedResponseBuilder.format_removed_car_confirmation(
                    lt_res.target_listing_id, lt_res.target_car
                )
                self.memory_service.record_turn(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=request.message,
                    assistant_response=response_text,
                    intent=UserIntentEnum.INVENTORY_SEARCH,
                    matched_cars=None,
                    referenced_listing_id=None,
                    replace_result_set=False
                )
                return ChatResponse(
                    user_id=user_id,
                    session_id=session_id,
                    response=response_text,
                    matched_cars=None,
                    intent=UserIntentEnum.INVENTORY_SEARCH,
                    total_matches=0,
                    requires_clarification=False
                )
            else:
                response_text = lt_res.clarification_message or "Which vehicle would you like to remove from your favorites?"
                self.memory_service.record_turn(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=request.message,
                    assistant_response=response_text,
                    intent=UserIntentEnum.INVENTORY_SEARCH,
                    matched_cars=None,
                    referenced_listing_id=None,
                    replace_result_set=False
                )
                return ChatResponse(
                    user_id=user_id,
                    session_id=session_id,
                    response=response_text,
                    matched_cars=None,
                    intent=UserIntentEnum.INVENTORY_SEARCH,
                    total_matches=0,
                    requires_clarification=True
                )

        # 4. RECALL_MEMORY (Transparency)
        elif lt_res.action == LongTermMemoryAction.RECALL_MEMORY:
            prefs = self.persistent_memory.get_preferences(user_id)
            liked_ids = self.persistent_memory.get_liked_listing_ids(user_id)
            response_text = GroundedResponseBuilder.format_preferences_summary_response(
                prefs, liked_count=len(liked_ids)
            )
            self.memory_service.record_turn(
                user_id=user_id,
                session_id=session_id,
                user_message=request.message,
                assistant_response=response_text,
                intent=UserIntentEnum.GENERAL_CHAT,
                matched_cars=None,
                referenced_listing_id=None,
                replace_result_set=False
            )
            return ChatResponse(
                user_id=user_id,
                session_id=session_id,
                response=response_text,
                matched_cars=None,
                intent=UserIntentEnum.GENERAL_CHAT,
                total_matches=0,
                requires_clarification=False
            )

        # 5. SAVE_PREFERENCE
        elif lt_res.action == LongTermMemoryAction.SAVE_PREFERENCE:
            if lt_res.preference_patch:
                updated_prefs = self.persistent_memory.save_preferences(user_id, lt_res.preference_patch)
                response_text = GroundedResponseBuilder.format_preference_saved_confirmation(updated_prefs)
            else:
                response_text = "I couldn't identify specific preferences to save. You can say 'I prefer GCC cars' or 'My budget is AED 120,000'."

            self.memory_service.record_turn(
                user_id=user_id,
                session_id=session_id,
                user_message=request.message,
                assistant_response=response_text,
                intent=UserIntentEnum.GENERAL_CHAT,
                matched_cars=None,
                referenced_listing_id=None,
                replace_result_set=False
            )
            return ChatResponse(
                user_id=user_id,
                session_id=session_id,
                response=response_text,
                matched_cars=None,
                intent=UserIntentEnum.GENERAL_CHAT,
                total_matches=0,
                requires_clarification=False
            )

        # 6. SEARCH_SAVED_PREFERENCES
        elif lt_res.action == LongTermMemoryAction.SEARCH_SAVED_PREFERENCES:
            prefs = self.persistent_memory.get_preferences(user_id)
            if not prefs or not prefs.has_explicit_preferences():
                response_text = (
                    "You don't have any saved preferences yet. "
                    "You can set preferences by saying things like 'I prefer GCC cars' or 'My budget is AED 120,000'."
                )
                self.memory_service.record_turn(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=request.message,
                    assistant_response=response_text,
                    intent=UserIntentEnum.INVENTORY_SEARCH,
                    matched_cars=None,
                    referenced_listing_id=None,
                    replace_result_set=False
                )
                return ChatResponse(
                    user_id=user_id,
                    session_id=session_id,
                    response=response_text,
                    matched_cars=None,
                    intent=UserIntentEnum.INVENTORY_SEARCH,
                    total_matches=0,
                    requires_clarification=False
                )
            else:
                car_filter = prefs.to_car_filter()
                if car_filter is None:
                    car_filter = CarFilter()

                # Apply explicit current-turn make override without contradictory duplicate criteria
                if lt_res.search_query_override and lt_res.search_query_override.make:
                    car_filter.make = lt_res.search_query_override.make

                raw_results = self.inventory_service.search(car_filter)
                matched_cars = [
                    CarListing.model_validate(r) if isinstance(r, dict) else r
                    for r in raw_results
                ]
                total_matches = len(matched_cars)
                response_text = GroundedResponseBuilder.format_inventory_search_response(
                    matched_cars, total_matches, None
                )
                self.memory_service.record_turn(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=request.message,
                    assistant_response=response_text,
                    intent=UserIntentEnum.INVENTORY_SEARCH,
                    matched_cars=matched_cars,
                    referenced_listing_id=None,
                    replace_result_set=True,
                    active_listing_id=None
                )
                return ChatResponse(
                    user_id=user_id,
                    session_id=session_id,
                    response=response_text,
                    matched_cars=matched_cars,
                    intent=UserIntentEnum.INVENTORY_SEARCH,
                    total_matches=total_matches,
                    requires_clarification=False
                )

        # 7. CLEAR_PREFERENCES
        elif lt_res.action == LongTermMemoryAction.CLEAR_PREFERENCES:
            self.persistent_memory.clear_preferences(user_id)
            response_text = GroundedResponseBuilder.format_clear_confirmation("preferences")
            self.memory_service.record_turn(
                user_id=user_id,
                session_id=session_id,
                user_message=request.message,
                assistant_response=response_text,
                intent=UserIntentEnum.GENERAL_CHAT,
                matched_cars=None,
                referenced_listing_id=None,
                replace_result_set=False
            )
            return ChatResponse(
                user_id=user_id,
                session_id=session_id,
                response=response_text,
                matched_cars=None,
                intent=UserIntentEnum.GENERAL_CHAT,
                total_matches=0,
                requires_clarification=False
            )

        # 8. CLEAR_LIKED_CARS
        elif lt_res.action == LongTermMemoryAction.CLEAR_LIKED_CARS:
            self.persistent_memory.clear_liked_cars(user_id)
            response_text = GroundedResponseBuilder.format_clear_confirmation("liked_cars")
            self.memory_service.record_turn(
                user_id=user_id,
                session_id=session_id,
                user_message=request.message,
                assistant_response=response_text,
                intent=UserIntentEnum.GENERAL_CHAT,
                matched_cars=None,
                referenced_listing_id=None,
                replace_result_set=False
            )
            return ChatResponse(
                user_id=user_id,
                session_id=session_id,
                response=response_text,
                matched_cars=None,
                intent=UserIntentEnum.GENERAL_CHAT,
                total_matches=0,
                requires_clarification=False
            )

        # 9. CLEAR_ALL_MEMORY
        elif lt_res.action == LongTermMemoryAction.CLEAR_ALL_MEMORY:
            self.persistent_memory.delete_user_data(user_id)
            response_text = GroundedResponseBuilder.format_clear_confirmation("all")
            self.memory_service.record_turn(
                user_id=user_id,
                session_id=session_id,
                user_message=request.message,
                assistant_response=response_text,
                intent=UserIntentEnum.GENERAL_CHAT,
                matched_cars=None,
                referenced_listing_id=None,
                replace_result_set=False
            )
            return ChatResponse(
                user_id=user_id,
                session_id=session_id,
                response=response_text,
                matched_cars=None,
                intent=UserIntentEnum.GENERAL_CHAT,
                total_matches=0,
                requires_clarification=False
            )

        # Fallback
        return ChatResponse(
            user_id=user_id,
            session_id=session_id,
            response="I've processed your request.",
            matched_cars=None,
            intent=UserIntentEnum.GENERAL_CHAT,
            total_matches=0,
            requires_clarification=False
        )

    @staticmethod
    def _has_searchable_criteria(query_filters) -> bool:
        """Returns True if any meaningful filter is set on the query."""
        fields = [
            query_filters.make,
            query_filters.model,
            query_filters.min_year,
            query_filters.max_year,
            query_filters.min_price_aed,
            query_filters.max_price_aed,
            query_filters.min_mileage_km,
            query_filters.max_mileage_km,
            query_filters.min_monthly_aed,
            query_filters.max_monthly_aed,
            query_filters.regional_specs,
            query_filters.warranty,
            query_filters.keywords,
        ]
        return any(f is not None for f in fields)

