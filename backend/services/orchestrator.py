"""Chat orchestrator connecting MemoryService, PersistentMemoryService, ContextResolver, LongTermMemoryResolver, BookingService, LeadService, Phase5Resolver, QueryInterpreter, InventoryService, and GroundedResponseBuilder."""

import uuid
import logging
import re
from datetime import datetime, timezone
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
    PendingSupportedSearch,
    SessionState,
)
from backend.models.persistent_memory import (
    LongTermMemoryAction,
    LongTermMemoryResolution,
)
from backend.models.booking import (
    BookingDraft,
    ConfirmedBooking,
    WorkflowStatus,
    BookingStatus,
)
from backend.models.lead import (
    LeadDraft,
    QualifiedLead,
)
from backend.services.query_interpreter import QueryInterpreter
from backend.services.inventory import InventoryService
from backend.services.response_builder import GroundedResponseBuilder
from backend.services.memory import MemoryService
from backend.services.persistent_memory import PersistentMemoryService
from backend.services.context_resolver import ContextResolver
from backend.services.long_term_resolver import LongTermMemoryResolver
from backend.services.booking import BookingService
from backend.services.lead import LeadService
from backend.services.phase5_resolver import Phase5Resolver, Phase5Action, Phase5Resolution

logger = logging.getLogger(__name__)

class ChatOrchestrator:
    """Core domain orchestrator coordinating session memory, persistent memory, bookings, leads, contextual resolution, NLP interpretation, and inventory retrieval."""

    def __init__(
        self,
        query_interpreter: Optional[QueryInterpreter] = None,
        inventory_service: Optional[InventoryService] = None,
        memory_service: Optional[MemoryService] = None,
        persistent_memory: Optional[PersistentMemoryService] = None,
        booking_service: Optional[BookingService] = None,
        lead_service: Optional[LeadService] = None,
        phase5_resolver: Optional[Phase5Resolver] = None,
        current_time_override: Optional[datetime] = None,
    ):
        self.query_interpreter = query_interpreter or QueryInterpreter()
        self.inventory_service = inventory_service or InventoryService()
        self.memory_service = memory_service or MemoryService()
        self.persistent_memory = persistent_memory or PersistentMemoryService()
        self.booking_service = booking_service or BookingService(persistent_memory=self.persistent_memory)
        self.lead_service = lead_service or LeadService()
        self.phase5_resolver = phase5_resolver or Phase5Resolver()
        self.current_time_override = current_time_override

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
            # 1. Phase 5 Workflow Resolution (Active drafts continuation or new actions)
            phase5_res: Phase5Resolution = self.phase5_resolver.evaluate(
                request.message, session, current_time=self.current_time_override
            )
            if phase5_res.action != Phase5Action.NOT_PHASE5:
                if phase5_res.action == Phase5Action.START_BOOKING:
                    context_cand = ContextResolver.resolve(request.message, session)
                    has_specific_car = (
                        context_cand.status == ResolutionStatus.RESOLVED
                        or session.active_listing_id is not None
                        or len(session.current_result_set) == 1
                    )
                    if not has_specific_car and any(w in request.message.lower() for w in ["bentley", "toyota", "nissan", "ford", "under", "gcc", "suv", "sedan", "coupe", "bmw", "mercedes", "honda", "porsche", "land rover", "range rover"]):
                        pass  # Fall through to Phase 3A VIEWING_OR_LEAD_REQUEST candidate retrieval
                    else:
                        return self._handle_phase5(request, session_id, session, phase5_res)
                else:
                    return self._handle_phase5(request, session_id, session, phase5_res)

            # 2. Evaluate deterministic LongTermMemoryResolver (Phase 4B)
            lt_res: LongTermMemoryResolution = LongTermMemoryResolver.evaluate(request.message, session)
            if lt_res.action != LongTermMemoryAction.NOT_MEMORY_ACTION:
                return self._handle_long_term_memory(request, session_id, lt_res)

            # 2. Evaluate deterministic ContextResolver on active session state (Phase 4A)
            context_result: ContextResolutionResult = ContextResolver.resolve(request.message, session)

            if context_result.status == ResolutionStatus.RESULT_SET_COMPARISON:
                matching_cars = context_result.resolved_cars or []
                target_year = context_result.comparison_year
                comp_type = context_result.comparison_type
                response_text = GroundedResponseBuilder.format_result_set_comparison_response(
                    matching_cars, comp_type, target_year
                )
                self.memory_service.record_turn(
                    user_id=request.user_id,
                    session_id=session_id,
                    user_message=request.message,
                    assistant_response=response_text,
                    intent=UserIntentEnum.INVENTORY_SEARCH,
                    matched_cars=matching_cars,
                    referenced_listing_id=None,
                    replace_result_set=False,
                    active_listing_id=session.active_listing_id
                )
                return ChatResponse(
                    user_id=request.user_id,
                    session_id=session_id,
                    response=response_text,
                    matched_cars=matching_cars,
                    intent=UserIntentEnum.INVENTORY_SEARCH,
                    total_matches=len(matching_cars),
                    requires_clarification=False
                )

            elif context_result.status == ResolutionStatus.RESOLVED:
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

            # 3. Resolve a pending unsupported-search confirmation only after
            # workflow, long-term-memory, and contextual-reference routing.
            if session.pending_supported_search is not None:
                confirmation = self._supported_search_confirmation(request.message)
                if confirmation is True:
                    pending = session.pending_supported_search
                    session.pending_supported_search = None
                    self.memory_service.save_session(session)
                    confirmed_intent = ParsedUserIntent(
                        intent=UserIntentEnum.INVENTORY_SEARCH,
                        query_filters=pending.query_filters,
                        requires_clarification=False,
                        clarification_question=None,
                        unsupported_constraints=[],
                        readiness_state=SearchReadinessState.READY,
                    )
                    return self._handle_inventory_search(
                        request.user_id,
                        session_id,
                        request.message,
                        confirmed_intent,
                    )
                if confirmation is False:
                    session.pending_supported_search = None
                    self.memory_service.save_session(session)
                    response_text = "Okay — I won't run that search."
                    self.memory_service.record_turn(
                        user_id=request.user_id,
                        session_id=session_id,
                        user_message=request.message,
                        assistant_response=response_text,
                        intent=UserIntentEnum.INVENTORY_SEARCH,
                        matched_cars=None,
                        referenced_listing_id=None,
                        replace_result_set=False,
                    )
                    return ChatResponse(
                        user_id=request.user_id,
                        session_id=session_id,
                        response=response_text,
                        matched_cars=None,
                        intent=UserIntentEnum.INVENTORY_SEARCH,
                        total_matches=0,
                        requires_clarification=False,
                    )

            # 4. Fresh query: Route via QueryInterpreter and Phase 3B logic
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
                session = self.memory_service.get_or_create_session(user_id, session_id)
                session.pending_supported_search = None
                self.memory_service.save_session(session)
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
            if parsed_intent.query_filters and self._has_searchable_criteria(parsed_intent.query_filters):
                session = self.memory_service.get_or_create_session(user_id, session_id)
                session.pending_supported_search = PendingSupportedSearch(
                    query_filters=parsed_intent.query_filters,
                    unsupported_constraints=parsed_intent.unsupported_constraints,
                )
                self.memory_service.save_session(session)
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

    def _handle_phase5(
        self,
        request: ChatRequest,
        session_id: str,
        session: SessionState,
        phase5_res: Phase5Resolution
    ) -> ChatResponse:
        """Handles Phase 5 test-drive bookings and lead qualification workflows."""
        user_id = request.user_id

        # -------------------------------------------------------------
        # 1. START_BOOKING
        # -------------------------------------------------------------
        if phase5_res.action == Phase5Action.START_BOOKING:
            # Resolve target vehicle
            ctx_res = ContextResolver.resolve(request.message, session)
            target_car = None
            if ctx_res.status == ResolutionStatus.RESOLVED:
                target_car = ctx_res.resolved_car
            elif session.active_listing_id is not None:
                target_car = self.inventory_service.get_by_listing_id(session.active_listing_id)
            elif len(session.current_result_set) == 1:
                target_car = session.current_result_set[0]

            if target_car is None:
                response_text = GroundedResponseBuilder.format_viewing_clarification_response()
                self.memory_service.record_turn(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=request.message,
                    assistant_response=response_text,
                    intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                    matched_cars=None,
                    referenced_listing_id=None,
                    replace_result_set=False,
                )
                return ChatResponse(
                    user_id=user_id,
                    session_id=session_id,
                    response=response_text,
                    matched_cars=None,
                    intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                    total_matches=0,
                    requires_clarification=True,
                )

            draft = BookingDraft(
                user_id=user_id,
                session_id=session_id,
                listing_id=target_car.listing_id,
                target_car=target_car,
                requested_date=phase5_res.date_val,
                requested_time=phase5_res.time_val,
                requested_date_str=phase5_res.raw_date_str,
                requested_time_str=phase5_res.raw_time_str,
                customer_name=phase5_res.extracted_name,
                customer_phone=phase5_res.extracted_phone,
                customer_email=phase5_res.extracted_email,
            )
            session.pending_booking = draft

            if phase5_res.is_ambiguous_time:
                response_text = phase5_res.clarification_prompt or "Do you mean AM or PM? (Our business hours are 8:00 AM to 8:00 PM Asia/Dubai)."
                self.memory_service.record_turn(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=request.message,
                    assistant_response=response_text,
                    intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                    matched_cars=[target_car],
                    referenced_listing_id=target_car.listing_id,
                    replace_result_set=False,
                )
                return ChatResponse(
                    user_id=user_id,
                    session_id=session_id,
                    response=response_text,
                    matched_cars=[target_car],
                    intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                    total_matches=1,
                    requires_clarification=True,
                )

            if draft.requested_date is None and draft.requested_time is None:
                response_text = (
                    f"I can help arrange a test drive for the {target_car.year} {target_car.make} {target_car.model} "
                    f"(Listing #{target_car.listing_id}). What date and time between Monday–Saturday, 8:00 AM to 8:00 PM would you prefer?"
                )
                self.memory_service.record_turn(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=request.message,
                    assistant_response=response_text,
                    intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                    matched_cars=[target_car],
                    referenced_listing_id=target_car.listing_id,
                    replace_result_set=False,
                )
                return ChatResponse(
                    user_id=user_id,
                    session_id=session_id,
                    response=response_text,
                    matched_cars=[target_car],
                    intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                    total_matches=1,
                    requires_clarification=False,
                )
            elif draft.requested_date is None:
                response_text = "What date would you like your appointment? We are open Monday through Saturday."
                self.memory_service.record_turn(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=request.message,
                    assistant_response=response_text,
                    intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                    matched_cars=[target_car],
                    referenced_listing_id=target_car.listing_id,
                    replace_result_set=False,
                )
                return ChatResponse(
                    user_id=user_id,
                    session_id=session_id,
                    response=response_text,
                    matched_cars=[target_car],
                    intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                    total_matches=1,
                    requires_clarification=False,
                )
            elif draft.requested_time is None:
                d_str = draft.requested_date.strftime("%A, %B %d")
                response_text = f"What time on {d_str} would you like your appointment? We are open from 8:00 AM to 8:00 PM."
                self.memory_service.record_turn(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=request.message,
                    assistant_response=response_text,
                    intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                    matched_cars=[target_car],
                    referenced_listing_id=target_car.listing_id,
                    replace_result_set=False,
                )
                return ChatResponse(
                    user_id=user_id,
                    session_id=session_id,
                    response=response_text,
                    matched_cars=[target_car],
                    intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                    total_matches=1,
                    requires_clarification=False,
                )
            else:
                dt = datetime.combine(draft.requested_date, draft.requested_time).replace(tzinfo=self.booking_service.get_timezone())
                val_res = self.booking_service.validate_appointment(dt, current_time=self.current_time_override)
                if not val_res.is_valid:
                    response_text = val_res.error_message or "Please select an appointment time within our business hours."
                    self.memory_service.record_turn(
                        user_id=user_id,
                        session_id=session_id,
                        user_message=request.message,
                        assistant_response=response_text,
                        intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                        matched_cars=[target_car],
                        referenced_listing_id=target_car.listing_id,
                        replace_result_set=False,
                    )
                    return ChatResponse(
                        user_id=user_id,
                        session_id=session_id,
                        response=response_text,
                        matched_cars=[target_car],
                        intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                        total_matches=1,
                        requires_clarification=True,
                    )
                else:
                    draft.appointment_at = dt
                    draft.status = WorkflowStatus.AWAITING_CONFIRMATION
                    response_text = GroundedResponseBuilder.format_booking_summary_for_confirmation(draft, tz_name=self.booking_service.timezone_name)
                    self.memory_service.record_turn(
                        user_id=user_id,
                        session_id=session_id,
                        user_message=request.message,
                        assistant_response=response_text,
                        intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                        matched_cars=[target_car],
                        referenced_listing_id=target_car.listing_id,
                        replace_result_set=False,
                    )
                    return ChatResponse(
                        user_id=user_id,
                        session_id=session_id,
                        response=response_text,
                        matched_cars=[target_car],
                        intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                        total_matches=1,
                        requires_clarification=False,
                    )

        # -------------------------------------------------------------
        # 2. CONTINUE_BOOKING
        # -------------------------------------------------------------
        elif phase5_res.action == Phase5Action.CONTINUE_BOOKING:
            draft = session.pending_booking
            if not draft or not draft.target_car:
                response_text = "Which vehicle would you like to book a test drive for?"
                return ChatResponse(user_id=user_id, session_id=session_id, response=response_text, total_matches=0, intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST)

            if phase5_res.is_ambiguous_time:
                response_text = phase5_res.clarification_prompt or "Do you mean AM or PM? (Our business hours are 8:00 AM to 8:00 PM Asia/Dubai)."
                self.memory_service.record_turn(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=request.message,
                    assistant_response=response_text,
                    intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                    matched_cars=[draft.target_car],
                    referenced_listing_id=draft.listing_id,
                    replace_result_set=False,
                )
                return ChatResponse(
                    user_id=user_id,
                    session_id=session_id,
                    response=response_text,
                    matched_cars=[draft.target_car],
                    intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                    total_matches=1,
                    requires_clarification=True,
                )

            if phase5_res.date_val:
                draft.requested_date = phase5_res.date_val
                draft.requested_date_str = phase5_res.raw_date_str
            if phase5_res.time_val:
                draft.requested_time = phase5_res.time_val
                draft.requested_time_str = phase5_res.raw_time_str
            if phase5_res.extracted_name:
                draft.customer_name = phase5_res.extracted_name
            if phase5_res.extracted_phone:
                draft.customer_phone = phase5_res.extracted_phone
            if phase5_res.extracted_email:
                draft.customer_email = phase5_res.extracted_email

            if draft.requested_date is None and draft.requested_time is None:
                response_text = "What date and time would you prefer for your appointment? We are open Monday through Saturday, 8:00 AM to 8:00 PM."
            elif draft.requested_date is None:
                response_text = "What date would you like your appointment? We are open Monday through Saturday."
            elif draft.requested_time is None:
                d_str = draft.requested_date.strftime("%A, %B %d")
                response_text = f"What time on {d_str} would you like your appointment? We are open from 8:00 AM to 8:00 PM."
            else:
                dt = datetime.combine(draft.requested_date, draft.requested_time).replace(tzinfo=self.booking_service.get_timezone())
                val_res = self.booking_service.validate_appointment(dt, current_time=self.current_time_override)
                if not val_res.is_valid:
                    response_text = val_res.error_message or "Please select an appointment time within our business hours."
                    draft.status = WorkflowStatus.COLLECTING
                else:
                    draft.appointment_at = dt
                    draft.status = WorkflowStatus.AWAITING_CONFIRMATION
                    response_text = GroundedResponseBuilder.format_booking_summary_for_confirmation(draft, tz_name=self.booking_service.timezone_name)

            self.memory_service.record_turn(
                user_id=user_id,
                session_id=session_id,
                user_message=request.message,
                assistant_response=response_text,
                intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                matched_cars=[draft.target_car],
                referenced_listing_id=draft.listing_id,
                replace_result_set=False,
            )
            return ChatResponse(
                user_id=user_id,
                session_id=session_id,
                response=response_text,
                matched_cars=[draft.target_car],
                intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                total_matches=1,
                requires_clarification=(draft.status != WorkflowStatus.AWAITING_CONFIRMATION),
            )

        # -------------------------------------------------------------
        # 3. CONFIRM_BOOKING
        # -------------------------------------------------------------
        elif phase5_res.action == Phase5Action.CONFIRM_BOOKING:
            draft = session.pending_booking
            if draft and draft.status == WorkflowStatus.AWAITING_CONFIRMATION and draft.appointment_at and draft.listing_id:
                confirmed = ConfirmedBooking(
                    booking_id=draft.booking_id,
                    user_id=user_id,
                    listing_id=draft.listing_id,
                    appointment_at=draft.appointment_at,
                    customer_name=draft.customer_name,
                    customer_phone=draft.customer_phone,
                    customer_email=draft.customer_email,
                    status=BookingStatus.CONFIRMED,
                    created_at=datetime.now(timezone.utc),
                )
                self.booking_service.save_booking(confirmed)
                saved_car = draft.target_car or self.inventory_service.get_by_listing_id(draft.listing_id)
                session.pending_booking = None
                response_text = GroundedResponseBuilder.format_booking_confirmed_response(confirmed, saved_car)
                self.memory_service.record_turn(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=request.message,
                    assistant_response=response_text,
                    intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                    matched_cars=[saved_car] if saved_car else None,
                    referenced_listing_id=draft.listing_id,
                    replace_result_set=False,
                )
                return ChatResponse(
                    user_id=user_id,
                    session_id=session_id,
                    response=response_text,
                    matched_cars=[saved_car] if saved_car else None,
                    intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                    total_matches=1 if saved_car else 0,
                    requires_clarification=False,
                )
            else:
                response_text = "No pending test-drive booking found to confirm. Please let me know which vehicle you'd like to book."
                self.memory_service.record_turn(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=request.message,
                    assistant_response=response_text,
                    intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                    matched_cars=None,
                    referenced_listing_id=None,
                    replace_result_set=False,
                )
                return ChatResponse(
                    user_id=user_id,
                    session_id=session_id,
                    response=response_text,
                    matched_cars=None,
                    intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                    total_matches=0,
                    requires_clarification=False,
                )

        # -------------------------------------------------------------
        # 4. CANCEL_BOOKING
        # -------------------------------------------------------------
        elif phase5_res.action == Phase5Action.CANCEL_BOOKING:
            session.pending_booking = None
            response_text = GroundedResponseBuilder.format_booking_cancelled_response()
            self.memory_service.record_turn(
                user_id=user_id,
                session_id=session_id,
                user_message=request.message,
                assistant_response=response_text,
                intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                matched_cars=None,
                referenced_listing_id=None,
                replace_result_set=False,
            )
            return ChatResponse(
                user_id=user_id,
                session_id=session_id,
                response=response_text,
                matched_cars=None,
                intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                total_matches=0,
                requires_clarification=False,
            )

        # -------------------------------------------------------------
        # 5. START_LEAD
        # -------------------------------------------------------------
        elif phase5_res.action == Phase5Action.START_LEAD:
            prefs = self.persistent_memory.get_preferences(user_id)
            seed_min_b = prefs.min_price_aed if (prefs and prefs.has_explicit_preferences()) else None
            seed_max_b = prefs.max_price_aed if (prefs and prefs.has_explicit_preferences()) else None
            seed_make = prefs.preferred_make if (prefs and prefs.has_explicit_preferences()) else None
            seed_model = prefs.preferred_model if (prefs and prefs.has_explicit_preferences()) else None
            seed_reqs = prefs.regional_specs if (prefs and prefs.has_explicit_preferences() and prefs.regional_specs) else None

            min_b = phase5_res.extracted_min_budget if phase5_res.extracted_min_budget is not None else seed_min_b
            max_b = phase5_res.extracted_max_budget if phase5_res.extracted_max_budget is not None else seed_max_b
            reqs = phase5_res.extracted_requirements or seed_reqs
            interested_lid = session.active_listing_id if session.active_listing_id else (session.current_result_set[0].listing_id if len(session.current_result_set) == 1 else None)

            ldraft = LeadDraft(
                user_id=user_id,
                session_id=session_id,
                name=phase5_res.extracted_name,
                phone=phase5_res.extracted_phone,
                email=phase5_res.extracted_email,
                min_budget_aed=min_b,
                max_budget_aed=max_b,
                interested_make=seed_make,
                interested_model=seed_model,
                interested_listing_id=interested_lid,
                requirements=reqs,
            )
            session.pending_lead = ldraft

            if not ldraft.has_budget():
                response_text = "What is your target budget or price range?"
            elif not ldraft.has_automotive_need():
                response_text = "What specific vehicle or requirements are you looking for (make, model, or specs)?"
            elif not ldraft.has_contact():
                response_text = "Please provide your phone number or email address so our sales representative can contact you."
            else:
                ldraft.status = WorkflowStatus.AWAITING_CONFIRMATION
                response_text = GroundedResponseBuilder.format_lead_summary_for_confirmation(ldraft)

            self.memory_service.record_turn(
                user_id=user_id,
                session_id=session_id,
                user_message=request.message,
                assistant_response=response_text,
                intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                matched_cars=None,
                referenced_listing_id=None,
                replace_result_set=False,
            )
            return ChatResponse(
                user_id=user_id,
                session_id=session_id,
                response=response_text,
                matched_cars=None,
                intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                total_matches=0,
                requires_clarification=(ldraft.status != WorkflowStatus.AWAITING_CONFIRMATION),
            )

        # -------------------------------------------------------------
        # 6. CONTINUE_LEAD
        # -------------------------------------------------------------
        elif phase5_res.action == Phase5Action.CONTINUE_LEAD:
            ldraft = session.pending_lead
            if not ldraft:
                return ChatResponse(user_id=user_id, session_id=session_id, response="No active enquiry found. What vehicle are you looking for?", total_matches=0, intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST)

            if phase5_res.extracted_name:
                ldraft.name = phase5_res.extracted_name
            if phase5_res.extracted_phone:
                ldraft.phone = phase5_res.extracted_phone
            if phase5_res.extracted_email:
                ldraft.email = phase5_res.extracted_email
            if phase5_res.extracted_min_budget is not None:
                ldraft.min_budget_aed = phase5_res.extracted_min_budget
            if phase5_res.extracted_max_budget is not None:
                ldraft.max_budget_aed = phase5_res.extracted_max_budget
            if phase5_res.extracted_requirements:
                ldraft.requirements = phase5_res.extracted_requirements

            if not ldraft.has_budget():
                response_text = "What is your target budget or price range?"
            elif not ldraft.has_automotive_need():
                response_text = "What specific vehicle or requirements are you looking for?"
            elif not ldraft.has_contact():
                response_text = "Please provide your phone number or email address so we can reach you."
            else:
                ldraft.status = WorkflowStatus.AWAITING_CONFIRMATION
                response_text = GroundedResponseBuilder.format_lead_summary_for_confirmation(ldraft)

            self.memory_service.record_turn(
                user_id=user_id,
                session_id=session_id,
                user_message=request.message,
                assistant_response=response_text,
                intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                matched_cars=None,
                referenced_listing_id=None,
                replace_result_set=False,
            )
            return ChatResponse(
                user_id=user_id,
                session_id=session_id,
                response=response_text,
                matched_cars=None,
                intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                total_matches=0,
                requires_clarification=(ldraft.status != WorkflowStatus.AWAITING_CONFIRMATION),
            )

        # -------------------------------------------------------------
        # 7. CONFIRM_LEAD
        # -------------------------------------------------------------
        elif phase5_res.action == Phase5Action.CONFIRM_LEAD:
            ldraft = session.pending_lead
            if ldraft and ldraft.status == WorkflowStatus.AWAITING_CONFIRMATION:
                qlead = QualifiedLead(
                    lead_id=ldraft.lead_id,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    user_id=user_id,
                    session_id=session_id,
                    name=ldraft.name,
                    phone=ldraft.phone,
                    email=ldraft.email,
                    min_budget_aed=ldraft.min_budget_aed,
                    max_budget_aed=ldraft.max_budget_aed,
                    interested_make=ldraft.interested_make,
                    interested_model=ldraft.interested_model,
                    interested_listing_id=ldraft.interested_listing_id,
                    requirements=ldraft.requirements,
                    booking_reference=ldraft.booking_reference,
                )
                self.lead_service.save_lead(qlead)
                session.pending_lead = None
                response_text = GroundedResponseBuilder.format_lead_submitted_response(qlead)
                self.memory_service.record_turn(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=request.message,
                    assistant_response=response_text,
                    intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                    matched_cars=None,
                    referenced_listing_id=None,
                    replace_result_set=False,
                )
                return ChatResponse(
                    user_id=user_id,
                    session_id=session_id,
                    response=response_text,
                    matched_cars=None,
                    intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                    total_matches=0,
                    requires_clarification=False,
                )
            else:
                response_text = "No pending enquiry found to confirm. Please let me know what vehicle or budget you are interested in."
                self.memory_service.record_turn(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=request.message,
                    assistant_response=response_text,
                    intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                    matched_cars=None,
                    referenced_listing_id=None,
                    replace_result_set=False,
                )
                return ChatResponse(
                    user_id=user_id,
                    session_id=session_id,
                    response=response_text,
                    matched_cars=None,
                    intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                    total_matches=0,
                    requires_clarification=False,
                )

        # -------------------------------------------------------------
        # 8. CANCEL_LEAD
        # -------------------------------------------------------------
        elif phase5_res.action == Phase5Action.CANCEL_LEAD:
            session.pending_lead = None
            response_text = GroundedResponseBuilder.format_lead_cancelled_response()
            self.memory_service.record_turn(
                user_id=user_id,
                session_id=session_id,
                user_message=request.message,
                assistant_response=response_text,
                intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                matched_cars=None,
                referenced_listing_id=None,
                replace_result_set=False,
            )
            return ChatResponse(
                user_id=user_id,
                session_id=session_id,
                response=response_text,
                matched_cars=None,
                intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                total_matches=0,
                requires_clarification=False,
            )

        # Fallback
        return ChatResponse(
            user_id=user_id,
            session_id=session_id,
            response="I've processed your request.",
            matched_cars=None,
            intent=UserIntentEnum.GENERAL_CHAT,
            total_matches=0,
            requires_clarification=False,
        )

    @staticmethod
    def _supported_search_confirmation(message: str) -> Optional[bool]:
        """Classifies only explicit standalone replies to a pending supported search."""
        normalized = re.sub(r"[^a-z\s]", "", message.strip().lower())
        normalized = re.sub(r"\s+", " ", normalized)
        if normalized in {"yes", "yes please", "sure", "okay", "ok", "go ahead"}:
            return True
        if normalized in {"no", "no thanks", "never mind", "nevermind", "cancel"}:
            return False
        return None

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
