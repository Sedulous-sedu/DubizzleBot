"""Chat orchestrator connecting MemoryService, ContextResolver, QueryInterpreter, InventoryService, and GroundedResponseBuilder."""

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
from backend.services.query_interpreter import QueryInterpreter
from backend.services.inventory import InventoryService
from backend.services.response_builder import GroundedResponseBuilder
from backend.services.memory import MemoryService
from backend.services.context_resolver import ContextResolver

logger = logging.getLogger(__name__)

class ChatOrchestrator:
    """Core domain orchestrator coordinating session memory, contextual resolution, NLP interpretation, and inventory retrieval."""

    def __init__(
        self,
        query_interpreter: Optional[QueryInterpreter] = None,
        inventory_service: Optional[InventoryService] = None,
        memory_service: Optional[MemoryService] = None,
    ):
        self.query_interpreter = query_interpreter or QueryInterpreter()
        self.inventory_service = inventory_service or InventoryService()
        self.memory_service = memory_service or MemoryService()

    def process_chat(self, request: ChatRequest) -> ChatResponse:
        """
        Executes end-to-end processing for incoming user chat request.
        Preserves or generates session_id, evaluates short-term contextual references,
        and routes deterministically by intent and readiness state.
        """
        session_id = request.session_id or str(uuid.uuid4())
        session = self.memory_service.get_or_create_session(request.user_id, session_id)

        try:
            # 1. Evaluate deterministic ContextResolver on active session state
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

            # 2. Fresh query: Route via QueryInterpreter and Phase 3B logic
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
