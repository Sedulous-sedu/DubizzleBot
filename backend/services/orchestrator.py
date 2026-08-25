"""Chat orchestrator connecting QueryInterpreter, InventoryService, and GroundedResponseBuilder."""

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
from backend.services.query_interpreter import QueryInterpreter
from backend.services.inventory import InventoryService
from backend.services.response_builder import GroundedResponseBuilder

logger = logging.getLogger(__name__)

class ChatOrchestrator:
    """Core domain orchestrator coordinating NLP interpretation and deterministic inventory retrieval."""

    def __init__(
        self,
        query_interpreter: Optional[QueryInterpreter] = None,
        inventory_service: Optional[InventoryService] = None,
    ):
        self.query_interpreter = query_interpreter or QueryInterpreter()
        self.inventory_service = inventory_service or InventoryService()

    def process_chat(self, request: ChatRequest) -> ChatResponse:
        """
        Executes end-to-end processing for incoming user chat request.
        Preserves or generates session_id and routes deterministically by intent and readiness state.
        """
        session_id = request.session_id or str(uuid.uuid4())

        try:
            parsed_intent: ParsedUserIntent = self.query_interpreter.interpret(request.message)

            # Route by primary intent
            if parsed_intent.intent == UserIntentEnum.INVENTORY_SEARCH:
                return self._handle_inventory_search(request.user_id, session_id, parsed_intent)

            elif parsed_intent.intent == UserIntentEnum.VIEWING_OR_LEAD_REQUEST:
                return self._handle_viewing_request(request.user_id, session_id, parsed_intent)

            elif parsed_intent.intent == UserIntentEnum.GENERAL_CHAT:
                response_text = GroundedResponseBuilder.format_general_chat_response(request.message)
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
            return ChatResponse(
                user_id=request.user_id,
                session_id=session_id,
                response=(
                    "I apologize, but I encountered an issue processing your request. "
                    "Please try again or rephrase your search criteria."
                ),
                matched_cars=None,
                intent=UserIntentEnum.UNKNOWN,
                total_matches=0,
                requires_clarification=False
            )

    def _handle_inventory_search(
        self,
        user_id: str,
        session_id: str,
        parsed_intent: ParsedUserIntent
    ) -> ChatResponse:
        """Handles inventory search according to readiness state."""
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
        parsed_intent: ParsedUserIntent
    ) -> ChatResponse:
        """Handles viewing / lead requests according to readiness state and filter presence."""
        if parsed_intent.readiness_state == SearchReadinessState.CLARIFICATION_REQUIRED:
            response_text = GroundedResponseBuilder.format_clarification_response(
                parsed_intent.clarification_question
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
