"""Unit and integration tests for Phase 3B Grounded Chat Orchestrator and Response Builder."""

import uuid
import pytest
from unittest.mock import MagicMock
from backend.models.car import CarListing
from backend.models.chat import ChatRequest, ChatResponse
from backend.models.intent import (
    UserIntentEnum,
    RegionalSpecEnum,
    SearchReadinessState,
    UnsupportedConstraint,
    ParsedInventoryQuery,
    ParsedUserIntent,
)
from backend.services.orchestrator import ChatOrchestrator
from backend.services.response_builder import GroundedResponseBuilder
from backend.services.inventory import InventoryService
from backend.services.memory import MemoryService

@pytest.fixture
def real_inventory():
    """Real deterministic InventoryService instance loaded with dataset."""
    return InventoryService()

@pytest.fixture
def mock_interpreter():
    """Mocked QueryInterpreter for 100% deterministic offline tests."""
    return MagicMock()

# ==============================================================================
# 1. WARRANTY STATUS PRESENTATION & REGRESSION TESTS
# ==============================================================================

def test_warranty_status_preserves_option_available_semantics():
    """Regression test proving has_positive_warranty=False with Option Available is not rendered as 'No Warranty'."""
    listing = CarListing(
        listing_id=999,
        year=2021,
        make="Ford",
        model="Explorer",
        trim="XLT",
        title="2021 Ford Explorer",
        description="Great condition, warranty available upon request.",
        price_aed=120000.0,
        has_positive_warranty=False,
        warranty_status="Warranty Option Available (Not Active)",
        mileage_km=45000,
        regional_specs="GCC",
        body_type="SUV"
    )
    formatted = GroundedResponseBuilder._format_listing_line(listing)
    assert "Warranty: Warranty Option Available (Not Active)" in formatted
    assert "Warranty: No Warranty" not in formatted
    assert "Warranty: No / Not Active" not in formatted

def test_warranty_status_positive_agency():
    """Test positive active agency warranty presentation."""
    listing = CarListing(
        listing_id=998,
        year=2022,
        make="Bentley",
        model="Bentayga",
        trim="V8",
        title="2022 Bentley Bentayga",
        description="Full Gargash warranty until 2026",
        price_aed=650000.0,
        has_positive_warranty=True,
        warranty_status="Agency Warranty",
        mileage_km=15000,
        regional_specs="GCC",
        body_type="SUV"
    )
    formatted = GroundedResponseBuilder._format_listing_line(listing)
    assert "Warranty: Agency Warranty" in formatted

# ==============================================================================
# 2. INVENTORY SEARCH ROUTING & RESPONSE TESTS
# ==============================================================================

def test_orchestrator_simple_inventory_search(mock_interpreter, real_inventory):
    """Test standard ready inventory search returns matching cars and grounded prose."""
    mock_interpreter.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(make="Bentley"),
        requires_clarification=False,
        clarification_question=None,
        unsupported_constraints=[],
        readiness_state=SearchReadinessState.READY
    )

    orchestrator = ChatOrchestrator(query_interpreter=mock_interpreter, inventory_service=real_inventory)
    req = ChatRequest(user_id="user_1", message="Show me Bentleys")
    res = orchestrator.process_chat(req)

    assert res.intent == UserIntentEnum.INVENTORY_SEARCH
    assert res.requires_clarification is False
    assert res.total_matches > 0
    assert res.matched_cars is not None
    assert len(res.matched_cars) > 0
    for car in res.matched_cars:
        assert car.make.lower() == "bentley"
    assert "Listing #" in res.response
    assert "bentley" in res.response.lower()

def test_orchestrator_combined_multi_filter_search(mock_interpreter, real_inventory):
    """Test multi-filter search (make, year, price, warranty) without unrequested GCC assertion."""
    mock_interpreter.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(
            make="Land Rover",
            min_year=2018,
            max_price_aed=150000.0,
            warranty=True
        ),
        requires_clarification=False,
        clarification_question=None,
        unsupported_constraints=[],
        readiness_state=SearchReadinessState.READY
    )

    orchestrator = ChatOrchestrator(query_interpreter=mock_interpreter, inventory_service=real_inventory)
    req = ChatRequest(user_id="user_1", message="Land Rover from 2018 under 150k with warranty")
    res = orchestrator.process_chat(req)

    assert res.intent == UserIntentEnum.INVENTORY_SEARCH
    if res.total_matches > 0:
        for car in res.matched_cars:
            assert car.make.lower() == "land rover"
            assert car.year >= 2018
            assert car.price_aed is not None and car.price_aed <= 150000.0
            assert car.has_positive_warranty is True

def test_orchestrator_keywords_search(mock_interpreter, real_inventory):
    """Test unstructured keywords search matching title or description."""
    mock_interpreter.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(keywords="panoramic"),
        requires_clarification=False,
        clarification_question=None,
        unsupported_constraints=[],
        readiness_state=SearchReadinessState.READY
    )

    orchestrator = ChatOrchestrator(query_interpreter=mock_interpreter, inventory_service=real_inventory)
    req = ChatRequest(user_id="user_1", message="Cars with panoramic roof")
    res = orchestrator.process_chat(req)

    assert res.intent == UserIntentEnum.INVENTORY_SEARCH
    assert res.total_matches > 0
    for car in res.matched_cars:
        text = f"{car.title} {car.description}".lower()
        assert "panoramic" in text

def test_orchestrator_explicit_result_limit(mock_interpreter, real_inventory):
    """Test that explicit user limit is strictly respected."""
    mock_interpreter.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(make="Ford", limit=3),
        requires_clarification=False,
        clarification_question=None,
        unsupported_constraints=[],
        readiness_state=SearchReadinessState.READY
    )

    orchestrator = ChatOrchestrator(query_interpreter=mock_interpreter, inventory_service=real_inventory)
    req = ChatRequest(user_id="user_1", message="Show 3 Ford cars")
    res = orchestrator.process_chat(req)

    assert res.intent == UserIntentEnum.INVENTORY_SEARCH
    assert len(res.matched_cars) == 3
    assert res.total_matches == 3

def test_orchestrator_zero_matches_no_hallucination(mock_interpreter, real_inventory):
    """Test zero-result search returns grounded non-match message with zero hallucinated cars."""
    mock_interpreter.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(make="Ferrari", max_price_aed=1000.0),
        requires_clarification=False,
        clarification_question=None,
        unsupported_constraints=[],
        readiness_state=SearchReadinessState.READY
    )

    orchestrator = ChatOrchestrator(query_interpreter=mock_interpreter, inventory_service=real_inventory)
    req = ChatRequest(user_id="user_1", message="Ferrari under 1000 AED")
    res = orchestrator.process_chat(req)

    assert res.intent == UserIntentEnum.INVENTORY_SEARCH
    assert res.total_matches == 0
    assert res.matched_cars == []
    assert "couldn't find any listings matching those exact criteria" in res.response

# ==============================================================================
# 3. CLARIFICATION & UNSUPPORTED CONSTRAINTS
# ==============================================================================

def _unsupported_toyota_ranking_intent():
    return ParsedUserIntent(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(make="Toyota"),
        requires_clarification=False,
        clarification_question=None,
        unsupported_constraints=[
            UnsupportedConstraint(
                field="ranking",
                requested_value="cheapest",
                reason="ranking_not_supported_by_inventory_filter",
            )
        ],
        readiness_state=SearchReadinessState.UNSUPPORTED_CONSTRAINTS_PRESENT,
    )

def test_orchestrator_clarification_required_no_inventory_search(mock_interpreter):
    """Test vague request returns clarification question without querying inventory."""
    mock_inventory = MagicMock()
    mock_interpreter.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=None,
        requires_clarification=True,
        clarification_question="What is your budget in AED and preferred maximum mileage?",
        unsupported_constraints=[],
        readiness_state=SearchReadinessState.CLARIFICATION_REQUIRED
    )

    orchestrator = ChatOrchestrator(query_interpreter=mock_interpreter, inventory_service=mock_inventory)
    req = ChatRequest(user_id="user_1", message="I want a cheap low mileage car")
    res = orchestrator.process_chat(req)

    assert res.intent == UserIntentEnum.INVENTORY_SEARCH
    assert res.requires_clarification is True
    assert res.matched_cars is None
    assert res.total_matches == 0
    assert res.response == "What is your budget in AED and preferred maximum mileage?"
    mock_inventory.search.assert_not_called()

def test_orchestrator_unsupported_ranking_no_partial_search(mock_interpreter):
    """Test unsupported ranking does NOT execute partial search and does NOT leak internal terms."""
    mock_inventory = MagicMock()
    mock_interpreter.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(make="Bentley"),
        requires_clarification=False,
        clarification_question=None,
        unsupported_constraints=[
            UnsupportedConstraint(
                field="ranking",
                requested_value="cheapest",
                reason="ranking_not_supported_by_inventory_filter"
            )
        ],
        readiness_state=SearchReadinessState.UNSUPPORTED_CONSTRAINTS_PRESENT
    )

    orchestrator = ChatOrchestrator(query_interpreter=mock_interpreter, inventory_service=mock_inventory)
    req = ChatRequest(user_id="user_1", message="Show me the 5 cheapest Bentleys")
    res = orchestrator.process_chat(req)

    assert res.intent == UserIntentEnum.INVENTORY_SEARCH
    assert res.matched_cars is None
    assert res.total_matches == 0
    mock_inventory.search.assert_not_called()
    assert "cheapest" in res.response
    assert "Bentley" in res.response
    assert "Listing ID" not in res.response
    assert "readiness_state" not in res.response
    assert "unsupported_constraints" not in res.response
    pending = orchestrator.memory_service.get_session("user_1", res.session_id).pending_supported_search
    assert pending is not None
    assert pending.query_filters.make == "Bentley"
    assert pending.unsupported_constraints[0].field == "ranking"

def test_pending_unsupported_search_confirmation_executes_supported_filters_only(
    mock_interpreter, real_inventory
):
    """An affirmative continuation deterministically runs only the saved Toyota filter."""
    inventory_spy = MagicMock(wraps=real_inventory)
    mock_interpreter.interpret.return_value = _unsupported_toyota_ranking_intent()
    orchestrator = ChatOrchestrator(
        query_interpreter=mock_interpreter,
        inventory_service=inventory_spy,
    )
    session_id = str(uuid.uuid4())

    first = orchestrator.process_chat(ChatRequest(
        user_id="toyota_yes_user",
        session_id=session_id,
        message="What is the cheapest Toyota?",
    ))
    assert first.total_matches == 0
    inventory_spy.search.assert_not_called()

    second = orchestrator.process_chat(ChatRequest(
        user_id="toyota_yes_user",
        session_id=session_id,
        message="yes",
    ))

    assert second.total_matches > 0
    assert second.matched_cars is not None
    assert all(car.make.lower() == "toyota" for car in second.matched_cars)
    search_filter = inventory_spy.search.call_args.args[0]
    assert search_filter.make == "Toyota"
    assert search_filter.limit is None
    assert orchestrator.memory_service.get_session(
        "toyota_yes_user", session_id
    ).pending_supported_search is None
    mock_interpreter.interpret.assert_called_once_with("What is the cheapest Toyota?")

def test_yes_please_executes_pending_supported_search(mock_interpreter, real_inventory):
    """The polite affirmative variant consumes the pending search."""
    inventory_spy = MagicMock(wraps=real_inventory)
    mock_interpreter.interpret.return_value = _unsupported_toyota_ranking_intent()
    orchestrator = ChatOrchestrator(
        query_interpreter=mock_interpreter,
        inventory_service=inventory_spy,
    )
    session_id = str(uuid.uuid4())
    orchestrator.process_chat(ChatRequest(
        user_id="toyota_yes_please_user",
        session_id=session_id,
        message="What is the cheapest Toyota?",
    ))

    result = orchestrator.process_chat(ChatRequest(
        user_id="toyota_yes_please_user",
        session_id=session_id,
        message="yes please",
    ))

    assert result.total_matches > 0
    assert all(car.make.lower() == "toyota" for car in result.matched_cars or [])
    inventory_spy.search.assert_called_once()
    assert mock_interpreter.interpret.call_count == 1

def test_no_thanks_clears_pending_supported_search_without_search(mock_interpreter):
    """A negative continuation clears pending state without inventory retrieval."""
    mock_inventory = MagicMock()
    mock_interpreter.interpret.return_value = _unsupported_toyota_ranking_intent()
    orchestrator = ChatOrchestrator(
        query_interpreter=mock_interpreter,
        inventory_service=mock_inventory,
    )
    session_id = str(uuid.uuid4())
    orchestrator.process_chat(ChatRequest(
        user_id="toyota_no_user",
        session_id=session_id,
        message="What is the cheapest Toyota?",
    ))

    result = orchestrator.process_chat(ChatRequest(
        user_id="toyota_no_user",
        session_id=session_id,
        message="No thanks",
    ))

    assert result.total_matches == 0
    assert result.matched_cars is None
    assert "won't run" in result.response
    mock_inventory.search.assert_not_called()
    assert orchestrator.memory_service.get_session(
        "toyota_no_user", session_id
    ).pending_supported_search is None

def test_unrelated_memory_recall_is_not_swallowed_by_pending_search(mock_interpreter):
    """Phase 4B memory recall retains priority and leaves the pending search available."""
    mock_inventory = MagicMock()
    mock_persistent_memory = MagicMock()
    mock_persistent_memory.get_liked_listing_ids.return_value = []
    mock_interpreter.interpret.return_value = _unsupported_toyota_ranking_intent()
    orchestrator = ChatOrchestrator(
        query_interpreter=mock_interpreter,
        inventory_service=mock_inventory,
        persistent_memory=mock_persistent_memory,
    )
    session_id = str(uuid.uuid4())
    orchestrator.process_chat(ChatRequest(
        user_id="memory_priority_user",
        session_id=session_id,
        message="What is the cheapest Toyota?",
    ))

    result = orchestrator.process_chat(ChatRequest(
        user_id="memory_priority_user",
        session_id=session_id,
        message="What cars did I like?",
    ))

    assert result.intent == UserIntentEnum.INVENTORY_SEARCH
    assert result.total_matches == 0
    mock_persistent_memory.get_liked_listing_ids.assert_called_once_with("memory_priority_user")
    assert mock_interpreter.interpret.call_count == 1
    assert orchestrator.memory_service.get_session(
        "memory_priority_user", session_id
    ).pending_supported_search is not None

def test_normal_toyota_search_remains_unchanged(mock_interpreter, real_inventory):
    """A normal supported Toyota query follows the existing deterministic path."""
    mock_interpreter.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(make="Toyota"),
        requires_clarification=False,
        clarification_question=None,
        unsupported_constraints=[],
        readiness_state=SearchReadinessState.READY,
    )
    orchestrator = ChatOrchestrator(
        query_interpreter=mock_interpreter,
        inventory_service=real_inventory,
    )

    result = orchestrator.process_chat(ChatRequest(
        user_id="normal_toyota_user",
        message="Show me Toyotas",
    ))

    assert result.total_matches > 0
    assert result.matched_cars is not None
    assert all(car.make.lower() == "toyota" for car in result.matched_cars)

@pytest.mark.parametrize("reply", ["sure", "okay", "go ahead"])
def test_other_explicit_affirmatives_execute_pending_search(
    reply, mock_interpreter, real_inventory
):
    """Documented standalone affirmative variants consume pending state."""
    inventory_spy = MagicMock(wraps=real_inventory)
    mock_interpreter.interpret.return_value = _unsupported_toyota_ranking_intent()
    orchestrator = ChatOrchestrator(
        query_interpreter=mock_interpreter,
        inventory_service=inventory_spy,
    )
    session_id = str(uuid.uuid4())
    orchestrator.process_chat(ChatRequest(
        user_id=f"affirmative_{reply}",
        session_id=session_id,
        message="What is the cheapest Toyota?",
    ))

    result = orchestrator.process_chat(ChatRequest(
        user_id=f"affirmative_{reply}",
        session_id=session_id,
        message=reply,
    ))

    assert result.total_matches > 0
    inventory_spy.search.assert_called_once()
    assert mock_interpreter.interpret.call_count == 1

# ==============================================================================
# 4. VIEWING INTENT & READINESS ROUTING
# ==============================================================================

def test_orchestrator_viewing_intent_with_candidate_filters(mock_interpreter, real_inventory):
    """Test viewing intent with vehicle criteria retrieves candidates and asks for specific Listing ID."""
    mock_interpreter.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
        query_filters=ParsedInventoryQuery(make="Bentley", regional_specs=RegionalSpecEnum.GCC),
        requires_clarification=False,
        clarification_question=None,
        unsupported_constraints=[],
        readiness_state=SearchReadinessState.READY
    )

    orchestrator = ChatOrchestrator(query_interpreter=mock_interpreter, inventory_service=real_inventory)
    req = ChatRequest(user_id="user_1", message="I want to test drive a GCC Bentley")
    res = orchestrator.process_chat(req)

    assert res.intent == UserIntentEnum.VIEWING_OR_LEAD_REQUEST
    assert res.total_matches > 0
    assert res.matched_cars is not None
    assert "arrange a viewing or test drive" in res.response
    assert "which specific Listing ID" in res.response
    assert "slot has been booked" not in res.response

def test_orchestrator_viewing_intent_without_filters(mock_interpreter, real_inventory):
    """Test viewing intent without vehicle criteria prompts user to select a vehicle."""
    mock_interpreter.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
        query_filters=None,
        requires_clarification=False,
        clarification_question=None,
        unsupported_constraints=[],
        readiness_state=SearchReadinessState.READY
    )

    orchestrator = ChatOrchestrator(query_interpreter=mock_interpreter, inventory_service=real_inventory)
    req = ChatRequest(user_id="user_1", message="I want to book a test drive")
    res = orchestrator.process_chat(req)

    assert res.intent == UserIntentEnum.VIEWING_OR_LEAD_REQUEST
    assert res.matched_cars is None
    assert res.total_matches == 0
    assert "Which vehicle from our inventory are you interested in viewing?" in res.response

def test_orchestrator_viewing_intent_clarification_required(mock_interpreter):
    """Test viewing intent with vague criteria triggers clarification without search."""
    mock_inventory = MagicMock()
    mock_interpreter.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
        query_filters=None,
        requires_clarification=True,
        clarification_question="Which vehicle make or model would you like to test drive?",
        unsupported_constraints=[],
        readiness_state=SearchReadinessState.CLARIFICATION_REQUIRED
    )

    orchestrator = ChatOrchestrator(query_interpreter=mock_interpreter, inventory_service=mock_inventory)
    req = ChatRequest(user_id="user_1", message="Book test drive for a cheap car")
    res = orchestrator.process_chat(req)

    assert res.intent == UserIntentEnum.VIEWING_OR_LEAD_REQUEST
    assert res.requires_clarification is True
    assert res.matched_cars is None
    mock_inventory.search.assert_not_called()

def test_orchestrator_viewing_intent_unsupported_constraints(mock_interpreter):
    """Test viewing intent with unsupported ranking triggers explanation without search."""
    mock_inventory = MagicMock()
    mock_interpreter.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
        query_filters=ParsedInventoryQuery(make="Bentley"),
        requires_clarification=False,
        clarification_question=None,
        unsupported_constraints=[
            UnsupportedConstraint(field="ranking", requested_value="cheapest", reason="ranking_not_supported")
        ],
        readiness_state=SearchReadinessState.UNSUPPORTED_CONSTRAINTS_PRESENT
    )

    orchestrator = ChatOrchestrator(query_interpreter=mock_interpreter, inventory_service=mock_inventory)
    req = ChatRequest(user_id="user_1", message="I want to test drive the cheapest Bentley")
    res = orchestrator.process_chat(req)

    assert res.intent == UserIntentEnum.VIEWING_OR_LEAD_REQUEST
    assert res.matched_cars is None
    mock_inventory.search.assert_not_called()
    assert "cheapest" in res.response

# ==============================================================================
# 5. GENERAL CHAT & GUARDRAILS
# ==============================================================================

def test_orchestrator_general_chat_greeting(mock_interpreter, real_inventory):
    """Test friendly greeting response for general chat."""
    mock_interpreter.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.GENERAL_CHAT,
        query_filters=None,
        requires_clarification=False,
        clarification_question=None,
        unsupported_constraints=[],
        readiness_state=SearchReadinessState.NON_INVENTORY_INTENT
    )

    orchestrator = ChatOrchestrator(query_interpreter=mock_interpreter, inventory_service=real_inventory)
    req = ChatRequest(user_id="user_1", message="Hello!")
    res = orchestrator.process_chat(req)

    assert res.intent == UserIntentEnum.GENERAL_CHAT
    assert res.matched_cars is None
    assert "DubizzleBot" in res.response

def test_orchestrator_competitor_query_redirect(mock_interpreter, real_inventory):
    """Test polite competitor query redirection to dubizzle verified inventory."""
    mock_interpreter.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.GENERAL_CHAT,
        query_filters=None,
        requires_clarification=False,
        clarification_question=None,
        unsupported_constraints=[],
        readiness_state=SearchReadinessState.NON_INVENTORY_INTENT
    )

    orchestrator = ChatOrchestrator(query_interpreter=mock_interpreter, inventory_service=real_inventory)
    req = ChatRequest(user_id="user_1", message="Is dubizzle better than YallaMotor?")
    res = orchestrator.process_chat(req)

    assert res.intent == UserIntentEnum.GENERAL_CHAT
    assert "dubizzle verified car inventory" in res.response

def test_orchestrator_non_automotive_refusal(mock_interpreter, real_inventory):
    """Test polite domain guardrail refusal for out-of-scope queries."""
    mock_interpreter.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.UNKNOWN,
        query_filters=None,
        requires_clarification=False,
        clarification_question=None,
        unsupported_constraints=[],
        readiness_state=SearchReadinessState.NON_INVENTORY_INTENT
    )

    orchestrator = ChatOrchestrator(query_interpreter=mock_interpreter, inventory_service=real_inventory)
    req = ChatRequest(user_id="user_1", message="Write Python code to reverse a list")
    res = orchestrator.process_chat(req)

    assert res.intent == UserIntentEnum.UNKNOWN
    assert res.matched_cars is None
    assert "specialized in helping you find cars" in res.response

# ==============================================================================
# 6. SESSION ID HANDLING & DETERMINISTIC ORDERING
# ==============================================================================

def test_orchestrator_session_id_preservation(mock_interpreter, real_inventory):
    """Test that existing session_id is preserved."""
    mock_interpreter.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.GENERAL_CHAT,
        query_filters=None,
        requires_clarification=False,
        clarification_question=None,
        unsupported_constraints=[],
        readiness_state=SearchReadinessState.NON_INVENTORY_INTENT
    )

    orchestrator = ChatOrchestrator(query_interpreter=mock_interpreter, inventory_service=real_inventory)
    req = ChatRequest(user_id="user_1", message="Hi", session_id="custom-session-uuid-123")
    res = orchestrator.process_chat(req)

    assert res.session_id == "custom-session-uuid-123"

def test_orchestrator_session_id_generation_when_omitted(mock_interpreter, real_inventory):
    """Test that a valid session_id is generated when omitted."""
    mock_interpreter.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.GENERAL_CHAT,
        query_filters=None,
        requires_clarification=False,
        clarification_question=None,
        unsupported_constraints=[],
        readiness_state=SearchReadinessState.NON_INVENTORY_INTENT
    )

    orchestrator = ChatOrchestrator(query_interpreter=mock_interpreter, inventory_service=real_inventory)
    req = ChatRequest(user_id="user_1", message="Hi", session_id=None)
    res = orchestrator.process_chat(req)

    assert res.session_id is not None
    assert len(res.session_id) > 10
    uuid.UUID(res.session_id)  # Verifies it's a valid UUID string

def test_orchestrator_deterministic_listing_id_ordering(mock_interpreter, real_inventory):
    """Test that matched_cars strictly preserves Listing_ID ascending order."""
    mock_interpreter.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(make="Bentley"),
        requires_clarification=False,
        clarification_question=None,
        unsupported_constraints=[],
        readiness_state=SearchReadinessState.READY
    )

    orchestrator = ChatOrchestrator(query_interpreter=mock_interpreter, inventory_service=real_inventory)
    req = ChatRequest(user_id="user_1", message="Show Bentleys")
    res = orchestrator.process_chat(req)

    listing_ids = [c.listing_id for c in res.matched_cars]
    assert listing_ids == sorted(listing_ids)

def test_orchestrator_prose_truncation_with_full_matched_cars(mock_interpreter, real_inventory):
    """Test that prose summarizes up to 5 cars while matched_cars contains all exact matches."""
    mock_interpreter.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(make="Ford"),
        requires_clarification=False,
        clarification_question=None,
        unsupported_constraints=[],
        readiness_state=SearchReadinessState.READY
    )

    orchestrator = ChatOrchestrator(query_interpreter=mock_interpreter, inventory_service=real_inventory)
    req = ChatRequest(user_id="user_1", message="Show all Ford cars")
    res = orchestrator.process_chat(req)

    if res.total_matches > 5:
        assert len(res.matched_cars) == res.total_matches
        assert "Here are the first 5 options:" in res.response
        assert res.response.count("• Listing #") == 5

# ==============================================================================
# 7. EXCEPTION SAFETY & GROUNDING INVARIANTS
# ==============================================================================

def test_orchestrator_exception_safety_no_traceback_leak():
    """Test that internal exceptions in interpreter or inventory do not leak tracebacks."""
    failing_interpreter = MagicMock()
    failing_interpreter.interpret.side_effect = RuntimeError("Database connection timed out")

    orchestrator = ChatOrchestrator(query_interpreter=failing_interpreter)
    req = ChatRequest(user_id="user_1", message="Any car")
    res = orchestrator.process_chat(req)

    assert res.intent == UserIntentEnum.UNKNOWN
    assert res.matched_cars is None
    assert "Database connection timed out" not in res.response
    assert "Traceback" not in res.response
    assert "I apologize, but I encountered an issue processing your request" in res.response

def test_grounded_response_builder_missing_attributes_invariant():
    """Test that missing/None attributes are formatted as 'Not stated' without hallucinating numbers."""
    car = CarListing(
        listing_id=101,
        year=2020,
        make="Toyota",
        model="Corolla",
        trim=None,
        title="Toyota Corolla 2020",
        description="Clean car",
        price_aed=None,
        monthly_payment_aed=None,
        mileage_km=None,
        regional_specs=None,
        has_positive_warranty=None,
        warranty_status=None,
        body_type=None
    )
    formatted = GroundedResponseBuilder._format_listing_line(car)
    assert "Price: Not stated" in formatted
    assert "Mileage: Not stated" in formatted
    assert "Specs: Not stated" in formatted
    assert "Monthly:" not in formatted
    assert "Warranty:" not in formatted
    assert "Body:" not in formatted

# ==============================================================================
# 8. PHASE 4A MULTI-TURN SESSION MEMORY & CONTEXTUAL VEHICLE RESOLUTION
# ==============================================================================

def test_orchestrator_multi_turn_land_rover_flow_with_spy(real_inventory):
    """
    Assessment demo flow:
    Turn 1: 'Show me Land Rovers' -> calls interpreter, returns real Land Rovers.
    Turn 2: 'What's the mileage on that first Land Rover?' -> does NOT call interpreter, returns mileage of first Land Rover.
    Turn 3: 'Is there a warranty on it?' -> does NOT call interpreter, returns warranty of SAME Land Rover.
    """
    mock_interp = MagicMock()
    mock_interp.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(make="Land Rover"),
        requires_clarification=False,
        readiness_state=SearchReadinessState.READY
    )

    orchestrator = ChatOrchestrator(query_interpreter=mock_interp, inventory_service=real_inventory)
    session_id = str(uuid.uuid4())

    # Turn 1: Search Land Rovers
    req1 = ChatRequest(user_id="user_1", message="Show me Land Rovers", session_id=session_id)
    res1 = orchestrator.process_chat(req1)
    assert res1.total_matches > 0
    assert len(res1.matched_cars) > 0
    first_car: CarListing = res1.matched_cars[0]
    assert first_car.make.lower() == "land rover"
    assert mock_interp.interpret.call_count == 1

    # Turn 2: Follow-up on first Land Rover's mileage
    req2 = ChatRequest(user_id="user_1", message="What's the mileage on that first Land Rover?", session_id=session_id)
    res2 = orchestrator.process_chat(req2)
    assert mock_interp.interpret.call_count == 1  # Spy confirms QueryInterpreter was NOT called!
    assert res2.total_matches == 1
    assert res2.matched_cars[0].listing_id == first_car.listing_id
    if first_car.mileage_km is not None:
        assert f"{first_car.mileage_km:,} km" in res2.response
    else:
        assert "mileage is not stated" in res2.response

    # Turn 3: Follow-up on warranty using pronoun 'it'
    req3 = ChatRequest(user_id="user_1", message="Is there a warranty on it?", session_id=session_id)
    res3 = orchestrator.process_chat(req3)
    assert mock_interp.interpret.call_count == 1  # Spy confirms QueryInterpreter was STILL not called!
    assert res3.total_matches == 1
    assert res3.matched_cars[0].listing_id == first_car.listing_id
    if first_car.warranty_status:
        assert first_car.warranty_status in res3.response

def test_orchestrator_result_set_replacement_on_new_search(real_inventory):
    """Verify fresh searches replace the current result set and reset the active vehicle."""
    mock_interp = MagicMock()
    orchestrator = ChatOrchestrator(query_interpreter=mock_interp, inventory_service=real_inventory)
    session_id = str(uuid.uuid4())

    # Turn 1: Search Land Rovers
    mock_interp.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(make="Land Rover"),
        requires_clarification=False,
        readiness_state=SearchReadinessState.READY
    )
    res1 = orchestrator.process_chat(ChatRequest(user_id="user_1", message="Show me Land Rovers", session_id=session_id))
    first_lr = res1.matched_cars[0]

    # Turn 2: Search Bentleys
    mock_interp.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(make="Bentley"),
        requires_clarification=False,
        readiness_state=SearchReadinessState.READY
    )
    res2 = orchestrator.process_chat(ChatRequest(user_id="user_1", message="Show me Bentleys", session_id=session_id))
    first_bentley = res2.matched_cars[0]
    assert first_bentley.make.lower() == "bentley"

    # Turn 3: "What's the price of the first one?" -> Should resolve to first Bentley, not Land Rover
    res3 = orchestrator.process_chat(ChatRequest(user_id="user_1", message="What's the price of the first one?", session_id=session_id))
    assert res3.total_matches == 1
    assert res3.matched_cars[0].listing_id == first_bentley.listing_id
    assert res3.matched_cars[0].listing_id != first_lr.listing_id

def test_orchestrator_zero_result_clears_context(real_inventory):
    """Verify zero-match search clears prior results so old results are not referenced."""
    mock_interp = MagicMock()
    orchestrator = ChatOrchestrator(query_interpreter=mock_interp, inventory_service=real_inventory)
    session_id = str(uuid.uuid4())

    # Turn 1: Search Land Rovers (matches found)
    mock_interp.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(make="Land Rover"),
        requires_clarification=False,
        readiness_state=SearchReadinessState.READY
    )
    orchestrator.process_chat(ChatRequest(user_id="user_1", message="Show me Land Rovers", session_id=session_id))
    assert mock_interp.interpret.call_count == 1

    # Turn 2: Search impossible Ferrari (0 matches)
    mock_interp.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(make="Ferrari", min_year=1950, max_year=1960),
        requires_clarification=False,
        readiness_state=SearchReadinessState.READY
    )
    res2 = orchestrator.process_chat(ChatRequest(user_id="user_1", message="Show me 1950 Ferraris", session_id=session_id))
    assert res2.total_matches == 0
    assert mock_interp.interpret.call_count == 2

    # Turn 3: Follow-up -> Result set is empty, so it returns clarification without calling interpreter
    res3 = orchestrator.process_chat(ChatRequest(user_id="user_1", message="What's the mileage on that first Land Rover?", session_id=session_id))
    assert res3.requires_clarification is True
    assert "no Land Rover vehicles in your current search results" in res3.response
    assert mock_interp.interpret.call_count == 2  # Proves interpreter is NOT called for referential query with no context

def test_orchestrator_general_chat_preserves_context(real_inventory):
    """Verify greeting and general chat do not erase current search results."""
    mock_interp = MagicMock()
    orchestrator = ChatOrchestrator(query_interpreter=mock_interp, inventory_service=real_inventory)
    session_id = str(uuid.uuid4())

    # Turn 1: Search Land Rovers
    mock_interp.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(make="Land Rover"),
        requires_clarification=False,
        readiness_state=SearchReadinessState.READY
    )
    res1 = orchestrator.process_chat(ChatRequest(user_id="user_1", message="Show me Land Rovers", session_id=session_id))
    first_lr = res1.matched_cars[0]

    # Turn 2: General chat
    mock_interp.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.GENERAL_CHAT,
        query_filters=None,
        requires_clarification=False,
        readiness_state=SearchReadinessState.NON_INVENTORY_INTENT
    )
    res2 = orchestrator.process_chat(ChatRequest(user_id="user_1", message="Hello there!", session_id=session_id))
    assert res2.intent == UserIntentEnum.GENERAL_CHAT

    # Turn 3: Follow-up on first Land Rover -> Context preserved
    res3 = orchestrator.process_chat(ChatRequest(user_id="user_1", message="What's the mileage on that first Land Rover?", session_id=session_id))
    assert res3.total_matches == 1
    assert res3.matched_cars[0].listing_id == first_lr.listing_id

def test_orchestrator_cross_user_isolation(real_inventory):
    """Verify same session_id across different user_ids does not leak context."""
    mock_interp = MagicMock()
    orchestrator = ChatOrchestrator(query_interpreter=mock_interp, inventory_service=real_inventory)
    shared_session = "shared_sess_uuid"

    # Alice searches Land Rovers
    mock_interp.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(make="Land Rover"),
        requires_clarification=False,
        readiness_state=SearchReadinessState.READY
    )
    orchestrator.process_chat(ChatRequest(user_id="alice", message="Show me Land Rovers", session_id=shared_session))
    assert mock_interp.interpret.call_count == 1

    # Bob asks about first Land Rover in shared_session without having searched
    res_bob = orchestrator.process_chat(ChatRequest(user_id="bob", message="What's the mileage on that first Land Rover?", session_id=shared_session))
    # Proves Bob does NOT get Alice's Land Rover (returns clarification since Bob has 0 Land Rovers)
    assert res_bob.requires_clarification is True
    assert "no Land Rover vehicles in your current search results" in res_bob.response
    assert mock_interp.interpret.call_count == 1

def test_orchestrator_pronoun_with_no_context_returns_clarification(real_inventory):
    """Verify pronoun on fresh session returns clarification and does NOT call QueryInterpreter."""
    mock_interp = MagicMock()
    orchestrator = ChatOrchestrator(query_interpreter=mock_interp, inventory_service=real_inventory)
    session_id = str(uuid.uuid4())

    req = ChatRequest(user_id="user_fresh", message="Does it have a warranty?", session_id=session_id)
    res = orchestrator.process_chat(req)

    assert res.requires_clarification is True
    assert "Which vehicle are you referring to?" in res.response
    assert res.matched_cars is None
    assert res.total_matches == 0
    assert mock_interp.interpret.call_count == 0, "QueryInterpreter should NOT be called for referential pronoun queries"

    req2 = ChatRequest(user_id="user_fresh", message="What's its mileage?", session_id=session_id)
    res2 = orchestrator.process_chat(req2)
    assert res2.requires_clarification is True
    assert mock_interp.interpret.call_count == 0

    req3 = ChatRequest(user_id="user_fresh", message="How much is it?", session_id=session_id)
    res3 = orchestrator.process_chat(req3)
    assert res3.requires_clarification is True
    assert mock_interp.interpret.call_count == 0

def test_orchestrator_bare_deictic_uses_active_vehicle_without_llm_or_search(real_inventory):
    """A resolved bare-deictic follow-up stays entirely on the deterministic path."""
    mock_interp = MagicMock()
    inventory_spy = MagicMock(wraps=real_inventory)
    memory_service = MemoryService()
    session_id = str(uuid.uuid4())
    session = memory_service.get_or_create_session("bare_deictic_user", session_id)
    session.current_result_set = [
        real_inventory.get_by_listing_id(9),
        real_inventory.get_by_listing_id(17),
    ]
    session.active_listing_id = 17
    memory_service.save_session(session)
    orchestrator = ChatOrchestrator(
        query_interpreter=mock_interp,
        inventory_service=inventory_spy,
        memory_service=memory_service,
    )

    result = orchestrator.process_chat(ChatRequest(
        user_id="bare_deictic_user",
        session_id=session_id,
        message="Which year is that?",
    ))

    assert result.total_matches == 1
    assert result.matched_cars[0].listing_id == 17
    assert result.response == "The bentley continental (Listing #17) is a 2020 model."
    mock_interp.interpret.assert_not_called()
    inventory_spy.search.assert_not_called()


# ==============================================================================
# 9. CONTEXTUAL MODEL-YEAR COMPARISON REGRESSION TESTS (Phase 4A Enhancement)
# ==============================================================================

def test_orchestrator_latest_year_comparison_flow_with_multi_turn_ordinal_preservation(real_inventory):
    """
    End-to-end multi-turn verification:
    Turn 1: 'Show me Fords' -> Search returns verified Fords.
    Turn 2: 'Which is the latest year model?' -> Deterministically returns max year listing(s) with zero LLM/inventory calls.
    Turn 3: 'What's the mileage on the second one?' -> Proves the second ordinal STILL refers to original search result #2.
    """
    mock_interp = MagicMock()
    mock_interp.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(make="Ford"),
        requires_clarification=False,
        readiness_state=SearchReadinessState.READY
    )

    inventory_spy = MagicMock(wraps=real_inventory)
    memory_service = MemoryService()
    orchestrator = ChatOrchestrator(
        query_interpreter=mock_interp,
        inventory_service=inventory_spy,
        memory_service=memory_service
    )
    session_id = str(uuid.uuid4())

    # Turn 1: Search Fords
    req1 = ChatRequest(user_id="user_test_flow", message="Show me Fords", session_id=session_id)
    res1 = orchestrator.process_chat(req1)
    assert res1.total_matches > 0
    original_results = res1.matched_cars
    max_year = max(c.year for c in original_results)
    expected_winners = [c for c in original_results if c.year == max_year]
    assert mock_interp.interpret.call_count == 1
    assert inventory_spy.search.call_count == 1

    # Turn 2: Ask for latest year model
    req2 = ChatRequest(user_id="user_test_flow", message="Which is the latest year model?", session_id=session_id)
    res2 = orchestrator.process_chat(req2)

    # Zero additional LLM / inventory calls
    assert mock_interp.interpret.call_count == 1
    assert inventory_spy.search.call_count == 1
    assert res2.total_matches == len(expected_winners)
    assert [c.listing_id for c in res2.matched_cars] == [c.listing_id for c in expected_winners]
    assert f"{max_year}" in res2.response
    assert "The latest model year in your current results is" in res2.response

    # Verify session.current_result_set is completely intact and unmutated
    session = memory_service.get_session("user_test_flow", session_id)
    assert len(session.current_result_set) == len(original_results)
    assert [c.listing_id for c in session.current_result_set] == [c.listing_id for c in original_results]

    # Turn 3: "What's the mileage on the second one?" must resolve against ORIGINAL result set car #2
    if len(original_results) >= 2:
        second_car = original_results[1]
        req3 = ChatRequest(user_id="user_test_flow", message="What's the mileage on the second one?", session_id=session_id)
        res3 = orchestrator.process_chat(req3)
        assert mock_interp.interpret.call_count == 1  # Still no extra LLM call
        assert res3.total_matches == 1
        assert res3.matched_cars[0].listing_id == second_car.listing_id
        if second_car.mileage_km is not None:
            assert f"{second_car.mileage_km:,} km" in res3.response

def test_orchestrator_comparison_tied_winners_order_and_counts(real_inventory):
    """Verify tied winners preserve original order and ChatResponse contains only tied winners."""
    mock_interp = MagicMock()
    inventory_spy = MagicMock(wraps=real_inventory)
    memory_service = MemoryService()
    session_id = str(uuid.uuid4())

    # Pre-populate session with known test listings: 2012, 2014, 2011, 2014
    car1 = CarListing(listing_id=101, make="Ford", model="Explorer", year=2012, title="2012 Ford Explorer", description="Clean")
    car2 = CarListing(listing_id=102, make="Ford", model="Edge", year=2014, title="2014 Ford Edge", description="Clean")
    car3 = CarListing(listing_id=103, make="Ford", model="Focus", year=2011, title="2011 Ford Focus", description="Clean")
    car4 = CarListing(listing_id=104, make="Ford", model="Mustang", year=2014, title="2014 Ford Mustang", description="Clean")

    session = memory_service.get_or_create_session("tie_user", session_id)
    session.current_result_set = [car1, car2, car3, car4]
    memory_service.save_session(session)

    orchestrator = ChatOrchestrator(
        query_interpreter=mock_interp,
        inventory_service=inventory_spy,
        memory_service=memory_service
    )

    req = ChatRequest(user_id="tie_user", message="Which one is the latest?", session_id=session_id)
    res = orchestrator.process_chat(req)

    assert mock_interp.interpret.call_count == 0
    assert inventory_spy.search.call_count == 0
    assert res.total_matches == 2
    assert [c.listing_id for c in res.matched_cars] == [102, 104]  # Preserved original relative order
    assert "The latest model year in your current results is 2014" in res.response
    assert "There are 2 vehicles from 2014" in res.response
    assert "Listing #102" in res.response
    assert "Listing #104" in res.response

def test_orchestrator_oldest_year_comparison_flow(real_inventory):
    """Verify 'Which is the oldest?' correctly finds min(year) listing."""
    mock_interp = MagicMock()
    inventory_spy = MagicMock(wraps=real_inventory)
    memory_service = MemoryService()
    session_id = str(uuid.uuid4())

    car1 = CarListing(listing_id=201, make="Toyota", model="Camry", year=2018, title="2018 Toyota Camry", description="Clean")
    car2 = CarListing(listing_id=202, make="Toyota", model="Corolla", year=2015, title="2015 Toyota Corolla", description="Clean")
    car3 = CarListing(listing_id=203, make="Toyota", model="Yaris", year=2020, title="2020 Toyota Yaris", description="Clean")

    session = memory_service.get_or_create_session("oldest_user", session_id)
    session.current_result_set = [car1, car2, car3]
    memory_service.save_session(session)

    orchestrator = ChatOrchestrator(
        query_interpreter=mock_interp,
        inventory_service=inventory_spy,
        memory_service=memory_service
    )

    req = ChatRequest(user_id="oldest_user", message="Which is the oldest model?", session_id=session_id)
    res = orchestrator.process_chat(req)

    assert mock_interp.interpret.call_count == 0
    assert inventory_spy.search.call_count == 0
    assert res.total_matches == 1
    assert res.matched_cars[0].listing_id == 202
    assert "The oldest model year in your current results is 2015" in res.response
    assert "Listing #202" in res.response

def test_orchestrator_empty_result_set_comparison_clarification(real_inventory):
    """Verify comparison on empty session returns clarification with zero LLM/inventory calls."""
    mock_interp = MagicMock()
    inventory_spy = MagicMock(wraps=real_inventory)
    memory_service = MemoryService()
    orchestrator = ChatOrchestrator(
        query_interpreter=mock_interp,
        inventory_service=inventory_spy,
        memory_service=memory_service
    )
    session_id = str(uuid.uuid4())

    req = ChatRequest(user_id="empty_user", message="Which is the latest year model?", session_id=session_id)
    res = orchestrator.process_chat(req)

    assert mock_interp.interpret.call_count == 0
    assert inventory_spy.search.call_count == 0
    assert res.requires_clarification is True
    assert res.total_matches == 0
    assert res.matched_cars is None
    assert "Search for some cars first" in res.response
    assert "latest model year" in res.response

def test_orchestrator_comparison_preserves_active_listing_id(real_inventory):
    """Verify session.active_listing_id is preserved exactly before and after comparison."""
    mock_interp = MagicMock()
    memory_service = MemoryService()
    session_id = str(uuid.uuid4())

    car1 = CarListing(listing_id=301, make="BMW", model="320i", year=2017, title="2017 BMW 320i", description="Clean")
    car2 = CarListing(listing_id=302, make="BMW", model="530i", year=2021, title="2021 BMW 530i", description="Clean")

    session = memory_service.get_or_create_session("active_id_user", session_id)
    session.current_result_set = [car1, car2]
    session.active_listing_id = 301  # Focused on 301 before comparison
    memory_service.save_session(session)

    orchestrator = ChatOrchestrator(
        query_interpreter=mock_interp,
        inventory_service=real_inventory,
        memory_service=memory_service
    )

    req = ChatRequest(user_id="active_id_user", message="Which one is the latest?", session_id=session_id)
    res = orchestrator.process_chat(req)

    assert res.total_matches == 1
    assert res.matched_cars[0].listing_id == 302
    # Active listing ID must remain 301 (preserved)
    post_session = memory_service.get_session("active_id_user", session_id)
    assert post_session.active_listing_id == 301

def test_orchestrator_fresh_ranking_queries_follow_interpreter_unsupported_path(real_inventory):
    """Verify fresh ranking queries like 'Show me the newest cars' are NOT intercepted as session comparisons."""
    mock_interp = MagicMock()
    mock_interp.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(),
        requires_clarification=False,
        clarification_question=None,
        unsupported_constraints=[UnsupportedConstraint(field="ranking", requested_value="newest", reason="Ranking not supported")],
        readiness_state=SearchReadinessState.UNSUPPORTED_CONSTRAINTS_PRESENT
    )

    memory_service = MemoryService()
    session_id = str(uuid.uuid4())
    # Session has existing cars, but user issues a fresh command
    car1 = CarListing(listing_id=401, make="Ford", model="Edge", year=2015, title="2015 Ford Edge", description="Clean")
    session = memory_service.get_or_create_session("fresh_user", session_id)
    session.current_result_set = [car1]
    memory_service.save_session(session)

    orchestrator = ChatOrchestrator(
        query_interpreter=mock_interp,
        inventory_service=real_inventory,
        memory_service=memory_service
    )

    req = ChatRequest(user_id="fresh_user", message="Show me the newest cars", session_id=session_id)
    res = orchestrator.process_chat(req)

    assert mock_interp.interpret.call_count == 1  # Verified: routed to interpreter, NOT intercepted!
    assert "reliably rank" in res.response or "ranking" in res.response.lower()

