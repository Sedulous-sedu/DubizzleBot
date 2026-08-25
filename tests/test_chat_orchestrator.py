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
