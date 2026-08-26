"""Comprehensive unit tests for MemoryService and ContextResolver."""

import pytest
from datetime import timezone
from backend.models.car import CarListing
from backend.models.intent import UserIntentEnum
from backend.models.memory import (
    SessionState,
    ConversationTurn,
    ResolutionStatus,
    TargetAttribute,
)
from backend.services.memory import MemoryService
from backend.services.context_resolver import ContextResolver

@pytest.fixture
def sample_cars():
    """Provides a list of sample CarListing objects for testing."""
    return [
        CarListing(
            listing_id=101,
            make="Honda",
            model="Civic",
            year=2021,
            title="2021 Honda Civic EX",
            description="Excellent condition GCC specs",
            price_aed=65000.0,
            mileage_km=45000,
            monthly_payment_aed=1200.0,
            regional_specs="GCC",
            warranty_status="Under Warranty",
            has_positive_warranty=True,
            body_type="Sedan"
        ),
        CarListing(
            listing_id=102,
            make="Toyota",
            model="Camry",
            year=2020,
            title="2020 Toyota Camry LE",
            description="Well maintained single owner",
            price_aed=75000.0,
            mileage_km=60000,
            monthly_payment_aed=1400.0,
            regional_specs="GCC",
            warranty_status="Warranty Option Available (Not Active)",
            has_positive_warranty=False,
            body_type="Sedan"
        ),
        CarListing(
            listing_id=103,
            make="Honda",
            model="Accord",
            year=2019,
            title="2019 Honda Accord Sport",
            description="GCC specs clean history",
            price_aed=58000.0,
            mileage_km=82000,
            monthly_payment_aed=1100.0,
            regional_specs="GCC",
            warranty_status=None,
            has_positive_warranty=None,
            body_type="Sedan"
        ),
    ]

# =============================================================================
# MemoryService Tests
# =============================================================================

def test_memory_service_initialization():
    """Verify memory service instance creation."""
    service = MemoryService()
    assert service is not None
    assert service._max_sessions == 1000
    assert service._max_turns_per_session == 50

def test_get_or_create_session():
    """Verify session creation with timezone-aware timestamps and default fields."""
    service = MemoryService()
    session = service.get_or_create_session("user_1", "sess_1")
    assert session.session_id == "sess_1"
    assert session.user_id == "user_1"
    assert session.created_at.tzinfo is not None
    assert session.created_at.tzinfo == timezone.utc
    assert session.turns == []
    assert session.current_result_set == []
    assert session.active_listing_id is None

def test_session_isolation_by_user_id(sample_cars):
    """Verify same session_id under different user_ids creates completely isolated states."""
    service = MemoryService()
    session_a = service.get_or_create_session("alice", "shared_session_id")
    session_b = service.get_or_create_session("bob", "shared_session_id")

    # Record turn for Alice
    service.record_turn(
        user_id="alice",
        session_id="shared_session_id",
        user_message="Show me Hondas",
        assistant_response="Found Hondas",
        intent=UserIntentEnum.INVENTORY_SEARCH,
        matched_cars=sample_cars,
        replace_result_set=True,
        active_listing_id=101
    )

    alice_state = service.get_session("alice", "shared_session_id")
    bob_state = service.get_session("bob", "shared_session_id")

    assert len(alice_state.turns) == 1
    assert len(alice_state.current_result_set) == 3
    assert alice_state.active_listing_id == 101

    assert len(bob_state.turns) == 0
    assert len(bob_state.current_result_set) == 0
    assert bob_state.active_listing_id is None

def test_record_turn_and_bounding(sample_cars):
    """Verify turn recording appends turns and trims when exceeding max_turns_per_session."""
    service = MemoryService(max_turns_per_session=3)

    for i in range(5):
        service.record_turn(
            user_id="user_test",
            session_id="sess_bounds",
            user_message=f"Message {i}",
            assistant_response=f"Response {i}",
            intent=UserIntentEnum.INVENTORY_SEARCH,
            matched_cars=sample_cars if i == 0 else None,
            replace_result_set=(i == 0)
        )

    session = service.get_session("user_test", "sess_bounds")
    assert len(session.turns) == 3
    assert session.turns[0].user_message == "Message 2"
    assert session.turns[2].user_message == "Message 4"
    assert len(session.current_result_set) == 3  # Result set preserved

def test_lru_session_eviction():
    """Verify oldest session is evicted when max_sessions capacity is reached."""
    service = MemoryService(max_sessions=2)
    service.get_or_create_session("u1", "s1")
    service.get_or_create_session("u2", "s2")
    service.get_or_create_session("u3", "s3")  # Evicts s1

    assert service.get_session("u1", "s1") is None
    assert service.get_session("u2", "s2") is not None
    assert service.get_session("u3", "s3") is not None

def test_clear_session():
    """Verify clear_session removes session from memory."""
    service = MemoryService()
    service.get_or_create_session("u1", "s1")
    assert service.get_session("u1", "s1") is not None
    assert service.clear_session("u1", "s1") is True
    assert service.get_session("u1", "s1") is None
    assert service.clear_session("u1", "s1") is False

# =============================================================================
# ContextResolver Tests
# =============================================================================

def test_context_resolver_fresh_searches_not_intercepted(sample_cars):
    """Verify fresh attribute searches remain NOT_CONTEXTUAL."""
    session = SessionState(session_id="s", user_id="u", current_result_set=sample_cars)

    fresh_queries = [
        "Show me GCC cars",
        "Find cars under warranty",
        "Show me low mileage Bentleys",
        "Find cars with monthly payments under AED 2,000",
        "Cars under 100k",
        "Show me Hondas from 2018 to 2022",
    ]

    for q in fresh_queries:
        res = ContextResolver.resolve(q, session)
        assert res.status == ResolutionStatus.NOT_CONTEXTUAL, f"Query '{q}' should be NOT_CONTEXTUAL"

def test_context_resolver_unqualified_ordinals(sample_cars):
    """Verify 'first one', 'second car', 'last one' resolve against current_result_set."""
    session = SessionState(session_id="s", user_id="u", current_result_set=sample_cars)

    # 1. First one
    res1 = ContextResolver.resolve("What's the mileage on the first one?", session)
    assert res1.status == ResolutionStatus.RESOLVED
    assert res1.resolved_car.listing_id == 101
    assert res1.target_attribute == TargetAttribute.MILEAGE

    # 2. Second car
    res2 = ContextResolver.resolve("Does the second car have a warranty?", session)
    assert res2.status == ResolutionStatus.RESOLVED
    assert res2.resolved_car.listing_id == 102
    assert res2.target_attribute == TargetAttribute.WARRANTY

    # 3. Last one
    res3 = ContextResolver.resolve("What is the price of the last one?", session)
    assert res3.status == ResolutionStatus.RESOLVED
    assert res3.resolved_car.listing_id == 103
    assert res3.target_attribute == TargetAttribute.PRICE

def test_context_resolver_qualified_ordinals_filters_before_indexing(sample_cars):
    """Verify 'first Honda' and 'second Honda' filter by make before applying index."""
    # sample_cars has: [101: Honda Civic, 102: Toyota Camry, 103: Honda Accord]
    session = SessionState(session_id="s", user_id="u", current_result_set=sample_cars)

    # "first Honda" -> Listing 101 (Honda Civic)
    res1 = ContextResolver.resolve("What's the mileage on that first Honda?", session)
    assert res1.status == ResolutionStatus.RESOLVED
    assert res1.resolved_car.listing_id == 101
    assert res1.resolved_car.model == "Civic"

    # "second Honda" -> Listing 103 (Honda Accord)
    res2 = ContextResolver.resolve("How much is the second Honda?", session)
    assert res2.status == ResolutionStatus.RESOLVED
    assert res2.resolved_car.listing_id == 103
    assert res2.resolved_car.model == "Accord"

    # "third Honda" -> Only 2 Hondas exist -> Clarification required
    res3 = ContextResolver.resolve("What's the price on the third Honda?", session)
    assert res3.status == ResolutionStatus.CLARIFICATION_REQUIRED
    assert "only 2 Honda" in res3.clarification_message

def test_context_resolver_qualified_ordinal_missing_make(sample_cars):
    """Verify qualified ordinal referencing a make not in current results asks for clarification."""
    session = SessionState(session_id="s", user_id="u", current_result_set=sample_cars)

    res = ContextResolver.resolve("What's the mileage on that first Bentley?", session)
    assert res.status == ResolutionStatus.CLARIFICATION_REQUIRED
    assert "no Bentley vehicles" in res.clarification_message

def test_context_resolver_unqualified_make_single_match(sample_cars):
    """Verify make reference with a single match in results resolves to that car."""
    session = SessionState(session_id="s", user_id="u", current_result_set=sample_cars)

    # Exactly 1 Toyota in sample_cars (Listing 102)
    res = ContextResolver.resolve("Is there a warranty on the Toyota?", session)
    assert res.status == ResolutionStatus.RESOLVED
    assert res.resolved_car.listing_id == 102
    assert res.target_attribute == TargetAttribute.WARRANTY

def test_context_resolver_unqualified_make_multiple_matches_clarification(sample_cars):
    """Verify make reference with multiple matches in results returns clarification."""
    session = SessionState(session_id="s", user_id="u", current_result_set=sample_cars)

    # 2 Hondas in sample_cars
    res = ContextResolver.resolve("What's the price for the Honda?", session)
    assert res.status == ResolutionStatus.CLARIFICATION_REQUIRED
    assert "2 Honda vehicles" in res.clarification_message

def test_context_resolver_single_visible_car_pronoun_fallback(sample_cars):
    """Verify 'it' resolves to sole visible car when active_listing_id is None."""
    single_car = [sample_cars[0]]  # Listing 101
    session = SessionState(session_id="s", user_id="u", current_result_set=single_car, active_listing_id=None)

    res = ContextResolver.resolve("Does it have a warranty?", session)
    assert res.status == ResolutionStatus.RESOLVED
    assert res.resolved_car.listing_id == 101
    assert res.target_attribute == TargetAttribute.WARRANTY

def test_context_resolver_multiple_visible_cars_pronoun_ambiguity(sample_cars):
    """Verify 'it' returns clarification when multiple cars are visible and active_listing_id is None."""
    session = SessionState(session_id="s", user_id="u", current_result_set=sample_cars, active_listing_id=None)

    res = ContextResolver.resolve("Does it have a warranty?", session)
    assert res.status == ResolutionStatus.CLARIFICATION_REQUIRED
    assert "Which vehicle from your search results" in res.clarification_message

def test_context_resolver_active_vehicle_pronoun(sample_cars):
    """Verify 'it' resolves to active_listing_id when established."""
    session = SessionState(session_id="s", user_id="u", current_result_set=sample_cars, active_listing_id=102)

    res = ContextResolver.resolve("What's its monthly payment?", session)
    assert res.status == ResolutionStatus.RESOLVED
    assert res.resolved_car.listing_id == 102
    assert res.target_attribute == TargetAttribute.MONTHLY_PAYMENT

def test_context_resolver_pronoun_with_no_context_clarification():
    """Verify referential pronouns on an empty/fresh session return clarification without hallucinating."""
    session = SessionState(session_id="s_empty", user_id="u_empty", current_result_set=[], active_listing_id=None)

    no_context_queries = [
        ("Does it have a warranty?", TargetAttribute.WARRANTY),
        ("What's its mileage?", TargetAttribute.MILEAGE),
        ("How much is it?", TargetAttribute.PRICE),
        ("How much is that car?", TargetAttribute.PRICE),
        ("Is this one GCC?", TargetAttribute.REGIONAL_SPECS),
        ("Tell me more about that vehicle", TargetAttribute.ALL_DETAILS),
    ]

    for query, expected_attr in no_context_queries:
        res = ContextResolver.resolve(query, session)
        assert res.status == ResolutionStatus.CLARIFICATION_REQUIRED, f"Query '{query}' should require clarification"
        assert res.resolved_car is None
        assert "Which vehicle are you referring to?" in res.clarification_message
