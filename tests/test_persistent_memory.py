"""Comprehensive unit and integration tests for Phase 4B Long-Term Returning-User Memory."""

import sqlite3
import uuid
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from backend.database.connection import init_db, get_db_connection, resolve_db_path
from backend.models.car import CarListing, CarFilter
from backend.models.chat import ChatRequest
from backend.models.intent import (
    UserIntentEnum,
    SearchReadinessState,
    ParsedUserIntent,
    ParsedInventoryQuery,
    RegionalSpecEnum,
)
from backend.models.memory import SessionState
from backend.models.persistent_memory import (
    UserProfile,
    UserPreferences,
    PreferencePatch,
    LongTermMemoryAction,
    LongTermMemoryResolution,
)
from backend.services.inventory import InventoryService
from backend.services.memory import MemoryService
from backend.services.persistent_memory import PersistentMemoryService
from backend.services.long_term_resolver import LongTermMemoryResolver
from backend.services.orchestrator import ChatOrchestrator
from backend.services.response_builder import GroundedResponseBuilder

# ==============================================================================
# 1. DATABASE & SCHEMA INTEGRITY TESTS
# ==============================================================================

def test_init_db_idempotent(tmp_path):
    """Verify init_db can be called repeatedly on the same SQLite file without errors."""
    db_file = str(tmp_path / "idempotent.db")
    init_db(db_file)
    init_db(db_file)  # Second call should not fail

    conn = get_db_connection(db_file)
    try:
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = [t["name"] for t in tables]
        assert "user_profiles" in table_names
        assert "user_preferences" in table_names
        assert "liked_cars" in table_names
    finally:
        conn.close()

def test_pragma_foreign_keys_enforced(tmp_path):
    """Verify PRAGMA foreign_keys = ON is enforced and prevents dangling liked_cars rows."""
    db_file = str(tmp_path / "fk_test.db")
    init_db(db_file)
    conn = get_db_connection(db_file)
    try:
        # Attempt to insert liked_car for nonexistent user_id
        with pytest.raises(sqlite3.IntegrityError):
            with conn:
                conn.execute(
                    "INSERT INTO liked_cars (user_id, listing_id, liked_at) VALUES (?, ?, ?)",
                    ("nonexistent_user", 9, datetime.now(timezone.utc).isoformat())
                )
    finally:
        conn.close()

# ==============================================================================
# 2. USER PROFILE & TENANT ISOLATION TESTS
# ==============================================================================

def test_create_and_get_user_profile(tmp_path):
    """Verify user profile creation with UTC timestamps."""
    pmem = PersistentMemoryService(db_path=str(tmp_path / "profile.db"))
    profile = pmem.get_or_create_profile("user_1")

    assert profile.user_id == "user_1"
    assert isinstance(profile.created_at, datetime)
    assert isinstance(profile.last_seen_at, datetime)

def test_returning_user_profile_preserves_created_at(tmp_path):
    """Verify returning user updates last_seen_at while preserving original created_at."""
    pmem = PersistentMemoryService(db_path=str(tmp_path / "profile.db"))
    p1 = pmem.get_or_create_profile("user_1")
    initial_created = p1.created_at

    p2 = pmem.get_or_create_profile("user_1")
    assert p2.created_at == initial_created
    assert p2.last_seen_at >= p1.last_seen_at

def test_user_tenant_isolation(tmp_path):
    """Verify complete tenant isolation between different users."""
    pmem = PersistentMemoryService(db_path=str(tmp_path / "isolation.db"))
    pmem.save_preferences("alice", PreferencePatch(preferred_make="Bentley", max_price_aed=150000.0))
    pmem.save_liked_car("alice", 9)

    bob_prefs = pmem.get_preferences("bob")
    assert bob_prefs is None

    bob_likes = pmem.get_liked_listing_ids("bob")
    assert len(bob_likes) == 0

# ==============================================================================
# 3. PATCH-BASED PREFERENCE STORAGE & UPDATE TESTS
# ==============================================================================

def test_save_preferences_patch_preserves_unrelated_fields(tmp_path):
    """
    Verify preference updates are patch-based:
    Turn 1: Make=Bentley, Specs=GCC, Budget=100k
    Turn 2: Budget=120k -> Make and Specs MUST survive!
    """
    pmem = PersistentMemoryService(db_path=str(tmp_path / "prefs.db"))

    # Turn 1
    p1 = PreferencePatch(
        preferred_make="Bentley",
        regional_specs="GCC",
        max_price_aed=100000.0
    )
    prefs1 = pmem.save_preferences("user_patch", p1)
    assert prefs1.preferred_make == "Bentley"
    assert prefs1.regional_specs == "GCC"
    assert prefs1.max_price_aed == 100000.0

    # Turn 2: Budget update only
    p2 = PreferencePatch(max_price_aed=120000.0)
    prefs2 = pmem.save_preferences("user_patch", p2)
    assert prefs2.preferred_make == "Bentley", "Make must be preserved"
    assert prefs2.regional_specs == "GCC", "Regional specs must be preserved"
    assert prefs2.max_price_aed == 120000.0, "Budget must be updated"

def test_explicit_preference_clearing(tmp_path):
    """Verify explicit field clears only clear the specified field and preserve others."""
    pmem = PersistentMemoryService(db_path=str(tmp_path / "prefs_clear.db"))

    # Initial preferences with warranty preference
    pmem.save_preferences("user_clear", PreferencePatch(
        preferred_make="Toyota",
        warranty_preference=True,
        max_price_aed=80000.0
    ))

    # User says "I don't care about warranty anymore" -> clear warranty_preference
    clear_patch = PreferencePatch(clear_fields={"warranty_preference"})
    updated = pmem.save_preferences("user_clear", clear_patch)

    assert updated.warranty_preference is None, "Warranty preference must be cleared"
    assert updated.preferred_make == "Toyota", "Make must be preserved"
    assert updated.max_price_aed == 80000.0, "Budget must be preserved"

def test_last_search_filters_persistence_and_separation(tmp_path):
    """Verify last_search_filters is stored as separate history and does NOT alter explicit preferences."""
    pmem = PersistentMemoryService(db_path=str(tmp_path / "last_search.db"))
    pmem.save_preferences("user_search", PreferencePatch(preferred_make="Honda"))

    search_query = ParsedInventoryQuery(make="Land Rover", min_year=2018, max_price_aed=150000.0)
    pmem.update_last_search("user_search", search_query)

    prefs = pmem.get_preferences("user_search")
    assert prefs.preferred_make == "Honda", "Explicit preference must remain Honda"
    assert prefs.last_search_filters is not None
    assert prefs.last_search_filters.make == "Land Rover"
    assert prefs.last_search_filters.min_year == 2018

def test_corrupted_last_search_filters_handled_safely(tmp_path):
    """Verify malformed JSON in last_search_filters does not crash preference retrieval."""
    db_file = str(tmp_path / "corrupted.db")
    pmem = PersistentMemoryService(db_path=db_file)
    pmem.get_or_create_profile("user_corrupt")

    # Manually inject invalid JSON into SQLite
    conn = get_db_connection(db_file)
    try:
        with conn:
            conn.execute(
                "INSERT INTO user_preferences (user_id, last_search_filters, updated_at) VALUES (?, ?, ?)",
                ("user_corrupt", "{invalid: json:: 123", datetime.now(timezone.utc).isoformat())
            )
    finally:
        conn.close()

    prefs = pmem.get_preferences("user_corrupt")
    assert prefs is not None
    assert prefs.last_search_filters is None, "Corrupted JSON must safely fallback to None"

# ==============================================================================
# 4. LIKED CARS STORAGE, IDEMPOTENCY & REHYDRATION
# ==============================================================================

def test_save_liked_car_idempotent(tmp_path):
    """Verify saving the same vehicle multiple times is idempotent."""
    pmem = PersistentMemoryService(db_path=str(tmp_path / "likes.db"))
    pmem.save_liked_car("user_like", 9)
    pmem.save_liked_car("user_like", 9)  # Duplicate save

    liked_ids = pmem.get_liked_listing_ids("user_like")
    assert liked_ids == [9]

def test_save_multiple_liked_cars_ordering(tmp_path):
    """Verify liked cars maintain insertion order."""
    pmem = PersistentMemoryService(db_path=str(tmp_path / "likes.db"))
    pmem.save_liked_car("user_multi", 9)
    pmem.save_liked_car("user_multi", 17)
    pmem.save_liked_car("user_multi", 24)

    liked_ids = pmem.get_liked_listing_ids("user_multi")
    assert liked_ids == [9, 17, 24]

def test_remove_liked_car(tmp_path):
    """Verify removing a liked car."""
    pmem = PersistentMemoryService(db_path=str(tmp_path / "likes.db"))
    pmem.save_liked_car("user_rem", 9)
    pmem.save_liked_car("user_rem", 17)

    assert pmem.remove_liked_car("user_rem", 9) is True
    assert pmem.get_liked_listing_ids("user_rem") == [17]

def test_inventory_service_get_by_listing_id():
    """Verify direct lookup of CarListing by Listing_ID from inventory dataset."""
    inv = InventoryService()
    car = inv.get_by_listing_id(9)
    assert car is not None
    assert car.listing_id == 9
    assert car.make.lower() == "bentley"

    invalid_car = inv.get_by_listing_id(999999)
    assert invalid_car is None

# ==============================================================================
# 5. LONG-TERM MEMORY RESOLVER TESTS
# ==============================================================================

@pytest.fixture
def real_inventory():
    return InventoryService()

@pytest.fixture
def sample_bentleys():
    return [
        CarListing(listing_id=9, year=2022, make="Bentley", model="Bentayga", price_aed=850000.0, title="2022 Bentley Bentayga", description="Luxury SUV"),
        CarListing(listing_id=17, year=2020, make="Bentley", model="Continental", price_aed=620000.0, title="2020 Bentley Continental", description="Luxury Coupe"),
        CarListing(listing_id=24, year=2013, make="Bentley", model="Flying Spur", price_aed=145000.0, title="2013 Bentley Flying Spur", description="Luxury Sedan"),
    ]

def test_long_term_resolver_save_liked_car_ordinal(sample_bentleys):
    """Verify 'I like the second one' resolves to second car and returns SAVE_LIKED_CAR."""
    session = SessionState(session_id="s", user_id="u", current_result_set=sample_bentleys)
    res = LongTermMemoryResolver.evaluate("I like the second one", session)

    assert res.action == LongTermMemoryAction.SAVE_LIKED_CAR
    assert res.target_car is not None
    assert res.target_car.listing_id == 17

def test_long_term_resolver_save_liked_car_qualified(sample_bentleys):
    """Verify 'Save the first Bentley' resolves to first Bentley."""
    session = SessionState(session_id="s", user_id="u", current_result_set=sample_bentleys)
    res = LongTermMemoryResolver.evaluate("Save the first Bentley", session)

    assert res.action == LongTermMemoryAction.SAVE_LIKED_CAR
    assert res.target_car.listing_id == 9

def test_long_term_resolver_save_liked_car_unresolvable_clarification():
    """Verify 'I like the second one' on an empty session returns clarification."""
    session = SessionState(session_id="s_empty", user_id="u", current_result_set=[])
    res = LongTermMemoryResolver.evaluate("I like the second one", session)

    assert res.action == LongTermMemoryAction.SAVE_LIKED_CAR
    assert res.target_car is None
    assert res.clarification_message is not None

def test_long_term_resolver_search_query_not_mistaken_for_save():
    """Verify 'I like Bentley cars, show me some' is treated as search, NOT saving a car."""
    session = SessionState(session_id="s", user_id="u", current_result_set=[])
    res = LongTermMemoryResolver.evaluate("I like Bentley cars, show me some", session)

    assert res.action == LongTermMemoryAction.NOT_MEMORY_ACTION

def test_long_term_resolver_recall_liked_cars():
    """Verify recall queries return RECALL_LIKED_CARS."""
    session = SessionState(session_id="s", user_id="u")
    queries = [
        "What cars did I like?",
        "Show my saved cars",
        "What did I save?",
        "Do I have any favorites?",
        "List my liked cars"
    ]
    for q in queries:
        res = LongTermMemoryResolver.evaluate(q, session)
        assert res.action == LongTermMemoryAction.RECALL_LIKED_CARS, f"Query '{q}' failed"

def test_long_term_resolver_recall_memory_transparency():
    """Verify memory transparency queries return RECALL_MEMORY."""
    session = SessionState(session_id="s", user_id="u")
    queries = [
        "What do you remember about me?",
        "What are my saved preferences?",
        "Show my saved profile",
        "What do you know about me?"
    ]
    for q in queries:
        res = LongTermMemoryResolver.evaluate(q, session)
        assert res.action == LongTermMemoryAction.RECALL_MEMORY, f"Query '{q}' failed"

def test_long_term_resolver_save_explicit_preferences():
    """Verify explicit preference extraction from natural language."""
    session = SessionState(session_id="s", user_id="u")

    # GCC specs
    res_gcc = LongTermMemoryResolver.evaluate("I prefer GCC cars", session)
    assert res_gcc.action == LongTermMemoryAction.SAVE_PREFERENCE
    assert res_gcc.preference_patch.regional_specs == RegionalSpecEnum.GCC.value

    # Budget
    res_budget = LongTermMemoryResolver.evaluate("My budget is under AED 120,000", session)
    assert res_budget.action == LongTermMemoryAction.SAVE_PREFERENCE
    assert res_budget.preference_patch.max_price_aed == 120000.0

    # Don't care about warranty
    res_warr_clear = LongTermMemoryResolver.evaluate("I don't care about warranty anymore", session)
    assert res_warr_clear.action == LongTermMemoryAction.SAVE_PREFERENCE
    assert "warranty_preference" in res_warr_clear.preference_patch.clear_fields

def test_long_term_resolver_memory_assisted_search():
    """Verify 'Show me cars matching my saved preferences' returns SEARCH_SAVED_PREFERENCES."""
    session = SessionState(session_id="s", user_id="u")
    res = LongTermMemoryResolver.evaluate("Show me cars matching my saved preferences", session)
    assert res.action == LongTermMemoryAction.SEARCH_SAVED_PREFERENCES

# ==============================================================================
# 6. ORCHESTRATOR CROSS-SESSION & RESTARTS INTEGRATION TESTS
# ==============================================================================

def test_orchestrator_cross_session_liked_car_flow(real_inventory, tmp_path):
    """
    Full cross-session flow:
    Session 1: Search Bentleys -> 'I like the second one' (Listing #17 saved).
    Session 2 (same user_id, different session_id): 'What cars did I like?' -> returns Listing #17!
    """
    db_file = str(tmp_path / "cross_session.db")
    pmem = PersistentMemoryService(db_path=db_file)
    mock_interp = MagicMock()
    orchestrator = ChatOrchestrator(
        query_interpreter=mock_interp,
        inventory_service=real_inventory,
        persistent_memory=pmem
    )

    user_id = "user_cross_demo"
    session_1 = "session_alpha"

    # Turn 1: Search Bentleys
    mock_interp.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(make="Bentley"),
        requires_clarification=False,
        readiness_state=SearchReadinessState.READY
    )
    res1 = orchestrator.process_chat(ChatRequest(user_id=user_id, message="Show me Bentleys", session_id=session_1))
    assert res1.total_matches > 1
    second_car = res1.matched_cars[1]

    # Turn 2: Like second car
    res2 = orchestrator.process_chat(ChatRequest(user_id=user_id, message="I like the second one", session_id=session_1))
    assert res2.total_matches == 1
    assert res2.matched_cars[0].listing_id == second_car.listing_id
    assert f"Listing #{second_car.listing_id}" in res2.response

    # Session 2: New session ID for same user!
    session_2 = "session_beta"
    res3 = orchestrator.process_chat(ChatRequest(user_id=user_id, message="What cars did I like?", session_id=session_2))
    assert res3.total_matches == 1
    assert res3.matched_cars[0].listing_id == second_car.listing_id
    assert mock_interp.interpret.call_count == 1, "Recall of saved cars must require ZERO LLM calls"

def test_orchestrator_simulated_service_restart_preserves_memory(real_inventory, tmp_path):
    """Verify that a completely new orchestrator instance connected to the same SQLite DB retains memory."""
    db_file = str(tmp_path / "restart.db")
    mock_interp = MagicMock()

    # Instance 1 saves preference and liked car
    pmem1 = PersistentMemoryService(db_path=db_file)
    orch1 = ChatOrchestrator(query_interpreter=mock_interp, inventory_service=real_inventory, persistent_memory=pmem1)
    orch1.process_chat(ChatRequest(user_id="user_restart", message="My budget is under AED 150,000", session_id="s1"))
    pmem1.save_liked_car("user_restart", 9)

    # Instance 2 (simulating process restart with fresh in-memory state)
    pmem2 = PersistentMemoryService(db_path=db_file)
    orch2 = ChatOrchestrator(query_interpreter=mock_interp, inventory_service=real_inventory, persistent_memory=pmem2)

    res_mem = orch2.process_chat(ChatRequest(user_id="user_restart", message="What do you remember about me?", session_id="s2"))
    assert "AED 150,000" in res_mem.response
    assert "1 vehicle in your favorites" in res_mem.response

    res_likes = orch2.process_chat(ChatRequest(user_id="user_restart", message="What cars did I like?", session_id="s3"))
    assert res_likes.total_matches == 1
    assert res_likes.matched_cars[0].listing_id == 9

def test_orchestrator_stale_liked_listing_reported_safely(real_inventory, tmp_path):
    """Verify that saved listing IDs no longer in current inventory are reported gracefully without crashing."""
    db_file = str(tmp_path / "stale.db")
    pmem = PersistentMemoryService(db_path=db_file)
    pmem.save_liked_car("user_stale", 9)       # Valid listing in dataset
    pmem.save_liked_car("user_stale", 999999)  # Stale listing NOT in dataset

    mock_interp = MagicMock()
    orch = ChatOrchestrator(query_interpreter=mock_interp, inventory_service=real_inventory, persistent_memory=pmem)

    res = orch.process_chat(ChatRequest(user_id="user_stale", message="What cars did I like?", session_id="sess_stale"))
    assert res.total_matches == 1
    assert res.matched_cars[0].listing_id == 9
    assert "no longer available in our active inventory" in res.response
    assert "#999999" in res.response

def test_memory_transparency_distinguishes_preference_vs_last_search():
    """Verify 'What do you remember about me?' clearly distinguishes explicit preferences from last search."""
    prefs = UserPreferences(
        user_id="u_demo",
        preferred_make="Toyota",
        max_price_aed=80000.0,
        last_search_filters=ParsedInventoryQuery(make="Land Rover", min_year=2018)
    )
    summary = GroundedResponseBuilder.format_preferences_summary_response(prefs, liked_count=3)

    assert "Saved Preferences:" in summary
    assert "Make: Toyota" in summary
    assert "Most Recent Search:" in summary
    assert "Land Rover (from 2018)" in summary
    assert "3 vehicles in your favorites" in summary
    assert "You prefer Land Rover" not in summary, "Must NOT confuse last search with preferred make"

def test_clear_preferences_preserves_last_search_filters(tmp_path):
    """
    Verify 'Forget my saved preferences' clears explicit preferences but preserves last_search_filters:
    Before: preferred_make=Bentley, regional_specs=GCC, max_price=100k, last_search={"make": "Land Rover"}
    After: explicit fields None, last_search remains Land Rover.
    """
    db_file = str(tmp_path / "clear_prefs_preserve_search.db")
    pmem = PersistentMemoryService(db_path=db_file)
    user_id = "user_clear_test"

    # Save explicit preferences
    pmem.save_preferences(user_id, PreferencePatch(
        preferred_make="Bentley",
        regional_specs="GCC",
        max_price_aed=100000.0
    ))
    # Record last search
    pmem.update_last_search(user_id, ParsedInventoryQuery(make="Land Rover", min_year=2018))

    # User says "Forget my saved preferences"
    pmem.clear_preferences(user_id)

    prefs = pmem.get_preferences(user_id)
    assert prefs is not None
    assert prefs.preferred_make is None
    assert prefs.regional_specs is None
    assert prefs.max_price_aed is None
    assert prefs.has_explicit_preferences() is False
    assert prefs.last_search_filters is not None
    assert prefs.last_search_filters.make == "Land Rover"

    # Summary response check
    summary = GroundedResponseBuilder.format_preferences_summary_response(prefs, liked_count=0)
    assert "Saved Preferences:" not in summary
    assert "Most Recent Search:" in summary
    assert "Land Rover" in summary

def test_current_turn_make_override_in_saved_preferences_search(real_inventory, tmp_path):
    """
    Verify 'Show me Land Rovers matching my saved preferences' overrides conflicting saved make (Toyota)
    with the current-turn make (Land Rover) without creating contradictory filter combinations.
    Stored preference in SQLite must remain Toyota.
    """
    db_file = str(tmp_path / "override.db")
    pmem = PersistentMemoryService(db_path=db_file)
    mock_interp = MagicMock()
    orch = ChatOrchestrator(query_interpreter=mock_interp, inventory_service=real_inventory, persistent_memory=pmem)
    user_id = "user_override"

    # Step 1: Explicitly persist conflicting saved preferences: make=Toyota, specs=GCC, max_price=150,000
    pmem.save_preferences(user_id, PreferencePatch(
        preferred_make="Toyota",
        regional_specs="GCC",
        max_price_aed=150000.0
    ))
    prefs_before = pmem.get_preferences(user_id)
    assert prefs_before.preferred_make == "Toyota"
    assert prefs_before.regional_specs == "GCC"
    assert prefs_before.max_price_aed == 150000.0

    # Step 2: User asks for Land Rovers matching saved preferences
    res = orch.process_chat(ChatRequest(
        user_id=user_id,
        message="Show me Land Rovers matching my saved preferences",
        session_id="sess_override"
    ))

    # Step 3: Verify result grounding
    assert res.total_matches > 0
    for car in res.matched_cars:
        assert car.make.lower() == "land rover"
        assert car.regional_specs.upper() == "GCC"
        if car.price_aed is not None:
            assert car.price_aed <= 150000.0

    # Verified exact listing in inventory
    matched_ids = [c.listing_id for c in res.matched_cars]
    assert 3 in matched_ids  # Listing #3 is 2018 Land Rover Velar, GCC, AED 119,750

    # Step 4: Verify persistent preferred_make remains Toyota in SQLite (no permanent mutation)
    prefs_after = pmem.get_preferences(user_id)
    assert prefs_after.preferred_make == "Toyota", "Saved preferred_make must remain Toyota!"
    assert prefs_after.regional_specs == "GCC"
    assert prefs_after.max_price_aed == 150000.0
    assert mock_interp.interpret.call_count == 0, "Deterministic override must not require LLM call"

def test_forget_everything_deletes_all_user_data_transactionally(tmp_path):
    """Verify 'Forget everything about me' transactionally removes profile, preferences, and likes."""
    db_file = str(tmp_path / "forget_all.db")
    pmem = PersistentMemoryService(db_path=db_file)
    user_id = "user_forget_all"

    # Set up user data
    pmem.get_or_create_profile(user_id)
    pmem.save_preferences(user_id, PreferencePatch(preferred_make="Nissan", max_price_aed=50000.0))
    pmem.save_liked_car(user_id, 9)

    # Delete all data
    assert pmem.delete_user_data(user_id) is True

    # Confirm all tables empty for this user
    assert pmem.get_preferences(user_id) is None
    assert pmem.get_liked_listing_ids(user_id) == []

def test_sqlite_connection_closure_guaranteed(tmp_path, monkeypatch):
    """Verify every PersistentMemoryService method closes its SQLite connection."""
    db_file = str(tmp_path / "closure_test.db")
    pmem = PersistentMemoryService(db_path=db_file)

    opened_connections = []
    closed_connections = []

    import backend.services.persistent_memory as pm_module
    orig_get_conn = pm_module.get_db_connection

    class ConnectionProxy:
        def __init__(self, raw_conn):
            self._raw = raw_conn
        def close(self):
            closed_connections.append(self)
            return self._raw.close()
        def __enter__(self):
            return self._raw.__enter__()
        def __exit__(self, *args):
            return self._raw.__exit__(*args)
        def __getattr__(self, name):
            return getattr(self._raw, name)

    def spy_get_conn(path=None, timeout=10.0):
        raw_c = orig_get_conn(path or db_file, timeout=timeout)
        proxy = ConnectionProxy(raw_c)
        opened_connections.append(proxy)
        return proxy

    monkeypatch.setattr(pm_module, "get_db_connection", spy_get_conn)

    # Execute service operations
    pmem.get_or_create_profile("user_c")
    pmem.save_preferences("user_c", PreferencePatch(preferred_make="Ford"))
    pmem.get_preferences("user_c")
    pmem.update_last_search("user_c", ParsedInventoryQuery(make="Ford"))
    pmem.save_liked_car("user_c", 9)
    pmem.get_liked_listing_ids("user_c")
    pmem.remove_liked_car("user_c", 9)
    pmem.clear_preferences("user_c")
    pmem.delete_user_data("user_c")

    assert len(opened_connections) > 0
    assert len(opened_connections) == len(closed_connections), (
        f"Connection leak detected! Opened: {len(opened_connections)}, Closed: {len(closed_connections)}"
    )

