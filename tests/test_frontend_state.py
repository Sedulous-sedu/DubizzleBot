"""Offline unit tests for DubizzleBot Streamlit frontend state transitions."""

import uuid
import pytest

from frontend.api_client import FrontendChatResponse, FrontendCarListing
from frontend.state import (
    ensure_initial_state,
    start_new_conversation,
    switch_user,
    queue_prompt,
    consume_queued_prompt,
    add_user_message,
    add_assistant_message,
    DEFAULT_USER_ID,
)

def test_state_initial_state_has_valid_defaults():
    """Verify ensure_initial_state sets default user, generated UUID session, and empty messages."""
    state = {}
    ensure_initial_state(state)
    assert state["active_user_id"] == DEFAULT_USER_ID
    assert state["user_id_input"] == DEFAULT_USER_ID
    assert isinstance(state["session_id"], str)
    assert len(state["session_id"]) > 10
    assert state["messages"] == []
    assert state["queued_prompt"] is None

def test_state_start_new_conversation_preserves_active_user():
    """Verify New Conversation preserves active user identity but generates new session UUID."""
    state = {
        "active_user_id": "buyer_99",
        "user_id_input": "buyer_99",
        "session_id": "sess_old_111",
        "messages": [{"role": "user", "content": "Show me Bentleys"}],
        "queued_prompt": "some prompt",
    }
    new_sess = start_new_conversation(state)
    # Active user must be preserved!
    assert state["active_user_id"] == "buyer_99"
    assert state["user_id_input"] == "buyer_99"
    # Session ID must change!
    assert state["session_id"] != "sess_old_111"
    assert state["session_id"] == new_sess
    # Visible messages must be cleared!
    assert state["messages"] == []
    assert state["queued_prompt"] is None

def test_state_switch_user_updates_identity_and_resets_session():
    """Verify switching user ID resets conversational session and clears history."""
    state = {
        "active_user_id": "user_a",
        "user_id_input": "user_a",
        "session_id": "sess_user_a",
        "messages": [{"role": "user", "content": "hello from a"}],
    }
    switched = switch_user(state, "user_b")
    assert switched is True
    assert state["active_user_id"] == "user_b"
    assert state["user_id_input"] == "user_b"
    assert state["session_id"] != "sess_user_a"
    assert state["messages"] == []

def test_state_switch_user_same_user_no_op():
    """Verify switching to the same user does not destroy active conversation."""
    state = {
        "active_user_id": "user_a",
        "user_id_input": "user_a",
        "session_id": "sess_user_a",
        "messages": [{"role": "user", "content": "hello"}],
    }
    switched = switch_user(state, "user_a")
    assert switched is False
    assert state["session_id"] == "sess_user_a"
    assert len(state["messages"]) == 1

def test_state_switch_user_empty_or_whitespace_rejected():
    """Verify empty or whitespace user ID input is rejected without changing state."""
    state = {
        "active_user_id": "user_a",
        "session_id": "sess_user_a",
        "messages": [{"role": "user", "content": "hello"}],
    }
    assert switch_user(state, "") is False
    assert switch_user(state, "   ") is False
    assert state["active_user_id"] == "user_a"
    assert len(state["messages"]) == 1

def test_state_queue_and_consume_prompt_one_shot():
    """Verify queued starter prompts are consumed exactly once."""
    state = {}
    ensure_initial_state(state)

    queue_prompt(state, "Show me Bentleys")
    assert state["queued_prompt"] == "Show me Bentleys"

    # First consumption succeeds
    consumed1 = consume_queued_prompt(state)
    assert consumed1 == "Show me Bentleys"
    assert state["queued_prompt"] is None

    # Second consumption returns None (preventing duplicate rerun submission)
    consumed2 = consume_queued_prompt(state)
    assert consumed2 is None

def test_state_add_user_message_structure():
    """Verify add_user_message appends well-formed user turn."""
    state = {}
    msg = add_user_message(state, "Show me Land Rovers")
    assert msg["role"] == "user"
    assert msg["content"] == "Show me Land Rovers"
    assert len(state["messages"]) == 1
    assert state["messages"][0] == msg

def test_state_add_assistant_message_preserves_all_matched_cars_order():
    """Verify add_assistant_message stores all received matched_cars in original order."""
    state = {}
    car1 = FrontendCarListing(
        listing_id=9, year=2022, make="Bentley", model="Bentayga",
        title="2022 Bentley Bentayga", regional_specs="GCC", warranty_status="Yes"
    )
    car2 = FrontendCarListing(
        listing_id=17, year=2020, make="Bentley", model="Continental",
        title="2020 Bentley Continental", regional_specs="GCC", warranty_status=None
    )
    car3 = FrontendCarListing(
        listing_id=24, year=2018, make="Bentley", model="Flying Spur",
        title="2018 Bentley Flying Spur", regional_specs="American", warranty_status=None
    )
    resp = FrontendChatResponse(
        user_id="u1",
        session_id="s1",
        response="Found 3 Bentleys.",
        matched_cars=[car1, car2, car3],
        intent="inventory_search",
        total_matches=3,
        requires_clarification=False,
    )
    msg = add_assistant_message(state, resp)
    assert msg["role"] == "assistant"
    assert msg["total_matches"] == 3
    assert len(msg["matched_cars"]) == 3
    # Order preserved 9, 17, 24
    assert msg["matched_cars"][0]["listing_id"] == 9
    assert msg["matched_cars"][1]["listing_id"] == 17
    assert msg["matched_cars"][2]["listing_id"] == 24
