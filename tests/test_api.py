"""Unit tests for FastAPI endpoints including inventory retrieval and chat orchestration."""

import uuid
import pytest
from unittest.mock import MagicMock
import backend.main as main_module
from backend.models.intent import (
    UserIntentEnum,
    SearchReadinessState,
    ParsedUserIntent,
)

@pytest.fixture(autouse=True)
def mock_api_chat_interpreter(monkeypatch):
    """Ensure all API /chat endpoint tests run 100% offline with zero network calls."""
    mock_interp = MagicMock()
    mock_interp.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.GENERAL_CHAT,
        query_filters=None,
        requires_clarification=False,
        clarification_question=None,
        unsupported_constraints=[],
        readiness_state=SearchReadinessState.NON_INVENTORY_INTENT
    )
    monkeypatch.setattr(main_module.chat_orchestrator, "query_interpreter", mock_interp)
    return mock_interp

def test_health_check(client):
    """Verify backend health endpoint responds with 200 OK and status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"

def test_inventory_summary_endpoint(client):
    """Verify GET /inventory/summary returns dataset statistics."""
    response = client.get("/inventory/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["total_listings"] == 100
    assert data["unique_makes"] == 35

def test_inventory_search_endpoint(client):
    """Verify POST /inventory/search filters listings properly."""
    payload = {"make": "bentley"}
    response = client.post("/inventory/search", json=payload)
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 7
    for car in results:
        assert car["make"] == "bentley"

# ==============================================================================
# POST /chat ENDPOINT TESTS
# ==============================================================================

def test_chat_endpoint_success(client):
    """Verify POST /chat processes request and returns valid ChatResponse schema."""
    payload = {
        "user_id": "test_user",
        "message": "Hello!",
        "session_id": "session-123"
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "test_user"
    assert data["session_id"] == "session-123"
    assert isinstance(data["response"], str)
    assert len(data["response"]) > 0
    assert "total_matches" in data
    assert "requires_clarification" in data

def test_chat_endpoint_session_generation_when_omitted(client):
    """Verify POST /chat generates a new session_id when omitted in request."""
    payload = {
        "user_id": "test_user_2",
        "message": "Hi"
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == "test_user_2"
    assert data["session_id"] is not None
    uuid.UUID(data["session_id"])  # Validates UUID format

def test_chat_endpoint_validation_empty_user_id(client):
    """Verify POST /chat returns 422 for empty or whitespace-only user_id."""
    # Empty string
    res1 = client.post("/chat", json={"user_id": "", "message": "hello"})
    assert res1.status_code == 422

    # Whitespace-only string
    res2 = client.post("/chat", json={"user_id": "   ", "message": "hello"})
    assert res2.status_code == 422

def test_chat_endpoint_validation_empty_message(client):
    """Verify POST /chat returns 422 for empty or whitespace-only message."""
    # Empty string
    res1 = client.post("/chat", json={"user_id": "u1", "message": ""})
    assert res1.status_code == 422

    # Whitespace-only string
    res2 = client.post("/chat", json={"user_id": "u1", "message": "   "})
    assert res2.status_code == 422

def test_chat_endpoint_validation_missing_fields(client):
    """Verify POST /chat returns 422 for missing required fields."""
    response = client.post("/chat", json={})
    assert response.status_code == 422

def test_chat_endpoint_multi_turn_land_rover_assessment_demo(client, mock_api_chat_interpreter):
    """
    Assessment demo flow via POST /chat endpoint:
    Turn 1: 'Show me Land Rovers' -> calls interpreter, returns matching Land Rovers, establishes session_id.
    Turn 2: 'What's the mileage on that first Land Rover?' -> resolves first Land Rover, returns exact mileage.
    Turn 3: 'Is there a warranty on it?' -> resolves same Land Rover, returns exact warranty.
    """
    from backend.models.intent import ParsedInventoryQuery

    # Configure mock interpreter for Turn 1
    mock_api_chat_interpreter.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(make="Land Rover"),
        requires_clarification=False,
        readiness_state=SearchReadinessState.READY
    )

    # Turn 1: Search Land Rovers
    res1 = client.post("/chat", json={"user_id": "demo_user", "message": "Show me Land Rovers"})
    assert res1.status_code == 200
    data1 = res1.json()
    session_id = data1["session_id"]
    assert data1["total_matches"] > 0
    first_car = data1["matched_cars"][0]
    assert mock_api_chat_interpreter.interpret.call_count == 1

    # Turn 2: Mileage on first Land Rover
    res2 = client.post("/chat", json={
        "user_id": "demo_user",
        "message": "What's the mileage on that first Land Rover?",
        "session_id": session_id
    })
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["session_id"] == session_id
    assert data2["total_matches"] == 1
    assert data2["matched_cars"][0]["listing_id"] == first_car["listing_id"]
    assert mock_api_chat_interpreter.interpret.call_count == 1  # No new call to QueryInterpreter!
    if first_car["mileage_km"] is not None:
        assert f"{first_car['mileage_km']:,} km" in data2["response"]

    # Turn 3: Warranty on it
    res3 = client.post("/chat", json={
        "user_id": "demo_user",
        "message": "Is there a warranty on it?",
        "session_id": session_id
    })
    assert res3.status_code == 200
    data3 = res3.json()
    assert data3["session_id"] == session_id
    assert data3["total_matches"] == 1
    assert data3["matched_cars"][0]["listing_id"] == first_car["listing_id"]
    assert mock_api_chat_interpreter.interpret.call_count == 1  # Still no call to QueryInterpreter!
    if first_car["warranty_status"]:
        assert first_car["warranty_status"] in data3["response"]

def test_chat_endpoint_multi_session_persistent_memory_flow(client, mock_api_chat_interpreter):
    """
    Assessment demo flow via POST /chat across DIFFERENT sessions:
    Session 1: 'Show me Bentleys' -> 'I like the second one' (saves Listing #17).
    Session 2 (same user_id, new session_id): 'What cars did I like?' -> returns Listing #17 without LLM call!
    """
    from backend.models.intent import ParsedInventoryQuery

    mock_api_chat_interpreter.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(make="Bentley"),
        requires_clarification=False,
        readiness_state=SearchReadinessState.READY
    )

    user_id = "returning_api_user"
    session_1 = "session_api_1"

    # Turn 1: Search Bentleys
    res1 = client.post("/chat", json={"user_id": user_id, "message": "Show me Bentleys", "session_id": session_1})
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["total_matches"] == 7
    second_bentley = data1["matched_cars"][1]
    assert mock_api_chat_interpreter.interpret.call_count == 1

    # Turn 2: Like second car
    res2 = client.post("/chat", json={"user_id": user_id, "message": "I like the second one", "session_id": session_1})
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["total_matches"] == 1
    assert data2["matched_cars"][0]["listing_id"] == second_bentley["listing_id"]
    assert f"Listing #{second_bentley['listing_id']}" in data2["response"]

    # Session 2: New session ID!
    session_2 = "session_api_2"
    res3 = client.post("/chat", json={"user_id": user_id, "message": "What cars did I like?", "session_id": session_2})
    assert res3.status_code == 200
    data3 = res3.json()
    assert data3["session_id"] == session_2
    assert data3["total_matches"] == 1
    assert data3["matched_cars"][0]["listing_id"] == second_bentley["listing_id"]
    assert mock_api_chat_interpreter.interpret.call_count == 1  # 0 LLM calls for recall!

    # Turn 4: Save explicit budget preference
    res4 = client.post("/chat", json={"user_id": user_id, "message": "My budget is now AED 150,000", "session_id": session_2})
    assert res4.status_code == 200
    data4 = res4.json()
    assert "AED 150,000" in data4["response"]

    # Turn 5: Ask what is remembered
    res5 = client.post("/chat", json={"user_id": user_id, "message": "What do you remember about me?", "session_id": session_2})
    assert res5.status_code == 200
    data5 = res5.json()
    assert "AED 150,000" in data5["response"]
    assert "Bentley" in data5["response"]
    assert "1 vehicle in your favorites" in data5["response"]

def test_chat_endpoint_booking_multi_turn_flow(client, mock_api_chat_interpreter):
    """Verify multi-turn test-drive booking flow through POST /chat."""
    from backend.models.intent import ParsedInventoryQuery

    mock_api_chat_interpreter.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        query_filters=ParsedInventoryQuery(make="Bentley"),
        requires_clarification=False,
        readiness_state=SearchReadinessState.READY
    )
    user_id = "user_api_booking"

    # Turn 1: Search Bentleys
    res1 = client.post("/chat", json={"user_id": user_id, "message": "Show me Bentleys"})
    assert res1.status_code == 200
    data1 = res1.json()
    session_id = data1["session_id"]
    second_car = data1["matched_cars"][1]

    # Turn 2: Test drive the second one
    res2 = client.post("/chat", json={"user_id": user_id, "session_id": session_id, "message": "I want to test drive the second one."})
    assert res2.status_code == 200
    assert "what date and time" in res2.json()["response"].lower()

    # Turn 3: Saturday at 3 PM
    res3 = client.post("/chat", json={"user_id": user_id, "session_id": session_id, "message": "Saturday at 3 PM"})
    assert res3.status_code == 200
    assert "would you like me to confirm this test drive?" in res3.json()["response"].lower()

    # Turn 4: Explicit confirmation
    res4 = client.post("/chat", json={"user_id": user_id, "session_id": session_id, "message": "Yes confirm"})
    assert res4.status_code == 200
    assert "has been confirmed" in res4.json()["response"].lower()
    assert "booking ref:" in res4.json()["response"].lower()

def test_chat_endpoint_lead_qualification_flow(client):
    """Verify lead capture workflow through POST /chat."""
    user_id = "user_api_lead"

    # Turn 1: Start lead inquiry
    res1 = client.post("/chat", json={
        "user_id": user_id,
        "message": "I'd like to submit an enquiry for a GCC car under AED 150,000."
    })
    assert res1.status_code == 200
    session_id = res1.json()["session_id"]
    assert "phone number or email" in res1.json()["response"].lower()

    # Turn 2: Provide contact
    res2 = client.post("/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "My phone is +971501234567, name is John"
    })
    assert res2.status_code == 200
    assert "summary of your enquiry" in res2.json()["response"].lower()

    # Turn 3: Confirm submission
    res3 = client.post("/chat", json={
        "user_id": user_id,
        "session_id": session_id,
        "message": "Yes please submit"
    })
    assert res3.status_code == 200
    assert "has been submitted to our sales team" in res3.json()["response"].lower()




