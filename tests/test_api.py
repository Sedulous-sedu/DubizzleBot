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
