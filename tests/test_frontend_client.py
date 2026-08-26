"""Offline unit tests for DubizzleBot Streamlit frontend API client."""

import json
import pytest
import httpx

from frontend.api_client import (
    DubizzleAPIClient,
    DubizzleAPIError,
    FrontendChatResponse,
    FrontendCarListing,
)

def test_api_client_health_check_success():
    """Verify health_check returns True when backend returns 200 OK with status='ok'."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "ok", "project": "DubizzleBot", "version": "0.1.0"})

    transport = httpx.MockTransport(handler)
    client = DubizzleAPIClient(base_url="http://test-server:8000", transport=transport)
    assert client.health_check() is True

def test_api_client_health_check_connection_error():
    """Verify health_check returns False when connection fails."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    transport = httpx.MockTransport(handler)
    client = DubizzleAPIClient(base_url="http://test-server:8000", transport=transport)
    assert client.health_check() is False

def test_api_client_health_check_timeout():
    """Verify health_check returns False on timeout."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Request timed out")

    transport = httpx.MockTransport(handler)
    client = DubizzleAPIClient(base_url="http://test-server:8000", transport=transport)
    assert client.health_check() is False

def test_api_client_send_chat_success_exact_mapping():
    """Verify send_chat parses full ChatResponse payload and preserves order and fields."""
    request_payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/chat"
        req_data = json.loads(request.content)
        request_payloads.append(req_data)
        return httpx.Response(200, json={
            "user_id": "user_demo",
            "session_id": "sess_123",
            "response": "Found 2 matching cars.",
            "matched_cars": [
                {
                    "listing_id": 9,
                    "year": 2022,
                    "make": "Bentley",
                    "model": "Bentayga",
                    "trim": "Speed",
                    "title": "2022 Bentley Bentayga Speed",
                    "description": "Mint condition",
                    "photo_url": "https://images.example.com/car9.jpg",
                    "price_aed": 750000.0,
                    "monthly_payment_aed": 12500.0,
                    "mileage_km": 15000.0,
                    "regional_specs": "GCC",
                    "has_positive_warranty": True,
                    "warranty_status": "Under dealership warranty until 2027",
                    "body_type": "SUV",
                    "provenance": None,
                },
                {
                    "listing_id": 17,
                    "year": 2020,
                    "make": "Bentley",
                    "model": "Continental",
                    "trim": "GT",
                    "title": "2020 Bentley Continental GT",
                    "description": None,
                    "photo_url": None,
                    "price_aed": None,
                    "monthly_payment_aed": None,
                    "mileage_km": 318.0,
                    "regional_specs": None,
                    "has_positive_warranty": False,
                    "warranty_status": None,
                    "body_type": None,
                    "provenance": None,
                }
            ],
            "intent": "inventory_search",
            "total_matches": 2,
            "requires_clarification": False,
        })

    transport = httpx.MockTransport(handler)
    client = DubizzleAPIClient(base_url="http://test-server:8000", transport=transport)

    resp = client.send_chat(user_id="user_demo", session_id="sess_123", message="Show me Bentleys")
    assert resp.user_id == "user_demo"
    assert resp.session_id == "sess_123"
    assert resp.total_matches == 2
    assert len(resp.matched_cars) == 2

    # Preserves order 9 then 17
    assert resp.matched_cars[0].listing_id == 9
    assert resp.matched_cars[0].warranty_status == "Under dealership warranty until 2027"
    assert resp.matched_cars[1].listing_id == 17
    assert resp.matched_cars[1].price_aed is None
    assert resp.matched_cars[1].warranty_status is None

    # Sent correct payload
    assert len(request_payloads) == 1
    assert request_payloads[0]["user_id"] == "user_demo"
    assert request_payloads[0]["session_id"] == "sess_123"
    assert request_payloads[0]["message"] == "Show me Bentleys"

def test_api_client_send_chat_matched_cars_none():
    """Verify send_chat handles matched_cars=None for general/clarification messages."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "user_id": "user_demo",
            "session_id": "sess_123",
            "response": "Hello! How can I help you?",
            "matched_cars": None,
            "intent": "general_chat",
            "total_matches": 0,
            "requires_clarification": False,
        })

    transport = httpx.MockTransport(handler)
    client = DubizzleAPIClient(base_url="http://test-server:8000", transport=transport)
    resp = client.send_chat(user_id="user_demo", session_id="sess_123", message="Hi")
    assert resp.matched_cars is None
    assert resp.intent == "general_chat"

def test_api_client_send_chat_http_500_error():
    """Verify backend 500 error raises user-safe DubizzleAPIError."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "Internal Server Error"})

    transport = httpx.MockTransport(handler)
    client = DubizzleAPIClient(base_url="http://test-server:8000", transport=transport)

    with pytest.raises(DubizzleAPIError) as exc_info:
        client.send_chat(user_id="u1", session_id="s1", message="test")
    assert "error (500)" in exc_info.value.message.lower()

def test_api_client_send_chat_connection_error():
    """Verify backend connection failure raises user-safe DubizzleAPIError."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    transport = httpx.MockTransport(handler)
    client = DubizzleAPIClient(base_url="http://test-server:8000", transport=transport)

    with pytest.raises(DubizzleAPIError) as exc_info:
        client.send_chat(user_id="u1", session_id="s1", message="test")
    assert "could not connect" in exc_info.value.message.lower()

def test_api_client_send_chat_timeout_error():
    """Verify read timeout raises user-safe DubizzleAPIError without retrying."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        raise httpx.ReadTimeout("Timeout")

    transport = httpx.MockTransport(handler)
    client = DubizzleAPIClient(base_url="http://test-server:8000", transport=transport)

    with pytest.raises(DubizzleAPIError) as exc_info:
        client.send_chat(user_id="u1", session_id="s1", message="test")
    assert "timed out" in exc_info.value.message.lower() or "in time" in exc_info.value.message.lower()
    # Confirm NO automatic retries on stateful chat requests
    assert call_count == 1

def test_api_client_send_chat_malformed_json():
    """Verify non-JSON response raises DubizzleAPIError."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"Bad Gateway HTML")

    transport = httpx.MockTransport(handler)
    client = DubizzleAPIClient(base_url="http://test-server:8000", transport=transport)

    with pytest.raises(DubizzleAPIError) as exc_info:
        client.send_chat(user_id="u1", session_id="s1", message="test")
    assert "malformed" in exc_info.value.message.lower()

def test_api_client_send_chat_schema_invalid():
    """Verify missing required fields raises DubizzleAPIError."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"user_id": "u1"})  # Missing session_id, response, intent

    transport = httpx.MockTransport(handler)
    client = DubizzleAPIClient(base_url="http://test-server:8000", transport=transport)

    with pytest.raises(DubizzleAPIError) as exc_info:
        client.send_chat(user_id="u1", session_id="s1", message="test")
    assert "schema mismatch" in exc_info.value.message.lower()

def test_api_client_send_chat_identity_mismatch():
    """Verify backend returning mismatched user/session identity is rejected safely."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "user_id": "DIFFERENT_USER",
            "session_id": "DIFFERENT_SESSION",
            "response": "Hello",
            "matched_cars": None,
            "intent": "general_chat",
            "total_matches": 0,
            "requires_clarification": False,
        })

    transport = httpx.MockTransport(handler)
    client = DubizzleAPIClient(base_url="http://test-server:8000", transport=transport)

    with pytest.raises(DubizzleAPIError) as exc_info:
        client.send_chat(user_id="user_expected", session_id="sess_expected", message="test")
    assert "inconsistent" in exc_info.value.message.lower()

def test_api_client_send_chat_empty_validation():
    """Verify client rejects empty user_id, session_id, or message."""
    client = DubizzleAPIClient(base_url="http://test-server:8000")
    with pytest.raises(DubizzleAPIError):
        client.send_chat(user_id="", session_id="s1", message="hi")
    with pytest.raises(DubizzleAPIError):
        client.send_chat(user_id="u1", session_id="  ", message="hi")
    with pytest.raises(DubizzleAPIError):
        client.send_chat(user_id="u1", session_id="s1", message="")
