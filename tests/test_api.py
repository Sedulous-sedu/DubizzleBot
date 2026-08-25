"""Unit tests for FastAPI endpoints."""

def test_health_check(client):
    """Verify backend health endpoint responds with 200 OK and status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "project" in data
