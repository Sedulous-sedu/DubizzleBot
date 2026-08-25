"""Unit tests for FastAPI endpoints including inventory retrieval."""

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
