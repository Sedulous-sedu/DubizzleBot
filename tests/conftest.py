"""Pytest configuration and test fixtures."""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

@pytest.fixture
def client():
    """Fixture providing FastAPI TestClient instance."""
    return TestClient(app)
