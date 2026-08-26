"""Pytest configuration and test fixtures."""

import pytest
from fastapi.testclient import TestClient
from backend.database.connection import init_db
from backend.main import create_app, app

@pytest.fixture(autouse=True)
def isolate_test_db(tmp_path, monkeypatch):
    """Ensure every test runs with an isolated temporary SQLite database."""
    test_db_path = str(tmp_path / "test_dubizzle.db")
    monkeypatch.setattr("backend.config.settings.DATABASE_URL", f"sqlite:///{test_db_path}")
    init_db(test_db_path)
    return test_db_path

@pytest.fixture
def client(tmp_path):
    """Fixture providing FastAPI TestClient instance wired to app."""
    return TestClient(app)
