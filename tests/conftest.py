"""Pytest configuration and test fixtures."""

import pytest
from fastapi.testclient import TestClient
from backend.database.connection import init_db
from backend.main import create_app, app

@pytest.fixture(autouse=True)
def isolate_test_storage(tmp_path, monkeypatch):
    """Ensure every test runs with isolated temporary SQLite database and leads CSV file."""
    test_db_path = str(tmp_path / "test_dubizzle.db")
    test_csv_path = str(tmp_path / "test_leads.csv")
    monkeypatch.setattr("backend.config.settings.DATABASE_URL", f"sqlite:///{test_db_path}")
    monkeypatch.setattr("backend.config.settings.LEADS_CSV_PATH", test_csv_path)
    init_db(test_db_path)
    return {"db_path": test_db_path, "csv_path": test_csv_path}

@pytest.fixture
def client(tmp_path):
    """Fixture providing FastAPI TestClient instance wired to app."""
    return TestClient(app)
