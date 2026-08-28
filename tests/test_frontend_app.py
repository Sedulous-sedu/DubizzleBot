"""Smoke AppTest for DubizzleBot Streamlit application."""

import pytest
from streamlit.testing.v1 import AppTest

def test_streamlit_app_loads_and_renders_empty_state():
    """Verify Streamlit app initializes without errors and renders title and welcome hero."""
    at = AppTest.from_file("../frontend/app.py", default_timeout=10.0)
    at.run()
    assert not at.exception
    # Title appears in main view
    assert any("DubizzleBot" in title.value for title in at.title)
    # Buttons for starter prompts exist
    assert len(at.button) > 0
    # Sidebar components exist
    assert any("User Identity" in inp.label for inp in at.sidebar.text_input)

def test_streamlit_app_renders_offline_warning_when_backend_down(monkeypatch):
    """Verify Streamlit app displays backend offline warning when FastAPI is unreachable."""
    from frontend.api_client import DubizzleAPIClient
    monkeypatch.setattr(DubizzleAPIClient, "health_check", lambda self: False)
    at = AppTest.from_file("../frontend/app.py", default_timeout=10.0)
    at.run()
    assert not at.exception
    assert len(at.sidebar.error) > 0 or len(at.warning) > 0
