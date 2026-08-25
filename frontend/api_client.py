"""HTTP Client Helper for communication between Streamlit app and FastAPI backend."""

import httpx
from typing import Dict, Any

class BackendAPIClient:
    """Client for calling DubizzleBot FastAPI backend endpoints."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def check_health(self) -> Dict[str, Any]:
        """Call backend health check endpoint."""
        with httpx.Client(base_url=self.base_url, timeout=5.0) as client:
            response = client.get("/health")
            response.raise_for_status()
            return response.json()
