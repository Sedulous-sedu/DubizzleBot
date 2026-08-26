"""Frontend configuration for DubizzleBot Streamlit client."""

import os

# Backend API server URL
BACKEND_URL: str = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
