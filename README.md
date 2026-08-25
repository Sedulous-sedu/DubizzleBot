# DubizzleBot - AI Assistant Prototype for dubizzle cars

DubizzleBot is an AI assistant prototype built for **dubizzle cars** that enables users to explore car listings, book test-drive viewing slots, qualify leads, and retain short-term & long-term conversation memory.

---

## Project Structure & Architecture

```text
DubizzleBot/
├── pyproject.toml               # uv project definition and dependency manifest
├── README.md                    # Project documentation & execution guide
├── .env.example                 # Environment variables template
├── Copy_of_sample_cars_dataset.xlsx # Provided car inventory dataset
│
├── backend/                     # FastAPI Backend Application Core
│   ├── main.py                  # FastAPI server entry point
│   ├── config.py                # App configuration loader
│   ├── models/                  # Pydantic schema contracts (chat, car, lead)
│   ├── services/                # Modular domain services
│   │   ├── llm.py               # LiteLLM & Gemini API integration
│   │   ├── inventory.py         # Inventory search engine (pandas/SQL)
│   │   ├── memory.py            # Session state & long-term memory
│   │   └── lead_qualifier.py    # Viewing slot booking & lead qualification recorder
│   └── database/                # SQLite storage lifecycle and models
│
├── frontend/                    # Streamlit Reactive Client Interface
│   ├── app.py                   # Streamlit UI interface
│   └── api_client.py            # Backend API HTTP client layer
│
└── tests/                       # Pytest Automated Test Suite
    ├── conftest.py              # Test fixtures and client configuration
    ├── test_api.py              # Endpoint tests
    ├── test_inventory.py        # Inventory search tests
    └── test_memory.py           # Memory persistence tests
```

---

## Setup & Dependency Management

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reproducible environment and package management.

### 1. Environment Setup

```bash
# Create a virtual environment with uv
uv venv

# Activate virtual environment
source .venv/bin/activate

# Install all project dependencies (including dev dependencies)
uv pip install -e ".[dev]"
```

### 2. Running Backend (FastAPI)

```bash
uv run uvicorn backend.main:app --reload --port 8000
```

### 3. Running Frontend (Streamlit)

```bash
uv run streamlit run frontend/app.py
```

### 4. Running Tests (pytest)

```bash
uv run pytest
```
