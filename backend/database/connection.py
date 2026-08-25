"""SQLite Database Connection Lifecycle Manager."""

import sqlite3
from backend.config import settings

def get_db_connection() -> sqlite3.Connection:
    """Return a configured SQLite database connection instance."""
    db_path = settings.DATABASE_URL.replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database tables schema."""
    # Schema setup will be implemented in subsequent task
    pass
