"""SQLite Database Connection Lifecycle Manager."""

import os
import sqlite3
from typing import Optional
from backend.config import settings

def resolve_db_path(db_url_or_path: Optional[str] = None) -> str:
    """Extract filesystem path from a sqlite:/// URL or return path directly."""
    target = db_url_or_path or settings.DATABASE_URL
    if target.startswith("sqlite:///"):
        return target.replace("sqlite:///", "", 1)
    return target

def get_db_connection(db_path: Optional[str] = None, timeout: float = 10.0) -> sqlite3.Connection:
    """Return a configured SQLite database connection with foreign keys and busy timeout enabled."""
    resolved_path = resolve_db_path(db_path)
    
    # Ensure parent directory exists if path is not in-memory or relative current dir
    dir_name = os.path.dirname(resolved_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        
    conn = sqlite3.connect(resolved_path, timeout=timeout)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn

def init_db(db_path: Optional[str] = None):
    """Initialize database tables schema idempotently."""
    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL
            );
            """)

            conn.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY REFERENCES user_profiles(user_id) ON DELETE CASCADE,
                preferred_make TEXT,
                preferred_model TEXT,
                min_year INTEGER,
                max_year INTEGER,
                min_price_aed REAL,
                max_price_aed REAL,
                min_mileage_km INTEGER,
                max_mileage_km INTEGER,
                min_monthly_payment REAL,
                max_monthly_payment REAL,
                regional_specs TEXT,
                warranty_preference INTEGER, -- 1=True, 0=False, NULL=unstated
                keywords TEXT,
                last_search_filters TEXT,    -- JSON-serialized ParsedInventoryQuery
                updated_at TEXT NOT NULL
            );
            """)

            conn.execute("""
            CREATE TABLE IF NOT EXISTS liked_cars (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL REFERENCES user_profiles(user_id) ON DELETE CASCADE,
                listing_id INTEGER NOT NULL,
                liked_at TEXT NOT NULL,
                UNIQUE(user_id, listing_id)
            );
            """)

            conn.execute("""
            CREATE TABLE IF NOT EXISTS bookings (
                booking_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES user_profiles(user_id) ON DELETE CASCADE,
                listing_id INTEGER NOT NULL,
                appointment_at TEXT NOT NULL,
                customer_name TEXT,
                customer_phone TEXT,
                customer_email TEXT,
                status TEXT NOT NULL DEFAULT 'confirmed',
                created_at TEXT NOT NULL
            );
            """)

            conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_liked_cars_user_id ON liked_cars(user_id);
            """)

            conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_bookings_user_id ON bookings(user_id);
            """)

            conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_bookings_appointment ON bookings(appointment_at);
            """)

            conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_profiles_last_seen ON user_profiles(last_seen_at);
            """)
    finally:
        conn.close()
