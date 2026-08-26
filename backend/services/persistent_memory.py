"""Persistent SQLite User Memory Service for returning-user profiles, preferences, and liked cars."""

import json
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from backend.database.connection import get_db_connection, init_db, resolve_db_path
from backend.models.persistent_memory import (
    UserProfile,
    UserPreferences,
    PreferencePatch,
    LikedCarRecord,
)
from backend.models.intent import ParsedInventoryQuery

logger = logging.getLogger(__name__)

class PersistentMemoryService:
    """Service managing persistent user profiles, search preferences, and liked cars in SQLite."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = resolve_db_path(db_path)
        init_db(self.db_path)

    def get_or_create_profile(self, user_id: str) -> UserProfile:
        """Retrieves an existing UserProfile or initializes a new one. Updates last_seen_at."""
        clean_user_id = str(user_id).strip()
        now_utc = datetime.now(timezone.utc).isoformat()

        conn = get_db_connection(self.db_path)
        try:
            with conn:
                row = conn.execute(
                    "SELECT user_id, created_at, updated_at, last_seen_at FROM user_profiles WHERE user_id = ?",
                    (clean_user_id,)
                ).fetchone()

                if row:
                    # Update last_seen_at and updated_at while preserving created_at
                    conn.execute(
                        "UPDATE user_profiles SET last_seen_at = ?, updated_at = ? WHERE user_id = ?",
                        (now_utc, now_utc, clean_user_id)
                    )
                    return UserProfile(
                        user_id=row["user_id"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                        updated_at=datetime.fromisoformat(now_utc),
                        last_seen_at=datetime.fromisoformat(now_utc)
                    )
                else:
                    # Insert new user profile
                    conn.execute(
                        "INSERT INTO user_profiles (user_id, created_at, updated_at, last_seen_at) VALUES (?, ?, ?, ?)",
                        (clean_user_id, now_utc, now_utc, now_utc)
                    )
                    return UserProfile(
                        user_id=clean_user_id,
                        created_at=datetime.fromisoformat(now_utc),
                        updated_at=datetime.fromisoformat(now_utc),
                        last_seen_at=datetime.fromisoformat(now_utc)
                    )
        finally:
            conn.close()

    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """Retrieves an existing UserProfile without creating one."""
        clean_user_id = str(user_id).strip()
        conn = get_db_connection(self.db_path)
        try:
            with conn:
                row = conn.execute(
                    "SELECT user_id, created_at, updated_at, last_seen_at FROM user_profiles WHERE user_id = ?",
                    (clean_user_id,)
                ).fetchone()
                if not row:
                    return None
                return UserProfile(
                    user_id=row["user_id"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                    last_seen_at=datetime.fromisoformat(row["last_seen_at"]),
                )
        finally:
            conn.close()

    def record_activity(self, user_id: str) -> None:
        """Updates last_seen_at for a given user_id."""
        self.get_or_create_profile(user_id)

    def get_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """Loads stored preferences for a given user_id, with safe JSON parsing for last_search_filters."""
        clean_user_id = str(user_id).strip()
        conn = get_db_connection(self.db_path)
        try:
            with conn:
                row = conn.execute(
                    """
                    SELECT user_id, preferred_make, preferred_model, min_year, max_year,
                           min_price_aed, max_price_aed, min_mileage_km, max_mileage_km,
                           min_monthly_payment, max_monthly_payment, regional_specs,
                           warranty_preference, keywords, last_search_filters, updated_at
                    FROM user_preferences
                    WHERE user_id = ?
                    """,
                    (clean_user_id,)
                ).fetchone()

                if not row:
                    return None

                # Safely parse last_search_filters JSON without failing if corrupt
                parsed_last_search: Optional[ParsedInventoryQuery] = None
                raw_json = row["last_search_filters"]
                if raw_json:
                    try:
                        parsed_last_search = ParsedInventoryQuery.model_validate_json(raw_json)
                    except Exception as e:
                        logger.warning(f"Corrupted last_search_filters for user {clean_user_id}: {e}")
                        parsed_last_search = None

                warranty_val: Optional[bool] = None
                if row["warranty_preference"] is not None:
                    warranty_val = bool(row["warranty_preference"])

                return UserPreferences(
                    user_id=row["user_id"],
                    preferred_make=row["preferred_make"],
                    preferred_model=row["preferred_model"],
                    min_year=row["min_year"],
                    max_year=row["max_year"],
                    min_price_aed=row["min_price_aed"],
                    max_price_aed=row["max_price_aed"],
                    min_mileage_km=row["min_mileage_km"],
                    max_mileage_km=row["max_mileage_km"],
                    min_monthly_payment=row["min_monthly_payment"],
                    max_monthly_payment=row["max_monthly_payment"],
                    regional_specs=row["regional_specs"],
                    warranty_preference=warranty_val,
                    keywords=row["keywords"],
                    last_search_filters=parsed_last_search,
                    updated_at=datetime.fromisoformat(row["updated_at"])
                )
        finally:
            conn.close()

    def save_preferences(self, user_id: str, patch: PreferencePatch) -> UserPreferences:
        """Applies a patch non-destructively to existing user preferences and upserts to SQLite."""
        clean_user_id = str(user_id).strip()
        self.get_or_create_profile(clean_user_id)

        current_prefs = self.get_preferences(clean_user_id)
        if current_prefs is None:
            current_prefs = UserPreferences(user_id=clean_user_id)

        # Apply patch non-destructively
        current_prefs.apply_patch(patch)
        now_utc = datetime.now(timezone.utc).isoformat()

        warranty_int: Optional[int] = None
        if current_prefs.warranty_preference is not None:
            warranty_int = 1 if current_prefs.warranty_preference else 0

        last_search_raw: Optional[str] = None
        if current_prefs.last_search_filters:
            last_search_raw = current_prefs.last_search_filters.model_dump_json(exclude_none=True)

        conn = get_db_connection(self.db_path)
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO user_preferences (
                        user_id, preferred_make, preferred_model, min_year, max_year,
                        min_price_aed, max_price_aed, min_mileage_km, max_mileage_km,
                        min_monthly_payment, max_monthly_payment, regional_specs,
                        warranty_preference, keywords, last_search_filters, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        preferred_make = excluded.preferred_make,
                        preferred_model = excluded.preferred_model,
                        min_year = excluded.min_year,
                        max_year = excluded.max_year,
                        min_price_aed = excluded.min_price_aed,
                        max_price_aed = excluded.max_price_aed,
                        min_mileage_km = excluded.min_mileage_km,
                        max_mileage_km = excluded.max_mileage_km,
                        min_monthly_payment = excluded.min_monthly_payment,
                        max_monthly_payment = excluded.max_monthly_payment,
                        regional_specs = excluded.regional_specs,
                        warranty_preference = excluded.warranty_preference,
                        keywords = excluded.keywords,
                        updated_at = excluded.updated_at
                    """,
                    (
                        clean_user_id,
                        current_prefs.preferred_make,
                        current_prefs.preferred_model,
                        current_prefs.min_year,
                        current_prefs.max_year,
                        current_prefs.min_price_aed,
                        current_prefs.max_price_aed,
                        current_prefs.min_mileage_km,
                        current_prefs.max_mileage_km,
                        current_prefs.min_monthly_payment,
                        current_prefs.max_monthly_payment,
                        current_prefs.regional_specs,
                        warranty_int,
                        current_prefs.keywords,
                        last_search_raw,
                        now_utc,
                    )
                )
        finally:
            conn.close()

        current_prefs.updated_at = datetime.fromisoformat(now_utc)
        return current_prefs

    def update_last_search(self, user_id: str, query_filters: ParsedInventoryQuery) -> None:
        """Persists the JSON-serialized last executed search query without altering explicit preferences."""
        clean_user_id = str(user_id).strip()
        self.get_or_create_profile(clean_user_id)

        now_utc = datetime.now(timezone.utc).isoformat()
        last_search_json = query_filters.model_dump_json(exclude_none=True)

        conn = get_db_connection(self.db_path)
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO user_preferences (
                        user_id, last_search_filters, updated_at
                    ) VALUES (?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        last_search_filters = excluded.last_search_filters,
                        updated_at = excluded.updated_at
                    """,
                    (clean_user_id, last_search_json, now_utc)
                )
        finally:
            conn.close()

    def save_liked_car(self, user_id: str, listing_id: int) -> bool:
        """Saves a verified listing_id to user's liked cars idempotently. Returns True."""
        clean_user_id = str(user_id).strip()
        self.get_or_create_profile(clean_user_id)

        now_utc = datetime.now(timezone.utc).isoformat()
        conn = get_db_connection(self.db_path)
        try:
            with conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO liked_cars (user_id, listing_id, liked_at)
                    VALUES (?, ?, ?)
                    """,
                    (clean_user_id, int(listing_id), now_utc)
                )
            return True
        finally:
            conn.close()

    def remove_liked_car(self, user_id: str, listing_id: int) -> bool:
        """Removes a listing_id from user's liked cars. Returns True if a record was removed."""
        clean_user_id = str(user_id).strip()
        conn = get_db_connection(self.db_path)
        try:
            with conn:
                cursor = conn.execute(
                    "DELETE FROM liked_cars WHERE user_id = ? AND listing_id = ?",
                    (clean_user_id, int(listing_id))
                )
                return cursor.rowcount > 0
        finally:
            conn.close()

    def get_liked_listing_ids(self, user_id: str) -> List[int]:
        """Returns ordered list of liked listing_ids for user_id (insertion order ascending)."""
        clean_user_id = str(user_id).strip()
        conn = get_db_connection(self.db_path)
        try:
            with conn:
                rows = conn.execute(
                    "SELECT listing_id FROM liked_cars WHERE user_id = ? ORDER BY id ASC",
                    (clean_user_id,)
                ).fetchall()
                return [int(r["listing_id"]) for r in rows]
        finally:
            conn.close()

    def clear_liked_cars(self, user_id: str) -> int:
        """Removes all liked cars for a user. Returns count of deleted rows."""
        clean_user_id = str(user_id).strip()
        conn = get_db_connection(self.db_path)
        try:
            with conn:
                cursor = conn.execute("DELETE FROM liked_cars WHERE user_id = ?", (clean_user_id,))
                return cursor.rowcount
        finally:
            conn.close()

    def clear_preferences(self, user_id: str) -> bool:
        """Resets explicit preference fields for a user while preserving last_search_filters."""
        clean_user_id = str(user_id).strip()
        now_utc = datetime.now(timezone.utc).isoformat()
        conn = get_db_connection(self.db_path)
        try:
            with conn:
                cursor = conn.execute(
                    """
                    UPDATE user_preferences
                    SET preferred_make = NULL,
                        preferred_model = NULL,
                        min_year = NULL,
                        max_year = NULL,
                        min_price_aed = NULL,
                        max_price_aed = NULL,
                        min_mileage_km = NULL,
                        max_mileage_km = NULL,
                        min_monthly_payment = NULL,
                        max_monthly_payment = NULL,
                        regional_specs = NULL,
                        warranty_preference = NULL,
                        keywords = NULL,
                        updated_at = ?
                    WHERE user_id = ?
                    """,
                    (now_utc, clean_user_id)
                )
                return cursor.rowcount > 0
        finally:
            conn.close()

    def delete_user_data(self, user_id: str) -> bool:
        """Deletes all user profile data, preferences, and liked cars transactionally."""
        clean_user_id = str(user_id).strip()
        conn = get_db_connection(self.db_path)
        try:
            with conn:
                cursor = conn.execute("DELETE FROM user_profiles WHERE user_id = ?", (clean_user_id,))
                return cursor.rowcount > 0
        finally:
            conn.close()
