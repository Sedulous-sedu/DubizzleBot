"""Service for validating business hours, appointment rules, and persisting bookings."""

import logging
from datetime import datetime, time, timezone
from typing import Optional, List
from zoneinfo import ZoneInfo

from backend.config import settings
from backend.database.connection import get_db_connection, resolve_db_path, init_db
from backend.models.booking import (
    BookingStatus,
    ConfirmedBooking,
    BookingValidationResult,
)
from backend.services.persistent_memory import PersistentMemoryService

logger = logging.getLogger(__name__)

class BookingService:
    """Handles appointment validation against business rules and SQLite booking persistence."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        persistent_memory: Optional[PersistentMemoryService] = None,
        timezone_name: Optional[str] = None,
    ):
        self.db_path = resolve_db_path(db_path)
        self.persistent_memory = persistent_memory or PersistentMemoryService(db_path=self.db_path)
        self.timezone_name = timezone_name or settings.BOOKING_TIMEZONE
        # Initialize schema idempotently
        init_db(self.db_path)

    def get_timezone(self) -> ZoneInfo:
        """Returns the configured ZoneInfo instance."""
        try:
            return ZoneInfo(self.timezone_name)
        except Exception:
            return ZoneInfo("Asia/Dubai")

    def validate_appointment(
        self,
        dt: datetime,
        current_time: Optional[datetime] = None,
    ) -> BookingValidationResult:
        """
        Validates an appointment datetime against business rules:
        - Must be strictly in the future.
        - Monday through Saturday only (Sunday is closed).
        - Operating hours: 08:00 to 20:00 (8:00 AM to 8:00 PM Asia/Dubai) inclusive.
        """
        tz = self.get_timezone()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        else:
            dt = dt.astimezone(tz)

        if current_time is None:
            now = datetime.now(tz)
        else:
            if current_time.tzinfo is None:
                now = current_time.replace(tzinfo=tz)
            else:
                now = current_time.astimezone(tz)

        # 1. Must be future-facing
        if dt <= now:
            return BookingValidationResult(
                is_valid=False,
                error_message="Appointments must be pre-booked for a future date and time. Please select an upcoming slot.",
            )

        # 2. Check Day of Week: Monday=0, ..., Saturday=5, Sunday=6
        if dt.weekday() == 6:  # Sunday
            return BookingValidationResult(
                is_valid=False,
                error_message=(
                    "We are open for test drives and viewings Monday through Saturday (8:00 AM to 8:00 PM). "
                    "We are closed on Sundays. Please select a time between Monday and Saturday."
                ),
            )

        # 3. Check Operating Hours: 08:00 to 20:00 inclusive
        app_time = dt.time()
        start_time = time(8, 0)
        end_time = time(20, 0)

        if app_time < start_time or app_time > end_time:
            return BookingValidationResult(
                is_valid=False,
                error_message=(
                    "Our viewing and test-drive hours are from 8:00 AM to 8:00 PM (Asia/Dubai). "
                    "Please select an appointment time within our operating hours."
                ),
            )

        return BookingValidationResult(
            is_valid=True,
            appointment_at=dt,
        )

    def save_booking(self, booking: ConfirmedBooking) -> bool:
        """
        Persists a confirmed booking to SQLite safely and idempotently.
        Guarantees user profile exists before insertion to satisfy foreign key constraints.
        Uses INSERT ... ON CONFLICT(booking_id) DO NOTHING to strictly preserve existing confirmed
        records and their original created_at timestamp against conflicting mutations.
        """
        # Ensure user profile exists
        self.persistent_memory.get_or_create_profile(booking.user_id)

        conn = get_db_connection(self.db_path)
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO bookings (
                        booking_id,
                        user_id,
                        listing_id,
                        appointment_at,
                        customer_name,
                        customer_phone,
                        customer_email,
                        status,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(booking_id) DO NOTHING
                    """,
                    (
                        booking.booking_id,
                        booking.user_id,
                        booking.listing_id,
                        booking.appointment_at.isoformat(),
                        booking.customer_name,
                        booking.customer_phone,
                        booking.customer_email,
                        booking.status.value,
                        booking.created_at.isoformat(),
                    ),
                )
            return True
        finally:
            conn.close()

    def get_booking(self, booking_id: str) -> Optional[ConfirmedBooking]:
        """Retrieves a single booking by booking_id."""
        conn = get_db_connection(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT booking_id, user_id, listing_id, appointment_at, customer_name, customer_phone, customer_email, status, created_at
                FROM bookings
                WHERE booking_id = ?
                """,
                (booking_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_booking(row)
        finally:
            conn.close()

    def get_user_bookings(self, user_id: str) -> List[ConfirmedBooking]:
        """Retrieves all bookings for a user ordered by appointment date."""
        conn = get_db_connection(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT booking_id, user_id, listing_id, appointment_at, customer_name, customer_phone, customer_email, status, created_at
                FROM bookings
                WHERE user_id = ?
                ORDER BY appointment_at ASC
                """,
                (user_id,),
            )
            rows = cur.fetchall()
            return [self._row_to_booking(row) for row in rows]
        finally:
            conn.close()

    def cancel_booking(self, user_id: str, booking_id: str) -> bool:
        """Cancels an existing booking for a user."""
        conn = get_db_connection(self.db_path)
        try:
            with conn:
                cur = conn.execute(
                    """
                    UPDATE bookings
                    SET status = ?
                    WHERE booking_id = ? AND user_id = ?
                    """,
                    (BookingStatus.CANCELLED.value, booking_id, user_id),
                )
                return cur.rowcount > 0
        finally:
            conn.close()

    def _row_to_booking(self, row) -> ConfirmedBooking:
        return ConfirmedBooking(
            booking_id=row["booking_id"],
            user_id=row["user_id"],
            listing_id=row["listing_id"],
            appointment_at=datetime.fromisoformat(row["appointment_at"]),
            customer_name=row["customer_name"],
            customer_phone=row["customer_phone"],
            customer_email=row["customer_email"],
            status=BookingStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )
