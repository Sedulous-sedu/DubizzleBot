"""Comprehensive offline tests for Phase 5 test-drive and viewing bookings."""

from datetime import datetime, date, time
from zoneinfo import ZoneInfo
from unittest.mock import MagicMock
import pytest

from backend.models.booking import (
    BookingDraft,
    ConfirmedBooking,
    WorkflowStatus,
    BookingStatus,
)
from backend.models.car import CarListing
from backend.models.chat import ChatRequest
from backend.models.intent import (
    ParsedUserIntent,
    UserIntentEnum,
    SearchReadinessState,
    ParsedInventoryQuery,
)
from backend.services.booking import BookingService
from backend.services.inventory import InventoryService
from backend.services.memory import MemoryService
from backend.services.persistent_memory import PersistentMemoryService
from backend.services.phase5_resolver import Phase5Resolver, Phase5Action
from backend.services.orchestrator import ChatOrchestrator

@pytest.fixture
def real_inventory():
    return InventoryService()

@pytest.fixture
def dubai_tz():
    return ZoneInfo("Asia/Dubai")

# Reference clock: Wednesday, August 26, 2026 at 10:00 AM Asia/Dubai
@pytest.fixture
def frozen_now(dubai_tz):
    return datetime(2026, 8, 26, 10, 0, 0, tzinfo=dubai_tz)

def test_validate_appointment_monday_morning_valid(tmp_path, frozen_now, dubai_tz):
    """Verify Monday 8:00 AM is a valid appointment."""
    db_file = str(tmp_path / "book_test.db")
    service = BookingService(db_path=db_file)
    # Next Monday: August 31, 2026 at 08:00
    mon_dt = datetime(2026, 8, 31, 8, 0, 0, tzinfo=dubai_tz)
    res = service.validate_appointment(mon_dt, current_time=frozen_now)
    assert res.is_valid is True
    assert res.error_message is None

def test_validate_appointment_saturday_evening_boundary_valid(tmp_path, frozen_now, dubai_tz):
    """Verify Saturday 8:00 PM (20:00) is a valid boundary appointment."""
    db_file = str(tmp_path / "book_test.db")
    service = BookingService(db_path=db_file)
    # Next Saturday: August 29, 2026 at 20:00
    sat_dt = datetime(2026, 8, 29, 20, 0, 0, tzinfo=dubai_tz)
    res = service.validate_appointment(sat_dt, current_time=frozen_now)
    assert res.is_valid is True
    assert res.error_message is None

def test_validate_appointment_sunday_invalid(tmp_path, frozen_now, dubai_tz):
    """Verify Sunday appointments are strictly rejected with explanation."""
    db_file = str(tmp_path / "book_test.db")
    service = BookingService(db_path=db_file)
    # Next Sunday: August 30, 2026 at 15:00
    sun_dt = datetime(2026, 8, 30, 15, 0, 0, tzinfo=dubai_tz)
    res = service.validate_appointment(sun_dt, current_time=frozen_now)
    assert res.is_valid is False
    assert "closed on sundays" in res.error_message.lower()

def test_validate_appointment_before_8am_invalid(tmp_path, frozen_now, dubai_tz):
    """Verify 7:30 AM appointment is rejected as outside business hours."""
    db_file = str(tmp_path / "book_test.db")
    service = BookingService(db_path=db_file)
    dt = datetime(2026, 8, 29, 7, 30, 0, tzinfo=dubai_tz)
    res = service.validate_appointment(dt, current_time=frozen_now)
    assert res.is_valid is False
    assert "8:00 am to 8:00 pm" in res.error_message.lower()

def test_validate_appointment_after_8pm_invalid(tmp_path, frozen_now, dubai_tz):
    """Verify 8:30 PM appointment is rejected as outside business hours."""
    db_file = str(tmp_path / "book_test.db")
    service = BookingService(db_path=db_file)
    dt = datetime(2026, 8, 29, 20, 30, 0, tzinfo=dubai_tz)
    res = service.validate_appointment(dt, current_time=frozen_now)
    assert res.is_valid is False
    assert "8:00 am to 8:00 pm" in res.error_message.lower()

def test_validate_appointment_past_datetime_invalid(tmp_path, frozen_now, dubai_tz):
    """Verify past datetime is rejected."""
    db_file = str(tmp_path / "book_test.db")
    service = BookingService(db_path=db_file)
    past_dt = datetime(2026, 8, 25, 14, 0, 0, tzinfo=dubai_tz)
    res = service.validate_appointment(past_dt, current_time=frozen_now)
    assert res.is_valid is False
    assert "future" in res.error_message.lower()

def test_booking_save_and_retrieve_idempotent(tmp_path, dubai_tz):
    """Verify save_booking persists and is idempotent by booking_id."""
    db_file = str(tmp_path / "book_test.db")
    service = BookingService(db_path=db_file)
    booking = ConfirmedBooking(
        booking_id="BK-TEST01",
        user_id="user_booking_test",
        listing_id=9,
        appointment_at=datetime(2026, 8, 29, 15, 0, 0, tzinfo=dubai_tz),
        created_at=datetime(2026, 8, 26, 10, 0, 0, tzinfo=dubai_tz),
    )
    # Save once
    assert service.save_booking(booking) is True
    # Save duplicate
    assert service.save_booking(booking) is True

    retrieved = service.get_booking("BK-TEST01")
    assert retrieved is not None
    assert retrieved.booking_id == "BK-TEST01"
    assert retrieved.listing_id == 9
    assert retrieved.status == BookingStatus.CONFIRMED

def test_booking_auto_creates_user_profile_for_new_user(tmp_path, dubai_tz):
    """Verify first-time user booking creates user_profile without foreign key failure."""
    db_file = str(tmp_path / "book_test.db")
    pmem = PersistentMemoryService(db_path=db_file)
    service = BookingService(db_path=db_file, persistent_memory=pmem)

    new_user_id = "brand_new_user_123"
    # Ensure profile does not exist yet
    assert pmem.get_profile(new_user_id) is None

    booking = ConfirmedBooking(
        booking_id="BK-NEW01",
        user_id=new_user_id,
        listing_id=17,
        appointment_at=datetime(2026, 8, 29, 11, 0, 0, tzinfo=dubai_tz),
        created_at=datetime(2026, 8, 26, 10, 0, 0, tzinfo=dubai_tz),
    )
    assert service.save_booking(booking) is True

    # Profile now automatically exists
    profile = pmem.get_profile(new_user_id)
    assert profile is not None
    assert profile.user_id == new_user_id

def test_phase5_resolver_date_and_time_parsing(frozen_now):
    """Verify Phase5Resolver parses relative dates and times with clock injection."""
    resolver = Phase5Resolver()

    # "tomorrow at 3 PM" -> August 27, 2026 15:00
    d, t, ambig, raw_d, raw_t = resolver.parse_datetime_expression("tomorrow at 3 PM", frozen_now)
    assert d == date(2026, 8, 27)
    assert t == time(15, 0)
    assert ambig is False

    # "Saturday at 11:30 AM" -> August 29, 2026 11:30
    d, t, ambig, raw_d, raw_t = resolver.parse_datetime_expression("Saturday at 11:30 AM", frozen_now)
    assert d == date(2026, 8, 29)
    assert t == time(11, 30)
    assert ambig is False

    # "at 4" -> ambiguous time
    d, t, ambig, raw_d, raw_t = resolver.parse_datetime_expression("at 4", frozen_now)
    assert ambig is True

def test_orchestrator_multi_turn_booking_flow(real_inventory, tmp_path, frozen_now):
    """
    Test complete multi-turn booking flow:
    Turn 1: Search Bentleys
    Turn 2: "I want to test drive the second one" (Listing #17) -> asks for date/time
    Turn 3: "Saturday" -> asks for time
    Turn 4: "3 PM" -> summarizes and asks for confirmation
    Turn 5: "Yes please" -> confirms and persists to SQLite
    """
    db_file = str(tmp_path / "orch_booking.db")
    pmem = PersistentMemoryService(db_path=db_file)
    bservice = BookingService(db_path=db_file, persistent_memory=pmem)
    mock_interp = MagicMock()
    mock_interp.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        readiness_state=SearchReadinessState.READY,
        query_filters=ParsedInventoryQuery(make="Bentley")
    )
    orch = ChatOrchestrator(
        query_interpreter=mock_interp,
        inventory_service=real_inventory,
        persistent_memory=pmem,
        booking_service=bservice,
        current_time_override=frozen_now
    )
    user_id = "booking_user_01"
    session_id = "sess_book_01"

    # Turn 1: Search Bentleys
    res1 = orch.process_chat(ChatRequest(user_id=user_id, session_id=session_id, message="Show me Bentleys"))
    assert res1.total_matches > 1
    second_car = res1.matched_cars[1]
    assert second_car.listing_id == 17

    # Turn 2: "I want to test drive the second one"
    res2 = orch.process_chat(ChatRequest(user_id=user_id, session_id=session_id, message="I want to test drive the second one."))
    assert "2020 bentley continental" in res2.response.lower()
    assert "what date and time" in res2.response.lower()

    # Turn 3: "Saturday"
    res3 = orch.process_chat(ChatRequest(user_id=user_id, session_id=session_id, message="Saturday"))
    assert "what time on saturday" in res3.response.lower()

    # Turn 4: "3 PM"
    res4 = orch.process_chat(ChatRequest(user_id=user_id, session_id=session_id, message="3 PM"))
    assert "please confirm your test-drive booking details" in res4.response.lower()
    assert "saturday, august 29, 2026 at 03:00 pm" in res4.response.lower()
    assert "would you like me to confirm this test drive?" in res4.response.lower()

    # Verify not yet in SQLite
    user_bookings = bservice.get_user_bookings(user_id)
    assert len(user_bookings) == 0

    # Turn 5: "Yes please" -> explicit confirmation
    res5 = orch.process_chat(ChatRequest(user_id=user_id, session_id=session_id, message="Yes please"))
    assert "has been confirmed for saturday, august 29, 2026 at 03:00 pm" in res5.response.lower()
    assert "booking ref:" in res5.response.lower()

    # Verify persisted in SQLite
    user_bookings = bservice.get_user_bookings(user_id)
    assert len(user_bookings) == 1
    assert user_bookings[0].listing_id == 17

def test_orchestrator_booking_sunday_rejection(real_inventory, tmp_path, frozen_now):
    """Verify requesting Sunday explains business hours and does not confirm."""
    db_file = str(tmp_path / "orch_sunday.db")
    pmem = PersistentMemoryService(db_path=db_file)
    bservice = BookingService(db_path=db_file, persistent_memory=pmem)
    mock_interp = MagicMock()
    mock_interp.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        readiness_state=SearchReadinessState.READY,
        query_filters=ParsedInventoryQuery(make="Bentley")
    )
    orch = ChatOrchestrator(
        query_interpreter=mock_interp,
        inventory_service=real_inventory,
        persistent_memory=pmem,
        booking_service=bservice,
        current_time_override=frozen_now
    )
    user_id = "user_sunday"
    session_id = "sess_sunday"

    # Turn 1: Search
    orch.process_chat(ChatRequest(user_id=user_id, session_id=session_id, message="Show me Bentleys"))

    # Turn 2: Test drive Sunday 3 PM
    res2 = orch.process_chat(ChatRequest(user_id=user_id, session_id=session_id, message="Can I test drive the second one Sunday at 3 PM?"))
    assert "closed on sundays" in res2.response.lower()

    # Verify zero bookings in SQLite
    assert bservice.get_user_bookings(user_id) == []

def test_orchestrator_unrelated_query_preserves_pending_booking(real_inventory, tmp_path, frozen_now):
    """Verify an unrelated vehicle fact question routes to Phase 4A while preserving the pending booking draft."""
    db_file = str(tmp_path / "orch_unrelated.db")
    pmem = PersistentMemoryService(db_path=db_file)
    bservice = BookingService(db_path=db_file, persistent_memory=pmem)
    mem_service = MemoryService()
    mock_interp = MagicMock()
    mock_interp.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        readiness_state=SearchReadinessState.READY,
        query_filters=ParsedInventoryQuery(make="Bentley")
    )
    orch = ChatOrchestrator(
        query_interpreter=mock_interp,
        inventory_service=real_inventory,
        memory_service=mem_service,
        persistent_memory=pmem,
        booking_service=bservice,
        current_time_override=frozen_now
    )
    user_id = "user_unrelated"
    session_id = "sess_unrelated"

    # Search Bentleys
    orch.process_chat(ChatRequest(user_id=user_id, session_id=session_id, message="Show me Bentleys"))

    # Start booking
    orch.process_chat(ChatRequest(user_id=user_id, session_id=session_id, message="I want to test drive the second one."))

    session = mem_service.get_or_create_session(user_id, session_id)
    assert session.pending_booking is not None

    # Unrelated fact question: "What's its mileage?"
    res_fact = orch.process_chat(ChatRequest(user_id=user_id, session_id=session_id, message="What's its mileage?"))
    assert "318 km" in res_fact.response

    # Pending booking survives!
    assert session.pending_booking is not None
    assert session.pending_booking.listing_id == 17

def test_orchestrator_descriptive_viewing_request_returns_candidates(real_inventory, tmp_path, frozen_now):
    """Verify 'I want to test drive a Bentley' returns candidate inventory vehicles rather than failing."""
    db_file = str(tmp_path / "orch_desc.db")
    pmem = PersistentMemoryService(db_path=db_file)
    bservice = BookingService(db_path=db_file, persistent_memory=pmem)
    mock_interp = MagicMock()
    mock_interp.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
        readiness_state=SearchReadinessState.READY,
        query_filters=ParsedInventoryQuery(make="Bentley")
    )
    orch = ChatOrchestrator(
        query_interpreter=mock_interp,
        inventory_service=real_inventory,
        persistent_memory=pmem,
        booking_service=bservice,
        current_time_override=frozen_now
    )
    user_id = "user_desc"
    session_id = "sess_desc"

    res = orch.process_chat(ChatRequest(user_id=user_id, session_id=session_id, message="I want to test drive a Bentley under AED 200,000"))
    assert res.total_matches > 0
    assert "candidate vehicles in our inventory" in res.response.lower()

def test_booking_simulated_restart_retrieves_confirmed_booking(real_inventory, tmp_path, frozen_now, dubai_tz):
    """Verify restarting service processes retains confirmed bookings from SQLite."""
    db_file = str(tmp_path / "restart_book.db")
    pmem1 = PersistentMemoryService(db_path=db_file)
    bservice1 = BookingService(db_path=db_file, persistent_memory=pmem1)
    user_id = "user_restart"

    booking = ConfirmedBooking(
        booking_id="BK-RESTART01",
        user_id=user_id,
        listing_id=17,
        appointment_at=datetime(2026, 8, 29, 14, 0, 0, tzinfo=dubai_tz),
        created_at=datetime(2026, 8, 26, 10, 0, 0, tzinfo=dubai_tz),
    )
    bservice1.save_booking(booking)

    # Completely re-instantiate services against the same DB file
    pmem2 = PersistentMemoryService(db_path=db_file)
    bservice2 = BookingService(db_path=db_file, persistent_memory=pmem2)

    retrieved = bservice2.get_booking("BK-RESTART01")
    assert retrieved is not None
    assert retrieved.booking_id == "BK-RESTART01"
    assert retrieved.listing_id == 17

def test_booking_cross_user_isolation(tmp_path, dubai_tz):
    """Verify User A's bookings are completely invisible to User B."""
    db_file = str(tmp_path / "iso_book.db")
    service = BookingService(db_path=db_file)

    service.save_booking(ConfirmedBooking(
        booking_id="BK-USERA",
        user_id="user_a",
        listing_id=9,
        appointment_at=datetime(2026, 8, 29, 10, 0, 0, tzinfo=dubai_tz),
        created_at=datetime(2026, 8, 26, 10, 0, 0, tzinfo=dubai_tz),
    ))

    bookings_b = service.get_user_bookings("user_b")
    assert len(bookings_b) == 0

def test_booking_unresolvable_ordinal_returns_clarification_no_booking(real_inventory, tmp_path, frozen_now):
    """Verify requesting an out-of-range ordinal (e.g. 5th car when 2 exist) does not book."""
    db_file = str(tmp_path / "orch_unres.db")
    pmem = PersistentMemoryService(db_path=db_file)
    bservice = BookingService(db_path=db_file, persistent_memory=pmem)
    mock_interp = MagicMock()
    mock_interp.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        readiness_state=SearchReadinessState.READY,
        query_filters=ParsedInventoryQuery(make="Bentley")
    )
    orch = ChatOrchestrator(
        query_interpreter=mock_interp,
        inventory_service=real_inventory,
        persistent_memory=pmem,
        booking_service=bservice,
        current_time_override=frozen_now
    )
    user_id = "user_unres"
    session_id = "sess_unres"

    # Search (returns 7 cars)
    orch.process_chat(ChatRequest(user_id=user_id, session_id=session_id, message="Show me Bentleys"))

    # Ask for 20th car
    res2 = orch.process_chat(ChatRequest(user_id=user_id, session_id=session_id, message="I want to test drive the 20th one."))
    assert "only 7" in res2.response.lower() or "which vehicle" in res2.response.lower()
    assert bservice.get_user_bookings(user_id) == []

def test_booking_idempotency_same_payload_one_row(tmp_path, dubai_tz):
    """Verify same booking_id with same payload results in exactly one row in SQLite."""
    db_file = str(tmp_path / "idem_book.db")
    service = BookingService(db_path=db_file)
    booking = ConfirmedBooking(
        booking_id="BK-IDEM01",
        user_id="user_idem",
        listing_id=15,
        appointment_at=datetime(2026, 8, 29, 14, 0, 0, tzinfo=dubai_tz),
        created_at=datetime(2026, 8, 26, 10, 0, 0, tzinfo=dubai_tz),
    )
    # Insert first time
    assert service.save_booking(booking) is True
    # Insert second time identical
    assert service.save_booking(booking) is True

    user_bookings = service.get_user_bookings("user_idem")
    assert len(user_bookings) == 1
    assert user_bookings[0].booking_id == "BK-IDEM01"
    assert user_bookings[0].listing_id == 15

def test_booking_conflicting_retry_different_appointment_unchanged(tmp_path, dubai_tz):
    """Verify conflicting retry with same booking_id but different appointment does not mutate the confirmed record."""
    db_file = str(tmp_path / "conflict_app_book.db")
    service = BookingService(db_path=db_file)
    orig_dt = datetime(2026, 8, 29, 10, 0, 0, tzinfo=dubai_tz)
    booking1 = ConfirmedBooking(
        booking_id="BK-CONF01",
        user_id="user_conf",
        listing_id=15,
        appointment_at=orig_dt,
        created_at=datetime(2026, 8, 26, 10, 0, 0, tzinfo=dubai_tz),
    )
    service.save_booking(booking1)

    # Conflicting second save with different appointment
    conflict_dt = datetime(2026, 8, 29, 16, 0, 0, tzinfo=dubai_tz)
    booking2 = ConfirmedBooking(
        booking_id="BK-CONF01",
        user_id="user_conf",
        listing_id=15,
        appointment_at=conflict_dt,
        created_at=datetime(2026, 8, 26, 12, 0, 0, tzinfo=dubai_tz),
    )
    service.save_booking(booking2)

    retrieved = service.get_booking("BK-CONF01")
    assert retrieved is not None
    # Original appointment time must remain unchanged!
    assert retrieved.appointment_at == orig_dt

def test_booking_conflicting_retry_different_listing_unchanged(tmp_path, dubai_tz):
    """Verify conflicting retry with same booking_id but different listing does not mutate the confirmed record."""
    db_file = str(tmp_path / "conflict_list_book.db")
    service = BookingService(db_path=db_file)
    booking1 = ConfirmedBooking(
        booking_id="BK-CONF02",
        user_id="user_conf2",
        listing_id=9,
        appointment_at=datetime(2026, 8, 29, 10, 0, 0, tzinfo=dubai_tz),
        created_at=datetime(2026, 8, 26, 10, 0, 0, tzinfo=dubai_tz),
    )
    service.save_booking(booking1)

    # Conflicting second save with different listing
    booking2 = ConfirmedBooking(
        booking_id="BK-CONF02",
        user_id="user_conf2",
        listing_id=17,
        appointment_at=datetime(2026, 8, 29, 10, 0, 0, tzinfo=dubai_tz),
        created_at=datetime(2026, 8, 26, 12, 0, 0, tzinfo=dubai_tz),
    )
    service.save_booking(booking2)

    retrieved = service.get_booking("BK-CONF02")
    assert retrieved is not None
    # Original listing ID must remain unchanged!
    assert retrieved.listing_id == 9

def test_booking_created_at_immutability_on_retry(tmp_path, dubai_tz):
    """Verify created_at remains the timestamp from original successful confirmation on retry."""
    db_file = str(tmp_path / "created_at_book.db")
    service = BookingService(db_path=db_file)
    orig_created_at = datetime(2026, 8, 26, 10, 0, 0, tzinfo=dubai_tz)
    booking1 = ConfirmedBooking(
        booking_id="BK-TIME01",
        user_id="user_time",
        listing_id=9,
        appointment_at=datetime(2026, 8, 29, 10, 0, 0, tzinfo=dubai_tz),
        created_at=orig_created_at,
    )
    service.save_booking(booking1)

    # Retry with later created_at timestamp
    later_created_at = datetime(2026, 8, 26, 15, 30, 0, tzinfo=dubai_tz)
    booking2 = ConfirmedBooking(
        booking_id="BK-TIME01",
        user_id="user_time",
        listing_id=9,
        appointment_at=datetime(2026, 8, 29, 10, 0, 0, tzinfo=dubai_tz),
        created_at=later_created_at,
    )
    service.save_booking(booking2)

    retrieved = service.get_booking("BK-TIME01")
    assert retrieved is not None
    # Original created_at MUST be preserved!
    assert retrieved.created_at == orig_created_at

def test_booking_combined_one_turn_request_awaiting_confirmation(real_inventory, tmp_path, frozen_now):
    """
    Verify: 'I want to test drive the second one Saturday at 3 PM' in a single turn:
    - Resolves second car
    - Parses date/time
    - Validates business hours
    - Becomes AWAITING_CONFIRMATION (zero DB rows yet)
    - Then 'Yes, confirm' confirms exactly one row.
    """
    db_file = str(tmp_path / "orch_oneturn.db")
    pmem = PersistentMemoryService(db_path=db_file)
    bservice = BookingService(db_path=db_file, persistent_memory=pmem)
    mock_interp = MagicMock()
    mock_interp.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        readiness_state=SearchReadinessState.READY,
        query_filters=ParsedInventoryQuery(make="Bentley")
    )
    orch = ChatOrchestrator(
        query_interpreter=mock_interp,
        inventory_service=real_inventory,
        persistent_memory=pmem,
        booking_service=bservice,
        current_time_override=frozen_now
    )
    user_id = "user_oneturn"
    session_id = "sess_oneturn"

    # Turn 1: Search Bentleys
    orch.process_chat(ChatRequest(user_id=user_id, session_id=session_id, message="Show me Bentleys"))

    # Turn 2: Combined request: vehicle + date + time in one shot!
    res2 = orch.process_chat(ChatRequest(
        user_id=user_id,
        session_id=session_id,
        message="I want to test drive the second one Saturday at 3 PM"
    ))
    assert "please confirm your test-drive booking details" in res2.response.lower()
    assert "saturday, august 29, 2026 at 03:00 pm" in res2.response.lower()
    assert "would you like me to confirm this test drive?" in res2.response.lower()
    assert len(bservice.get_user_bookings(user_id)) == 0

    # Turn 3: "Yes, confirm"
    res3 = orch.process_chat(ChatRequest(
        user_id=user_id,
        session_id=session_id,
        message="Yes, confirm"
    ))
    assert "has been confirmed" in res3.response.lower()
    user_bookings = bservice.get_user_bookings(user_id)
    assert len(user_bookings) == 1
    assert user_bookings[0].listing_id == 17

def test_booking_invalid_slot_recovery_multi_turn(real_inventory, tmp_path, frozen_now):
    """
    Test multi-turn recovery after an invalid time slot:
    Turn 1: Search Bentleys
    Turn 2: "I want to test drive the first one Sunday at 3 PM" -> Sunday rejected, zero DB rows, vehicle retained in draft
    Turn 3: "Saturday at 3 PM" -> valid replacement accepted, asks for confirmation
    Turn 4: "Confirm" -> one booking persisted
    """
    db_file = str(tmp_path / "orch_recovery.db")
    pmem = PersistentMemoryService(db_path=db_file)
    bservice = BookingService(db_path=db_file, persistent_memory=pmem)
    mock_interp = MagicMock()
    mock_interp.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        readiness_state=SearchReadinessState.READY,
        query_filters=ParsedInventoryQuery(make="Bentley")
    )
    orch = ChatOrchestrator(
        query_interpreter=mock_interp,
        inventory_service=real_inventory,
        persistent_memory=pmem,
        booking_service=bservice,
        current_time_override=frozen_now
    )
    user_id = "user_recov"
    session_id = "sess_recov"

    # Search
    orch.process_chat(ChatRequest(user_id=user_id, session_id=session_id, message="Show me Bentleys"))

    # Invalid Sunday attempt
    res2 = orch.process_chat(ChatRequest(
        user_id=user_id,
        session_id=session_id,
        message="I want to test drive the first one Sunday at 3 PM"
    ))
    assert "closed on sundays" in res2.response.lower()
    assert len(bservice.get_user_bookings(user_id)) == 0

    # User corrects to Saturday at 3 PM without having to re-specify the car
    res3 = orch.process_chat(ChatRequest(
        user_id=user_id,
        session_id=session_id,
        message="Saturday at 3 PM"
    ))
    assert "please confirm your test-drive booking details" in res3.response.lower()
    assert "saturday, august 29, 2026 at 03:00 pm" in res3.response.lower()

    # Confirm
    res4 = orch.process_chat(ChatRequest(
        user_id=user_id,
        session_id=session_id,
        message="Confirm"
    ))
    assert "has been confirmed" in res4.response.lower()
    assert len(bservice.get_user_bookings(user_id)) == 1

def test_booking_pending_workflow_interruption_preserves_draft(real_inventory, tmp_path, frozen_now):
    """
    Test pending workflow interruption:
    1. Pending booking awaiting confirmation for Listing #17
    2. User asks: "What's its mileage?" -> Phase 4A answers, pending booking draft remains intact
    3. User asks: "What cars did I like?" -> Phase 4B answers, pending booking draft remains intact
    4. User says: "Confirm the test drive" -> original booking confirms and persists to SQLite!
    """
    db_file = str(tmp_path / "orch_interrup.db")
    pmem = PersistentMemoryService(db_path=db_file)
    bservice = BookingService(db_path=db_file, persistent_memory=pmem)
    mock_interp = MagicMock()
    mock_interp.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        readiness_state=SearchReadinessState.READY,
        query_filters=ParsedInventoryQuery(make="Bentley")
    )
    orch = ChatOrchestrator(
        query_interpreter=mock_interp,
        inventory_service=real_inventory,
        persistent_memory=pmem,
        booking_service=bservice,
        current_time_override=frozen_now
    )
    user_id = "user_interrup"
    session_id = "sess_interrup"

    # Search & initiate booking
    orch.process_chat(ChatRequest(user_id=user_id, session_id=session_id, message="Show me Bentleys"))
    orch.process_chat(ChatRequest(user_id=user_id, session_id=session_id, message="I want to test drive the second one."))
    orch.process_chat(ChatRequest(user_id=user_id, session_id=session_id, message="Saturday at 3 PM"))

    # Step 2: Unrelated Phase 4A vehicle attribute question
    res_mileage = orch.process_chat(ChatRequest(user_id=user_id, session_id=session_id, message="What's its mileage?"))
    assert "318 km" in res_mileage.response
    assert len(bservice.get_user_bookings(user_id)) == 0

    # Step 3: Unrelated Phase 4B memory question
    res_liked = orch.process_chat(ChatRequest(user_id=user_id, session_id=session_id, message="What cars did I like?"))
    assert "saved cars" in res_liked.response.lower() or "favorites" in res_liked.response.lower()
    assert len(bservice.get_user_bookings(user_id)) == 0

    # Step 4: Resume and confirm booking
    res_confirm = orch.process_chat(ChatRequest(user_id=user_id, session_id=session_id, message="Confirm the test drive"))
    assert "has been confirmed" in res_confirm.response.lower()
    saved = bservice.get_user_bookings(user_id)
    assert len(saved) == 1
    assert saved[0].listing_id == 17

