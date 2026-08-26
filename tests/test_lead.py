"""Comprehensive offline tests for Phase 5 lead qualification and CSV capture."""

import csv
import os
import threading
from unittest.mock import MagicMock
import pytest

from backend.models.chat import ChatRequest
from backend.models.intent import (
    ParsedUserIntent,
    UserIntentEnum,
    SearchReadinessState,
    ParsedInventoryQuery,
)
from backend.models.lead import LeadDraft, QualifiedLead
from backend.models.persistent_memory import PreferencePatch
from backend.services.inventory import InventoryService
from backend.services.memory import MemoryService
from backend.services.lead import LeadService
from backend.services.persistent_memory import PersistentMemoryService
from backend.services.orchestrator import ChatOrchestrator

@pytest.fixture
def real_inventory():
    return InventoryService()

def test_lead_service_creates_csv_with_header(tmp_path):
    """Verify LeadService creates the CSV file with standard headers."""
    csv_file = str(tmp_path / "leads_test.csv")
    service = LeadService(csv_path=csv_file)
    assert os.path.exists(csv_file)

    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == LeadService.CSV_HEADERS

def test_lead_service_save_lead_appends_correct_row(tmp_path):
    """Verify save_lead appends formatted qualified lead to CSV."""
    csv_file = str(tmp_path / "leads_test.csv")
    service = LeadService(csv_path=csv_file)

    lead = QualifiedLead(
        lead_id="LEAD-001",
        created_at="2026-08-26T10:00:00Z",
        user_id="user_lead_01",
        session_id="sess_lead_01",
        name="John Doe",
        phone="+971501234567",
        email="john@example.com",
        min_budget_aed=50000.0,
        max_budget_aed=120000.0,
        interested_make="Toyota",
        interested_model="Land Cruiser",
        interested_listing_id=15,
        requirements="GCC Specs, Low Mileage",
        booking_reference="BK-001",
    )

    assert service.save_lead(lead) is True

    leads = service.get_leads()
    assert len(leads) == 1
    assert leads[0].lead_id == "LEAD-001"
    assert leads[0].name == "John Doe"
    assert leads[0].phone == "+971501234567"
    assert leads[0].max_budget_aed == 120000.0
    assert leads[0].interested_listing_id == 15

def test_lead_service_idempotent_no_duplicate_lead_id(tmp_path):
    """Verify submitting the same lead_id twice results in exactly one row in the CSV."""
    csv_file = str(tmp_path / "leads_test.csv")
    service = LeadService(csv_path=csv_file)

    lead = QualifiedLead(
        lead_id="LEAD-DUPE",
        created_at="2026-08-26T10:00:00Z",
        user_id="user_dupe",
        session_id="sess_dupe",
        name="Alex Smith",
        phone="+971509998888",
        max_budget_aed=150000.0,
        interested_make="Bentley",
    )

    # Save twice
    assert service.save_lead(lead) is True
    assert service.save_lead(lead) is True

    leads = service.get_leads()
    assert len(leads) == 1

def test_lead_service_special_characters_commas_and_quotes(tmp_path):
    """Verify requirements with commas and quotes are safely escaped by the CSV writer."""
    csv_file = str(tmp_path / "leads_test.csv")
    service = LeadService(csv_path=csv_file)

    lead = QualifiedLead(
        lead_id="LEAD-QUOTES",
        created_at="2026-08-26T10:00:00Z",
        user_id="user_quotes",
        session_id="sess_quotes",
        name='Dr. O\'Connor, "Chief Buyer"',
        phone="+971501112222",
        requirements='GCC, SUV, "Leather, Sunroof", Under Warranty',
    )

    service.save_lead(lead)

    leads = service.get_leads()
    assert len(leads) == 1
    assert leads[0].name == 'Dr. O\'Connor, "Chief Buyer"'
    assert 'Leather, Sunroof' in leads[0].requirements

def test_lead_draft_completeness_criteria():
    """Verify completeness rules: Contact + Budget + Automotive Need."""
    # 1. Contact + listing but NO budget -> NOT qualified
    draft1 = LeadDraft(
        user_id="u1", session_id="s1",
        phone="+971501234567",
        interested_listing_id=9
    )
    assert draft1.has_contact() is True
    assert draft1.has_automotive_need() is True
    assert draft1.has_budget() is False
    assert draft1.is_fully_qualified() is False

    # 2. Contact + budget but NO automotive need -> NOT qualified
    draft2 = LeadDraft(
        user_id="u2", session_id="s2",
        phone="+971501234567",
        max_budget_aed=100000.0
    )
    assert draft2.has_contact() is True
    assert draft2.has_budget() is True
    assert draft2.has_automotive_need() is False
    assert draft2.is_fully_qualified() is False

    # 3. Budget + automotive need but NO contact -> NOT qualified
    draft3 = LeadDraft(
        user_id="u3", session_id="s3",
        max_budget_aed=100000.0,
        interested_make="Nissan"
    )
    assert draft3.has_contact() is False
    assert draft3.has_budget() is True
    assert draft3.has_automotive_need() is True
    assert draft3.is_fully_qualified() is False

    # 4. Contact + budget + automotive need -> FULLY QUALIFIED
    draft4 = LeadDraft(
        user_id="u4", session_id="s4",
        email="test@example.com",
        max_budget_aed=150000.0,
        requirements="GCC SUV"
    )
    assert draft4.is_fully_qualified() is True

def test_orchestrator_multi_turn_lead_qualification_flow(real_inventory, tmp_path):
    """
    Test complete multi-turn lead qualification flow:
    Turn 1: "I want to submit an enquiry for a GCC car under AED 150,000" (needs contact) -> asks for contact
    Turn 2: "My name is John Doe, phone +971501234567" -> summarizes and asks for confirmation
    Turn 3: "Yes please submit" -> appends to CSV and confirms
    """
    db_file = str(tmp_path / "orch_lead.db")
    csv_file = str(tmp_path / "orch_leads.csv")
    pmem = PersistentMemoryService(db_path=db_file)
    lservice = LeadService(csv_path=csv_file)
    mock_interp = MagicMock()
    mock_interp.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.GENERAL_CHAT,
        readiness_state=SearchReadinessState.READY
    )
    orch = ChatOrchestrator(
        query_interpreter=mock_interp,
        inventory_service=real_inventory,
        persistent_memory=pmem,
        lead_service=lservice
    )
    user_id = "user_lead_flow"
    session_id = "sess_lead_flow"

    # Turn 1: Start lead with budget & requirements
    res1 = orch.process_chat(ChatRequest(
        user_id=user_id, session_id=session_id,
        message="I'd like to submit an enquiry for a GCC SUV under AED 150,000."
    ))
    assert "please provide your phone number or email" in res1.response.lower()

    # Verify zero leads in CSV
    assert len(lservice.get_leads()) == 0

    # Turn 2: Provide contact details
    res2 = orch.process_chat(ChatRequest(
        user_id=user_id, session_id=session_id,
        message="My name is John Doe, phone +971501234567"
    ))
    assert "summary of your enquiry" in res2.response.lower()
    assert "would you like me to submit your enquiry" in res2.response.lower()
    assert "aed 150,000" in res2.response.lower()

    # Verify still not in CSV before confirmation
    assert len(lservice.get_leads()) == 0

    # Turn 3: Confirm
    res3 = orch.process_chat(ChatRequest(
        user_id=user_id, session_id=session_id,
        message="Yes please submit"
    ))
    assert "has been submitted to our sales team" in res3.response.lower()

    # Verify exactly one row in CSV
    leads = lservice.get_leads()
    assert len(leads) == 1
    assert leads[0].name == "John Doe"
    assert leads[0].phone == "+971501234567"
    assert leads[0].max_budget_aed == 150000.0
    assert "GCC Specs" in leads[0].requirements

def test_orchestrator_lead_seeded_from_phase4b_preferences(real_inventory, tmp_path):
    """Verify saved Phase 4B preferences seed a lead draft without treating last search as permanent preference."""
    db_file = str(tmp_path / "orch_seed.db")
    csv_file = str(tmp_path / "orch_seed.csv")
    pmem = PersistentMemoryService(db_path=db_file)
    lservice = LeadService(csv_path=csv_file)
    mock_interp = MagicMock()
    orch = ChatOrchestrator(
        query_interpreter=mock_interp,
        inventory_service=real_inventory,
        persistent_memory=pmem,
        lead_service=lservice
    )
    user_id = "user_seed"
    session_id = "sess_seed"

    # Set explicit Phase 4B preferences: Bentley, max price 180,000, GCC specs
    pmem.save_preferences(user_id, PreferencePatch(
        preferred_make="Bentley",
        max_price_aed=180000.0,
        regional_specs="GCC"
    ))

    # User initiates enquiry
    res1 = orch.process_chat(ChatRequest(
        user_id=user_id, session_id=session_id,
        message="I want a salesperson to contact me about buying a car."
    ))
    # Since budget (180k) and make (Bentley) are seeded from preferences, bot asks for contact
    assert "phone number or email" in res1.response.lower()

    # User provides email
    res2 = orch.process_chat(ChatRequest(
        user_id=user_id, session_id=session_id,
        message="My email is buyer@example.com"
    ))
    assert "summary of your enquiry" in res2.response.lower()
    assert "bentley" in res2.response.lower()
    assert "aed 180,000" in res2.response.lower()

    # User confirms
    orch.process_chat(ChatRequest(user_id=user_id, session_id=session_id, message="Yes confirm"))

    leads = lservice.get_leads()
    assert len(leads) == 1
    assert leads[0].email == "buyer@example.com"
    assert leads[0].interested_make == "Bentley"
    assert leads[0].max_budget_aed == 180000.0

def test_lead_service_thread_safety(tmp_path):
    """Verify concurrent lead saves across threads do not corrupt the CSV."""
    csv_file = str(tmp_path / "concurrent_leads.csv")
    service = LeadService(csv_path=csv_file)

    def worker(i):
        lead = QualifiedLead(
            lead_id=f"LEAD-T{i}",
            created_at="2026-08-26T10:00:00Z",
            user_id=f"user_{i}",
            session_id=f"sess_{i}",
            name=f"Worker {i}",
            phone=f"+97150000000{i}",
            max_budget_aed=100000.0 + i,
            requirements="GCC Specs",
        )
        service.save_lead(lead)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    leads = service.get_leads()
    assert len(leads) == 10
    lead_ids = {l.lead_id for l in leads}
    assert len(lead_ids) == 10

def test_orchestrator_lead_cancellation(real_inventory, tmp_path):
    """Verify cancelling an in-progress enquiry clears pending_lead with zero CSV writes."""
    db_file = str(tmp_path / "orch_cancel.db")
    csv_file = str(tmp_path / "orch_cancel.csv")
    pmem = PersistentMemoryService(db_path=db_file)
    lservice = LeadService(csv_path=csv_file)
    mock_interp = MagicMock()
    orch = ChatOrchestrator(
        query_interpreter=mock_interp,
        inventory_service=real_inventory,
        persistent_memory=pmem,
        lead_service=lservice
    )
    user_id = "user_cancel"
    session_id = "sess_cancel"

    # Start lead
    orch.process_chat(ChatRequest(user_id=user_id, session_id=session_id, message="I'd like to submit an enquiry for an SUV under AED 150k."))

    # Cancel
    res2 = orch.process_chat(ChatRequest(user_id=user_id, session_id=session_id, message="Never mind the enquiry."))
    assert "cancelled your enquiry draft" in res2.response.lower()

    # Zero writes to CSV
    assert len(lservice.get_leads()) == 0

def test_lead_cross_user_isolation(real_inventory, tmp_path):
    """Verify Lead Draft state in SessionState is strictly isolated across users."""
    mem_service = MemoryService()
    orch = ChatOrchestrator(memory_service=mem_service)

    # User A starts lead
    orch.process_chat(ChatRequest(user_id="user_a", session_id="sess_a", message="I'd like to submit an enquiry for a GCC car under AED 120,000."))
    session_a = mem_service.get_or_create_session("user_a", "sess_a")
    assert session_a.pending_lead is not None

    # User B has no pending lead
    session_b = mem_service.get_or_create_session("user_b", "sess_b")
    assert session_b.pending_lead is None

def test_lead_missing_budget_prompts_for_budget_zero_csv_writes(real_inventory, tmp_path):
    """Verify having contact and automotive need but NO budget prompts for budget with zero CSV writes."""
    db_file = str(tmp_path / "orch_nobudget.db")
    csv_file = str(tmp_path / "orch_nobudget.csv")
    pmem = PersistentMemoryService(db_path=db_file)
    lservice = LeadService(csv_path=csv_file)
    mock_interp = MagicMock()
    orch = ChatOrchestrator(
        query_interpreter=mock_interp,
        inventory_service=real_inventory,
        persistent_memory=pmem,
        lead_service=lservice
    )
    user_id = "user_nobudget"
    session_id = "sess_nobudget"

    # Turn 1: Contact + Need, NO budget
    res1 = orch.process_chat(ChatRequest(
        user_id=user_id,
        session_id=session_id,
        message="I want to submit an enquiry. My phone is +971501234567, interested in Nissan Patrol."
    ))
    assert "budget" in res1.response.lower()
    assert len(lservice.get_leads()) == 0

    # Turn 2: Provide budget
    res2 = orch.process_chat(ChatRequest(
        user_id=user_id,
        session_id=session_id,
        message="My budget is up to AED 120,000"
    ))
    assert "summary of your enquiry" in res2.response.lower()
    assert "would you like me to submit your enquiry" in res2.response.lower()
    assert len(lservice.get_leads()) == 0

    # Turn 3: Confirm
    res3 = orch.process_chat(ChatRequest(
        user_id=user_id,
        session_id=session_id,
        message="Yes please submit"
    ))
    assert "has been submitted to our sales team" in res3.response.lower()
    leads = lservice.get_leads()
    assert len(leads) == 1
    assert leads[0].phone == "+971501234567"
    assert leads[0].max_budget_aed == 120000.0

def test_normal_search_does_not_create_lead_or_csv(real_inventory, tmp_path):
    """Verify normal inventory search never creates a pending_lead or writes to leads CSV."""
    db_file = str(tmp_path / "orch_norm.db")
    csv_file = str(tmp_path / "orch_norm.csv")
    pmem = PersistentMemoryService(db_path=db_file)
    lservice = LeadService(csv_path=csv_file)
    mem_service = MemoryService()
    mock_interp = MagicMock()
    mock_interp.interpret.return_value = ParsedUserIntent(
        intent=UserIntentEnum.INVENTORY_SEARCH,
        readiness_state=SearchReadinessState.READY,
        query_filters=ParsedInventoryQuery(regional_specs="GCC", max_price_aed=150000.0)
    )
    orch = ChatOrchestrator(
        query_interpreter=mock_interp,
        inventory_service=real_inventory,
        memory_service=mem_service,
        persistent_memory=pmem,
        lead_service=lservice
    )
    user_id = "user_norm"
    session_id = "sess_norm"

    res = orch.process_chat(ChatRequest(
        user_id=user_id,
        session_id=session_id,
        message="Show me GCC cars under AED 150,000"
    ))
    assert res.total_matches > 0
    session = mem_service.get_or_create_session(user_id, session_id)
    assert session.pending_lead is None
    assert session.pending_booking is None
    # No CSV rows created
    assert len(lservice.get_leads()) == 0

def test_lead_retry_confirmation_does_not_duplicate_csv_row(real_inventory, tmp_path):
    """Verify sending repeated confirmations for a lead does not duplicate rows in the CSV."""
    csv_file = str(tmp_path / "orch_dupe_conf.csv")
    lservice = LeadService(csv_path=csv_file)

    lead = QualifiedLead(
        lead_id="LEAD-CONF01",
        created_at="2026-08-26T10:00:00Z",
        user_id="user_c1",
        session_id="sess_c1",
        name="Alice Smith",
        phone="+971501112233",
        max_budget_aed=140000.0,
        interested_make="Toyota",
    )

    # Save once
    assert lservice.save_lead(lead) is True
    # Save duplicate
    assert lservice.save_lead(lead) is True

    leads = lservice.get_leads()
    assert len(leads) == 1


