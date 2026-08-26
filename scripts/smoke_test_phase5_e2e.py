"""Live Smoke Test for Phase 5: Lead Qualification & Test-Drive / Viewing Bookings."""

import os
import shutil
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo

from backend.config import settings
from backend.models.chat import ChatRequest
from backend.services.inventory import InventoryService
from backend.services.memory import MemoryService
from backend.services.persistent_memory import PersistentMemoryService
from backend.services.booking import BookingService
from backend.services.lead import LeadService
from backend.services.orchestrator import ChatOrchestrator
from backend.services.query_interpreter import QueryInterpreter

def run_smoke_test():
    print("=" * 70)
    print("DUBIZZLEBOT PHASE 5: LIVE END-TO-END VERIFICATION")
    print("=" * 70)

    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "smoke_test_phase5.db")
    csv_path = os.path.join(tmp_dir, "smoke_test_leads.csv")
    dubai_tz = ZoneInfo("Asia/Dubai")

    # Reference clock: Wednesday, August 26, 2026 at 10:00 AM Asia/Dubai
    frozen_now = datetime(2026, 8, 26, 10, 0, 0, tzinfo=dubai_tz)

    print(f"Temporary SQLite Database: {db_path}")
    print(f"Temporary Leads CSV: {csv_path}")
    print(f"Timezone: {settings.BOOKING_TIMEZONE}")

    try:
        inv_service = InventoryService()
        mem_service = MemoryService()
        pmem_service = PersistentMemoryService(db_path=db_path)
        booking_service = BookingService(db_path=db_path, persistent_memory=pmem_service)
        lead_service = LeadService(csv_path=csv_path)
        interpreter = QueryInterpreter()

        orch = ChatOrchestrator(
            query_interpreter=interpreter,
            inventory_service=inv_service,
            memory_service=mem_service,
            persistent_memory=pmem_service,
            booking_service=booking_service,
            lead_service=lead_service,
            current_time_override=frozen_now,
        )

        user_id = "smoke_user_phase5"
        session_id = "smoke_session_phase5"

        # =====================================================================
        # FLOW A: DESCRIPTIVE VIEWING REGRESSION & TEST-DRIVE MULTI-TURN
        # =====================================================================
        print("\n" + "=" * 50)
        print("FLOW A: DESCRIPTIVE VIEWING CANDIDATES & MULTI-TURN BOOKING")
        print("=" * 50)

        # Turn 1: Fresh session, descriptive viewing request (Live Gemini)
        print(f"\n--- Turn 1: 'I want to test drive a Bentley' (Live Gemini: {settings.LLM_MODEL}) ---")
        req1 = ChatRequest(user_id=user_id, message="I want to test drive a Bentley", session_id=session_id)
        res1 = orch.process_chat(req1)
        print(f"Response (truncated):\n{res1.response[:180]}...\n")
        assert res1.total_matches > 1
        assert "matching candidate vehicles in our inventory" in res1.response.lower()
        second_car = res1.matched_cars[1]
        print(f"Verified candidate result set populated. Second car: Listing #{second_car.listing_id} ({second_car.year} {second_car.make} {second_car.model})")

        # Turn 2: "I want to test drive the second one."
        print("\n--- Turn 2: 'I want to test drive the second one.' ---")
        req2 = ChatRequest(user_id=user_id, message="I want to test drive the second one.", session_id=session_id)
        res2 = orch.process_chat(req2)
        print(f"Response:\n{res2.response}\n")
        assert "what date and time" in res2.response.lower()

        # Turn 3: "Saturday at 3 PM"
        print("\n--- Turn 3: 'Saturday at 3 PM' ---")
        req3 = ChatRequest(user_id=user_id, message="Saturday at 3 PM", session_id=session_id)
        res3 = orch.process_chat(req3)
        print(f"Response:\n{res3.response}\n")
        assert "would you like me to confirm this test drive?" in res3.response.lower()

        # Verify not yet in SQLite
        assert len(booking_service.get_user_bookings(user_id)) == 0

        # Turn 4: "Yes please confirm"
        print("\n--- Turn 4: 'Yes please confirm' ---")
        req4 = ChatRequest(user_id=user_id, message="Yes please confirm", session_id=session_id)
        res4 = orch.process_chat(req4)
        print(f"Response:\n{res4.response}\n")
        assert "has been confirmed" in res4.response.lower()
        assert "booking ref:" in res4.response.lower()

        # Verify persisted in SQLite
        saved_bookings = booking_service.get_user_bookings(user_id)
        assert len(saved_bookings) == 1
        assert saved_bookings[0].listing_id == second_car.listing_id
        print(f"Verified booking saved in SQLite: Ref #{saved_bookings[0].booking_id}, Listing #{saved_bookings[0].listing_id}")

        # =====================================================================
        # FLOW B: COMBINED ONE-TURN BOOKING REQUEST
        # =====================================================================
        print("\n" + "=" * 50)
        print("FLOW B: COMBINED ONE-TURN BOOKING REQUEST")
        print("=" * 50)

        user_ot = "smoke_user_oneturn"
        sess_ot = "smoke_sess_oneturn"

        orch.process_chat(ChatRequest(user_id=user_ot, message="Show me Bentleys", session_id=sess_ot))
        print("\n--- Turn 5: 'I want to test drive the second one Saturday at 3 PM' ---")
        res_ot1 = orch.process_chat(ChatRequest(user_id=user_ot, message="I want to test drive the second one Saturday at 3 PM", session_id=sess_ot))
        print(f"Response:\n{res_ot1.response}\n")
        assert "would you like me to confirm this test drive?" in res_ot1.response.lower()
        assert len(booking_service.get_user_bookings(user_ot)) == 0

        res_ot2 = orch.process_chat(ChatRequest(user_id=user_ot, message="Yes, confirm", session_id=sess_ot))
        assert "has been confirmed" in res_ot2.response.lower()
        assert len(booking_service.get_user_bookings(user_ot)) == 1
        print("Verified combined one-turn booking parsed, validated, summarized, and confirmed.")

        # =====================================================================
        # FLOW C: INVALID SLOT RECOVERY
        # =====================================================================
        print("\n" + "=" * 50)
        print("FLOW C: INVALID SLOT RECOVERY")
        print("=" * 50)

        user_rec = "smoke_user_recovery"
        sess_rec = "smoke_sess_recovery"

        orch.process_chat(ChatRequest(user_id=user_rec, message="Show me Bentleys", session_id=sess_rec))
        print("\n--- Turn 6: 'I want to test drive the first one Sunday at 3 PM' ---")
        res_rec1 = orch.process_chat(ChatRequest(user_id=user_rec, message="I want to test drive the first one Sunday at 3 PM", session_id=sess_rec))
        print(f"Response:\n{res_rec1.response}\n")
        assert "closed on sundays" in res_rec1.response.lower()
        assert len(booking_service.get_user_bookings(user_rec)) == 0

        print("--- Turn 7: 'Saturday at 3 PM' ---")
        res_rec2 = orch.process_chat(ChatRequest(user_id=user_rec, message="Saturday at 3 PM", session_id=sess_rec))
        print(f"Response:\n{res_rec2.response}\n")
        assert "would you like me to confirm this test drive?" in res_rec2.response.lower()

        res_rec3 = orch.process_chat(ChatRequest(user_id=user_rec, message="Confirm", session_id=sess_rec))
        assert "has been confirmed" in res_rec3.response.lower()
        assert len(booking_service.get_user_bookings(user_rec)) == 1
        print("Verified invalid Sunday rejection recovered without re-specifying vehicle.")

        # =====================================================================
        # FLOW D: PENDING WORKFLOW INTERRUPTION
        # =====================================================================
        print("\n" + "=" * 50)
        print("FLOW D: PENDING WORKFLOW INTERRUPTION")
        print("=" * 50)

        user_int = "smoke_user_interrup"
        sess_int = "smoke_sess_interrup"

        orch.process_chat(ChatRequest(user_id=user_int, message="Show me Bentleys", session_id=sess_int))
        orch.process_chat(ChatRequest(user_id=user_int, message="I want to test drive the second one Saturday at 3 PM", session_id=sess_int))

        print("\n--- Turn 8: 'What's its mileage?' ---")
        res_int1 = orch.process_chat(ChatRequest(user_id=user_int, message="What's its mileage?", session_id=sess_int))
        print(f"Response:\n{res_int1.response}\n")
        assert "318 km" in res_int1.response
        assert len(booking_service.get_user_bookings(user_int)) == 0

        print("--- Turn 9: 'What cars did I like?' ---")
        res_int2 = orch.process_chat(ChatRequest(user_id=user_int, message="What cars did I like?", session_id=sess_int))
        print(f"Response:\n{res_int2.response}\n")
        assert "saved cars" in res_int2.response.lower() or "favorites" in res_int2.response.lower()
        assert len(booking_service.get_user_bookings(user_int)) == 0

        print("--- Turn 10: 'Confirm the test drive' ---")
        res_int3 = orch.process_chat(ChatRequest(user_id=user_int, message="Confirm the test drive", session_id=sess_int))
        print(f"Response:\n{res_int3.response}\n")
        assert "has been confirmed" in res_int3.response.lower()
        assert len(booking_service.get_user_bookings(user_int)) == 1
        print("Verified pending booking draft survived unrelated Phase 4A & 4B queries and confirmed.")

        # =====================================================================
        # FLOW E: LEAD QUALIFICATION & CSV CAPTURE
        # =====================================================================
        print("\n" + "=" * 50)
        print("FLOW E: LEAD QUALIFICATION & CSV CAPTURE")
        print("=" * 50)

        user_lead = "smoke_user_lead"
        sess_lead = "smoke_sess_lead"

        # Turn 11: Contact + Need, NO budget
        print("\n--- Turn 11: 'I want to submit an enquiry. My phone is +971501234567, interested in Nissan Patrol.' ---")
        res_l1 = orch.process_chat(ChatRequest(user_id=user_lead, message="I want to submit an enquiry. My phone is +971501234567, interested in Nissan Patrol.", session_id=sess_lead))
        print(f"Response:\n{res_l1.response}\n")
        assert "budget" in res_l1.response.lower()
        assert len(lead_service.get_leads()) == 0

        # Turn 12: Supply budget
        print("--- Turn 12: 'My budget is up to AED 120,000' ---")
        res_l2 = orch.process_chat(ChatRequest(user_id=user_lead, message="My budget is up to AED 120,000", session_id=sess_lead))
        print(f"Response:\n{res_l2.response}\n")
        assert "summary of your enquiry" in res_l2.response.lower()
        assert "would you like me to submit your enquiry" in res_l2.response.lower()
        assert len(lead_service.get_leads()) == 0

        # Turn 13: Confirm
        print("--- Turn 13: 'Yes please submit' ---")
        res_l3 = orch.process_chat(ChatRequest(user_id=user_lead, message="Yes please submit", session_id=sess_lead))
        print(f"Response:\n{res_l3.response}\n")
        assert "has been submitted to our sales team" in res_l3.response.lower()

        leads = lead_service.get_leads()
        assert len(leads) == 1
        assert leads[0].phone == "+971501234567"
        assert leads[0].max_budget_aed == 120000.0
        print(f"Verified qualified lead in CSV: Ref #{leads[0].lead_id}, Phone: {leads[0].phone}, Budget: AED {leads[0].max_budget_aed:,.0f}")

        # Duplicate confirmation retry
        res_l4 = orch.process_chat(ChatRequest(user_id=user_lead, message="Yes submit again", session_id=sess_lead))
        assert len(lead_service.get_leads()) == 1
        print("Verified duplicate submission did not create duplicate CSV rows.")

        # =====================================================================
        # FLOW F: NORMAL SEARCH NON-LEAD VERIFICATION
        # =====================================================================
        print("\n" + "=" * 50)
        print("FLOW F: NORMAL SEARCH NON-LEAD VERIFICATION")
        print("=" * 50)

        user_norm = "smoke_user_norm"
        sess_norm = "smoke_sess_norm"

        res_norm = orch.process_chat(ChatRequest(user_id=user_norm, message="Show me GCC cars under AED 150,000", session_id=sess_norm))
        assert res_norm.total_matches > 0
        norm_sess = mem_service.get_or_create_session(user_norm, sess_norm)
        assert norm_sess.pending_lead is None
        assert norm_sess.pending_booking is None
        print("Verified normal search did NOT create pending lead or write to CSV.")

        # =====================================================================
        # FLOW G: FIRST-TIME USER PROFILE AUTO-CREATION ON BOOKING
        # =====================================================================
        print("\n" + "=" * 50)
        print("FLOW G: FIRST-TIME USER PROFILE AUTO-CREATION ON BOOKING")
        print("=" * 50)

        brand_new_user = "brand_new_user_999"
        brand_new_sess = "brand_new_sess_999"

        # Check profile does NOT exist yet
        assert pmem_service.get_profile(brand_new_user) is None

        orch.process_chat(ChatRequest(user_id=brand_new_user, message="Show me Bentleys", session_id=brand_new_sess))
        orch.process_chat(ChatRequest(user_id=brand_new_user, message="I want to test drive the first one Saturday at 11 AM", session_id=brand_new_sess))
        orch.process_chat(ChatRequest(user_id=brand_new_user, message="Confirm", session_id=brand_new_sess))

        # Profile was auto-created and booking saved
        assert pmem_service.get_profile(brand_new_user) is not None
        assert len(booking_service.get_user_bookings(brand_new_user)) == 1
        print(f"Verified brand-new user profile auto-created and booking saved in SQLite.")

        print("\n" + "=" * 70)
        print("PHASE 5 PRE-FREEZE LIVE VERIFICATION PASSED")
        print("=" * 70)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    run_smoke_test()
