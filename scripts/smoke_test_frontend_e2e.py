"""Live end-to-end smoke test for DubizzleBot Streamlit frontend client and state workflows."""

import os
import shutil
import tempfile
import threading
import time
import uvicorn
import httpx
from datetime import datetime
from zoneinfo import ZoneInfo

from backend.main import create_app
from backend.services.inventory import InventoryService
from backend.services.memory import MemoryService
from backend.services.persistent_memory import PersistentMemoryService
from backend.services.booking import BookingService
from backend.services.lead import LeadService
from backend.services.orchestrator import ChatOrchestrator
from backend.services.query_interpreter import QueryInterpreter
from frontend.api_client import DubizzleAPIClient
from frontend.state import (
    ensure_initial_state,
    start_new_conversation,
    switch_user,
    queue_prompt,
    consume_queued_prompt,
    add_user_message,
    add_assistant_message,
)

def run_live_frontend_smoke_test():
    print("=" * 70)
    print("DUBIZZLEBOT PHASE 6: LIVE FRONTEND CLIENT & STATE VERIFICATION")
    print("=" * 70)

    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "smoke_frontend.db")
    csv_path = os.path.join(tmp_dir, "smoke_frontend_leads.csv")
    test_port = 8899
    base_url = f"http://127.0.0.1:{test_port}"
    dubai_tz = ZoneInfo("Asia/Dubai")
    frozen_now = datetime(2026, 8, 26, 10, 0, 0, tzinfo=dubai_tz)

    # 1. Setup Backend Services & FastAPI Server
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

    test_app = create_app(
        orchestrator_inst=orch,
        inventory_service_inst=inv_service,
        memory_service_inst=mem_service,
        persistent_memory_inst=pmem_service,
        booking_service_inst=booking_service,
        lead_service_inst=lead_service,
    )

    from fastapi.testclient import TestClient
    test_client = TestClient(test_app)
    client = DubizzleAPIClient(base_url="http://testserver", client=test_client)

    try:

        # ---------------------------------------------------------------------
        # HEALTH CHECK
        # ---------------------------------------------------------------------
        print("\n--- Testing Backend Health Check ---")
        assert client.health_check() is True
        print("Backend Health Status: ONLINE (200 OK)")

        # Initialize Streamlit Session State Simulation
        state = {}
        ensure_initial_state(state)

        # ---------------------------------------------------------------------
        # FLOW A: SEARCH + ORDINAL CONTEXT
        # ---------------------------------------------------------------------
        print("\n" + "=" * 50)
        print("FLOW A: SEARCH + ORDINAL CONTEXT")
        print("=" * 50)

        # User: "Show me Bentleys"
        add_user_message(state, "Show me Bentleys")
        res1 = client.send_chat(
            user_id=state["active_user_id"],
            session_id=state["session_id"],
            message="Show me Bentleys"
        )
        add_assistant_message(state, res1)
        print(f"Assistant response:\n{res1.response[:140]}...\n")
        assert res1.total_matches > 1
        assert len(res1.matched_cars) > 1
        second_car = res1.matched_cars[1]
        print(f"Verified {len(res1.matched_cars)} cars received in state. Second car: Listing #{second_car.listing_id}")

        # User: "What's the mileage on the second one?"
        add_user_message(state, "What's the mileage on the second one?")
        res2 = client.send_chat(
            user_id=state["active_user_id"],
            session_id=state["session_id"],
            message="What's the mileage on the second one?"
        )
        add_assistant_message(state, res2)
        print(f"Assistant response:\n{res2.response}\n")
        assert "318 km" in res2.response
        print("Verified ordinal reference 'second one' returned 318 km for Listing #17.")

        # ---------------------------------------------------------------------
        # FLOW B: RETURNING USER (PHASE 4B PERSISTENT MEMORY)
        # ---------------------------------------------------------------------
        print("\n" + "=" * 50)
        print("FLOW B: RETURNING USER ACROSS NEW CONVERSATION")
        print("=" * 50)

        # User: "I like the second one"
        add_user_message(state, "I like the second one")
        res3 = client.send_chat(
            user_id=state["active_user_id"],
            session_id=state["session_id"],
            message="I like the second one"
        )
        add_assistant_message(state, res3)
        print(f"Assistant response:\n{res3.response}\n")
        assert f"Listing #{second_car.listing_id}" in res3.response

        # Action: New Conversation
        old_session = state["session_id"]
        start_new_conversation(state)
        assert state["active_user_id"] == "demo_user"
        assert state["session_id"] != old_session
        assert state["messages"] == []
        print(f"Simulated 'New Conversation' click: Preserved User ID '{state['active_user_id']}', Generated New Session ID '{state['session_id'][:8]}...', Cleared Chat View.")

        # User in new session: "What cars did I like?"
        add_user_message(state, "What cars did I like?")
        res4 = client.send_chat(
            user_id=state["active_user_id"],
            session_id=state["session_id"],
            message="What cars did I like?"
        )
        add_assistant_message(state, res4)
        print(f"Assistant response:\n{res4.response}\n")
        assert res4.total_matches == 1
        assert res4.matched_cars[0].listing_id == second_car.listing_id
        print(f"Verified Phase 4B memory recall: Listing #{second_car.listing_id} successfully returned in new conversation!")

        # ---------------------------------------------------------------------
        # FLOW C: USER ISOLATION
        # ---------------------------------------------------------------------
        print("\n" + "=" * 50)
        print("FLOW C: USER ISOLATION")
        print("=" * 50)

        # Switch user to 'other_user'
        switch_user(state, "other_user")
        assert state["active_user_id"] == "other_user"
        assert state["messages"] == []
        print(f"Simulated 'Switch User' click: Switched to '{state['active_user_id']}', Generated New Session.")

        # New user asks: "What cars did I like?"
        add_user_message(state, "What cars did I like?")
        res_iso = client.send_chat(
            user_id=state["active_user_id"],
            session_id=state["session_id"],
            message="What cars did I like?"
        )
        add_assistant_message(state, res_iso)
        print(f"Assistant response:\n{res_iso.response}\n")
        assert "don't have any saved cars" in res_iso.response.lower()
        print("Verified User Isolation: Other user has zero saved cars.")

        # Switch back to 'demo_user'
        switch_user(state, "demo_user")

        # ---------------------------------------------------------------------
        # FLOW D: TEST-DRIVE BOOKING
        # ---------------------------------------------------------------------
        print("\n" + "=" * 50)
        print("FLOW D: TEST-DRIVE BOOKING FLOW")
        print("=" * 50)

        # Search Bentleys
        res_b1 = client.send_chat(user_id=state["active_user_id"], session_id=state["session_id"], message="Show me Bentleys")
        add_assistant_message(state, res_b1)

        # Book
        res_b2 = client.send_chat(
            user_id=state["active_user_id"],
            session_id=state["session_id"],
            message="I want to test drive the second one Saturday at 3 PM"
        )
        add_assistant_message(state, res_b2)
        print(f"Assistant response:\n{res_b2.response}\n")
        assert "would you like me to confirm this test drive?" in res_b2.response.lower()

        # Confirm
        res_b3 = client.send_chat(
            user_id=state["active_user_id"],
            session_id=state["session_id"],
            message="Confirm"
        )
        add_assistant_message(state, res_b3)
        print(f"Assistant response:\n{res_b3.response}\n")
        assert "has been confirmed" in res_b3.response.lower()
        print("Verified Test-Drive Booking completed and confirmed.")

        # ---------------------------------------------------------------------
        # FLOW E: LEAD QUALIFICATION
        # ---------------------------------------------------------------------
        print("\n" + "=" * 50)
        print("FLOW E: LEAD QUALIFICATION FLOW")
        print("=" * 50)

        start_new_conversation(state)

        # Turn 1: Inquiry with requirements
        res_l1 = client.send_chat(
            user_id=state["active_user_id"],
            session_id=state["session_id"],
            message="I'd like someone to contact me about buying a GCC SUV."
        )
        add_assistant_message(state, res_l1)
        print(f"Assistant response:\n{res_l1.response}\n")
        assert "budget" in res_l1.response.lower()

        # Turn 2: Provide budget & contact
        res_l2 = client.send_chat(
            user_id=state["active_user_id"],
            session_id=state["session_id"],
            message="My budget is up to AED 140,000, phone is +971501112233"
        )
        add_assistant_message(state, res_l2)
        print(f"Assistant response:\n{res_l2.response}\n")
        assert "summary of your enquiry" in res_l2.response.lower()

        # Turn 3: Confirm
        res_l3 = client.send_chat(
            user_id=state["active_user_id"],
            session_id=state["session_id"],
            message="Yes please submit"
        )
        add_assistant_message(state, res_l3)
        print(f"Assistant response:\n{res_l3.response}\n")
        assert "has been submitted to our sales team" in res_l3.response.lower()
        print("Verified Lead Qualification submitted.")

        # ---------------------------------------------------------------------
        # FLOW F: DOMAIN GUARDRAIL
        # ---------------------------------------------------------------------
        print("\n" + "=" * 50)
        print("FLOW F: DOMAIN GUARDRAIL REDIRECTION")
        print("=" * 50)

        res_g = client.send_chat(
            user_id=state["active_user_id"],
            session_id=state["session_id"],
            message="Write me Python code"
        )
        print(f"Assistant response:\n{res_g.response}\n")
        assert "specialized in helping you find cars" in res_g.response or "non-automotive" in res_g.response
        print("Verified Guardrail: Non-automotive request politely redirected.")

        # ---------------------------------------------------------------------
        # FLOW G: LARGE RESULTS
        # ---------------------------------------------------------------------
        print("\n" + "=" * 50)
        print("FLOW G: LARGE RESULTS HANDLING")
        print("=" * 50)

        start_new_conversation(state)
        res_large = client.send_chat(
            user_id=state["active_user_id"],
            session_id=state["session_id"],
            message="Show me GCC cars"
        )
        add_assistant_message(state, res_large)
        print(f"Total matches: {res_large.total_matches}, Matched cars in array: {len(res_large.matched_cars)}")
        assert res_large.total_matches > 10
        assert len(res_large.matched_cars) > 10
        print(f"Verified complete matched_cars list ({len(res_large.matched_cars)} items) retained in frontend state.")

        # ---------------------------------------------------------------------
        # FLOW H: BACKEND OFFLINE RECOVERY
        # ---------------------------------------------------------------------
        print("\n" + "=" * 50)
        print("FLOW H: BACKEND OFFLINE HANDLING")
        print("=" * 50)

        # Test against a non-existent / unreachable port
        offline_client = DubizzleAPIClient(base_url="http://127.0.0.1:9999")
        assert offline_client.health_check() is False
        print("Offline health check correctly returned False.")

        try:
            offline_client.send_chat(user_id="u1", session_id="s1", message="hello")
            assert False, "Should have raised DubizzleAPIError"
        except Exception as e:
            from frontend.api_client import DubizzleAPIError
            assert isinstance(e, DubizzleAPIError)
            assert "could not connect" in e.message.lower()
            print(f"Verified offline client error message: '{e.message}'")

        print("\n" + "=" * 70)
        print("PHASE 6 LIVE FRONTEND CLIENT & STATE VERIFICATION PASSED")
        print("=" * 70)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    run_live_frontend_smoke_test()
