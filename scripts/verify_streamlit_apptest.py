"""Comprehensive Live AppTest script for DubizzleBot Streamlit application.
Runs Streamlit AppTest simulating real browser user interactions against FastAPI server.
"""

import os
import shutil
import tempfile
import threading
import time
import uvicorn
from datetime import datetime
from zoneinfo import ZoneInfo
from unittest.mock import MagicMock
from streamlit.testing.v1 import AppTest

from backend.main import create_app
from backend.services.inventory import InventoryService
from backend.services.memory import MemoryService
from backend.services.persistent_memory import PersistentMemoryService
from backend.services.booking import BookingService
from backend.services.lead import LeadService
from backend.services.orchestrator import ChatOrchestrator
from backend.services.query_interpreter import QueryInterpreter
from backend.models.intent import ParsedUserIntent, UserIntentEnum, ParsedInventoryQuery, RegionalSpecEnum, SearchReadinessState

def make_deterministic_interpreter() -> QueryInterpreter:
    mock_qi = MagicMock(spec=QueryInterpreter)
    def mock_interpret(prompt: str) -> ParsedUserIntent:
        p_lower = prompt.lower()
        if "bentley" in p_lower:
            return ParsedUserIntent(
                intent=UserIntentEnum.INVENTORY_SEARCH,
                query_filters=ParsedInventoryQuery(make="Bentley"),
                readiness_state=SearchReadinessState.READY,
            )
        elif "gcc" in p_lower:
            return ParsedUserIntent(
                intent=UserIntentEnum.INVENTORY_SEARCH,
                query_filters=ParsedInventoryQuery(regional_specs=RegionalSpecEnum.GCC),
                readiness_state=SearchReadinessState.READY,
            )
        elif "python" in p_lower:
            return ParsedUserIntent(
                intent=UserIntentEnum.UNKNOWN,
                query_filters=None,
                readiness_state=SearchReadinessState.NON_INVENTORY_INTENT,
            )
        return ParsedUserIntent(
            intent=UserIntentEnum.GENERAL_CHAT,
            query_filters=None,
            readiness_state=SearchReadinessState.NON_INVENTORY_INTENT,
        )
    mock_qi.interpret.side_effect = mock_interpret
    return mock_qi

def verify_streamlit_live_apptest():
    print("=" * 70)
    print("DUBIZZLEBOT PHASE 6: REAL STREAMLIT APPTEST BROWSER SIMULATION")
    print("=" * 70)

    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test_apptest.db")
    csv_path = os.path.join(tmp_dir, "test_apptest_leads.csv")
    test_port = 8999
    base_url = f"http://127.0.0.1:{test_port}"
    dubai_tz = ZoneInfo("Asia/Dubai")
    frozen_now = datetime(2026, 8, 26, 10, 0, 0, tzinfo=dubai_tz)

    inv_service = InventoryService()
    mem_service = MemoryService()
    pmem_service = PersistentMemoryService(db_path=db_path)
    booking_service = BookingService(db_path=db_path, persistent_memory=pmem_service)
    lead_service = LeadService(csv_path=csv_path)
    interpreter = make_deterministic_interpreter()

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

    server_config = uvicorn.Config(app=test_app, host="127.0.0.1", port=test_port, log_level="error")
    server = uvicorn.Server(server_config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    time.sleep(1.0)

    os.environ["BACKEND_URL"] = base_url

    try:
        # 1. Initialize Streamlit App
        app_path = os.path.abspath("frontend/app.py")
        at = AppTest.from_file(app_path, default_timeout=30.0)
        at.run()
        assert not at.exception, f"App threw exception on startup: {at.exception}"

        # -------------------------------------------------------------------------
        # 2. VERIFY EMPTY STATE & SIDEBAR
        # -------------------------------------------------------------------------
        print("\n[Step 1] Verifying Empty State & Sidebar Layout...")
        assert any("DubizzleBot" in title.value for title in at.title)
        print("  Header & title verified.")

        user_inputs = [inp for inp in at.sidebar.text_input if "User Identity" in inp.label]
        assert len(user_inputs) > 0
        assert at.session_state["active_user_id"] == "demo_user"
        print(f"  Active User ID in sidebar: {at.session_state['active_user_id']}")

        new_conv_btns = [b for b in at.sidebar.button if "New Conversation" in b.label]
        assert len(new_conv_btns) > 0
        print("  'New Conversation' button present.")

        success_msgs = [s.value for s in at.sidebar.success]
        assert any("Connected to Backend" in s for s in success_msgs)
        print("  🟢 Backend health indicator: Connected to Backend")

        # -------------------------------------------------------------------------
        # 3. REAL SEARCH & VEHICLE CARD RENDERING
        # -------------------------------------------------------------------------
        print("\n[Step 2] Testing 'Show me Bentleys' search submission...")
        at.chat_input[0].set_value("Show me Bentleys").run()
        assert not at.exception

        assert len(at.chat_message) >= 2
        user_turn = at.chat_message[0]
        assistant_turn = at.chat_message[1]
        assert "Show me Bentleys" in user_turn.markdown[0].value
        assert "7" in assistant_turn.markdown[0].value or "bentley" in assistant_turn.markdown[0].value.lower()
        print("  Assistant response prose received.")

        last_msg = at.session_state["messages"][-1]
        assert last_msg["matched_cars"] is not None
        assert len(last_msg["matched_cars"]) == 7
        print(f"  Verified exactly 7 matched cars retained in state. 2nd car: Listing #{last_msg['matched_cars'][1]['listing_id']}")
        assert last_msg["matched_cars"][1]["listing_id"] == 17

        # -------------------------------------------------------------------------
        # 4. REAL ORDINAL FOLLOW-UP
        # -------------------------------------------------------------------------
        print("\n[Step 3] Testing ordinal follow-up 'What is the mileage on the second one?'...")
        at.chat_input[0].set_value("What is the mileage on the second one?").run()
        assert not at.exception

        last_assistant_msg = at.chat_message[-1]
        response_text = last_assistant_msg.markdown[0].value
        print(f"  Assistant response: {response_text}")
        assert "318" in response_text
        assert "17" in response_text
        print("  Verified grounded mileage (318 km) for Listing #17.")

        # -------------------------------------------------------------------------
        # 5. RETURNING USER PERSISTENT MEMORY & NEW CONVERSATION BUTTON
        # -------------------------------------------------------------------------
        print("\n[Step 4] Testing persistent memory saving and 'New Conversation' button...")
        at.chat_input[0].set_value("I like the second one").run()
        assert not at.exception
        save_response = at.chat_message[-1].markdown[0].value
        print(f"  Save response: {save_response}")
        assert "17" in save_response or "favorites" in save_response.lower()

        # Click the actual sidebar "New Conversation" button
        print("  Clicking '➕ New Conversation' button...")
        old_session_id = at.session_state["session_id"]
        new_conv_btn = [b for b in at.sidebar.button if "New Conversation" in b.label][0]
        new_conv_btn.click().run()
        assert not at.exception

        assert at.session_state["active_user_id"] == "demo_user"
        assert at.session_state["session_id"] != old_session_id
        assert len(at.session_state["messages"]) == 0
        print(f"  Verified User ID preserved ({at.session_state['active_user_id']}), new session generated, chat cleared.")

        print("  Asking 'What cars did I like?' in new conversation...")
        at.chat_input[0].set_value("What cars did I like?").run()
        assert not at.exception
        recall_response = at.chat_message[-1].markdown[0].value
        print(f"  Recall response: {recall_response}")
        assert "17" in recall_response or "bentley" in recall_response.lower()
        print("  Verified Listing #17 successfully recalled from persistent memory across sessions!")

        # -------------------------------------------------------------------------
        # 6. SWITCH USER BUTTON
        # -------------------------------------------------------------------------
        print("\n[Step 5] Testing 'Switch User' button in sidebar...")
        user_inp = [inp for inp in at.sidebar.text_input if "User Identity" in inp.label][0]
        user_inp.set_value("phase6_other_user")
        switch_btn = [b for b in at.sidebar.button if "Switch User" in b.label][0]
        switch_btn.click().run()
        assert not at.exception

        assert at.session_state["active_user_id"] == "phase6_other_user"
        assert len(at.session_state["messages"]) == 0
        print(f"  Switched user to '{at.session_state['active_user_id']}', chat cleared.")

        at.chat_input[0].set_value("What cars did I like?").run()
        assert not at.exception
        other_recall = at.chat_message[-1].markdown[0].value
        print(f"  Other user recall response: {other_recall}")
        assert "don't have any saved cars" in other_recall.lower()
        print("  Verified User Isolation: New user has zero saved cars.")

        # Switch back to demo_user
        user_inp.set_value("demo_user")
        switch_btn.click().run()
        assert at.session_state["active_user_id"] == "demo_user"

        # -------------------------------------------------------------------------
        # 7. LARGE RESULT HANDLING (45 CARS)
        # -------------------------------------------------------------------------
        print("\n[Step 6] Testing large result set 'Show me GCC cars'...")
        new_conv_btn.click().run()
        at.chat_input[0].set_value("Show me GCC cars").run()
        assert not at.exception

        last_msg = at.session_state["messages"][-1]
        assert last_msg["total_matches"] == 45
        assert len(last_msg["matched_cars"]) == 45
        print(f"  Total matched cars: {last_msg['total_matches']}, Retained in state: {len(last_msg['matched_cars'])}")
        print("  Verified complete 45-car dataset retained in frontend message history.")

        # -------------------------------------------------------------------------
        # 8. TEST-DRIVE BOOKING FLOW
        # -------------------------------------------------------------------------
        print("\n[Step 7] Testing conversational test-drive booking...")
        new_conv_btn.click().run()
        at.chat_input[0].set_value("Show me Bentleys").run()
        at.chat_input[0].set_value("I want to test drive the second one Saturday at 3 PM").run()
        assert not at.exception
        booking_prompt = at.chat_message[-1].markdown[0].value
        print(f"  Booking summary: {booking_prompt}")
        assert "confirm" in booking_prompt.lower()

        at.chat_input[0].set_value("Confirm").run()
        assert not at.exception
        booking_conf = at.chat_message[-1].markdown[0].value
        print(f"  Booking confirmation: {booking_conf}")
        assert "confirmed" in booking_conf.lower()
        assert "#BK-" in booking_conf
        print("  Verified test-drive booking confirmed.")

        # -------------------------------------------------------------------------
        # 9. LEAD QUALIFICATION FLOW
        # -------------------------------------------------------------------------
        print("\n[Step 8] Testing lead qualification enquiry...")
        new_conv_btn.click().run()
        at.chat_input[0].set_value("I'd like someone to contact me about buying a GCC SUV").run()
        assert not at.exception

        at.chat_input[0].set_value("My budget is up to AED 140,000, phone is +971501112233").run()
        assert not at.exception

        at.chat_input[0].set_value("Yes please submit").run()
        assert not at.exception
        lead_conf = at.chat_message[-1].markdown[0].value
        print(f"  Lead confirmation: {lead_conf}")
        assert "submitted" in lead_conf.lower()
        assert "#LEAD-" in lead_conf
        print("  Verified lead qualification submitted.")

        # -------------------------------------------------------------------------
        # 10. DOMAIN GUARDRAIL FLOW
        # -------------------------------------------------------------------------
        print("\n[Step 9] Testing non-automotive domain guardrail...")
        at.chat_input[0].set_value("Write me Python code").run()
        assert not at.exception
        guardrail_resp = at.chat_message[-1].markdown[0].value
        print(f"  Guardrail response: {guardrail_resp}")
        assert "specialized in helping you find cars" in guardrail_resp or "non-automotive" in guardrail_resp
        print("  Verified domain guardrail redirection.")

        print("\n" + "=" * 70)
        print("REAL STREAMLIT APPTEST VERIFICATION PASSED ALL SCENARIOS")
        print("=" * 70)

    finally:
        server.should_exit = True
        server_thread.join(timeout=2.0)
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    verify_streamlit_live_apptest()
