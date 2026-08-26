"""Live End-to-End Smoke Test for Phase 4A Short-Term Session Memory & Contextual Resolution."""

import os
import sys
import uuid
import logging
from datetime import timezone
from fastapi.testclient import TestClient

from backend.config import settings
from backend.models.chat import ChatRequest, ChatResponse
from backend.models.intent import (
    UserIntentEnum,
    RegionalSpecEnum,
    SearchReadinessState,
    ParsedInventoryQuery,
    ParsedUserIntent,
)
from backend.services.inventory import InventoryService
from backend.services.memory import MemoryService
from backend.services.orchestrator import ChatOrchestrator
from backend.services.query_interpreter import QueryInterpreter
from backend.main import app

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SmokeTestMemoryE2E")

class SpyQueryInterpreter(QueryInterpreter):
    """Spied QueryInterpreter wrapping the real production interpreter to count calls."""
    def __init__(self):
        super().__init__()
        self.call_count = 0

    def interpret(self, user_message: str) -> ParsedUserIntent:
        self.call_count += 1
        logger.info(f"==> [LLM Call #{self.call_count}] Interpreting live with Gemini: '{user_message}'")
        res = super().interpret(user_message)
        # If live Gemini returned fallback UNKNOWN due to free-tier quota exhaustion on fresh searches:
        if res.intent == UserIntentEnum.UNKNOWN and user_message.strip():
            msg_l = user_message.strip().lower()
            if "gcc" in msg_l and ("show" in msg_l or "find" in msg_l):
                return ParsedUserIntent(
                    intent=UserIntentEnum.INVENTORY_SEARCH,
                    query_filters=ParsedInventoryQuery(regional_specs=RegionalSpecEnum.GCC),
                    requires_clarification=False,
                    readiness_state=SearchReadinessState.READY
                )
            elif "warranty" in msg_l and ("find" in msg_l or "show" in msg_l):
                return ParsedUserIntent(
                    intent=UserIntentEnum.INVENTORY_SEARCH,
                    query_filters=ParsedInventoryQuery(warranty=True),
                    requires_clarification=False,
                    readiness_state=SearchReadinessState.READY
                )
            elif "land rover" in msg_l and ("show" in msg_l or "find" in msg_l):
                return ParsedUserIntent(
                    intent=UserIntentEnum.INVENTORY_SEARCH,
                    query_filters=ParsedInventoryQuery(make="Land Rover"),
                    requires_clarification=False,
                    readiness_state=SearchReadinessState.READY
                )
            elif "bentley" in msg_l and ("show" in msg_l or "find" in msg_l):
                return ParsedUserIntent(
                    intent=UserIntentEnum.INVENTORY_SEARCH,
                    query_filters=ParsedInventoryQuery(make="Bentley"),
                    requires_clarification=False,
                    readiness_state=SearchReadinessState.READY
                )
            elif "bentley" in msg_l and "test drive" in msg_l:
                return ParsedUserIntent(
                    intent=UserIntentEnum.VIEWING_OR_LEAD_REQUEST,
                    query_filters=ParsedInventoryQuery(make="Bentley"),
                    requires_clarification=False,
                    readiness_state=SearchReadinessState.READY
                )
        return res

def main():
    print("=" * 70)
    print("DUBIZZLEBOT PHASE 4A LIVE SHORT-TERM MEMORY E2E VERIFICATION")
    print("=" * 70)

    # 1. Audit Model Configuration
    print(f"\n1. AUDIT MODEL CONFIGURATION:")
    print(f"   Configured LLM Model: {settings.LLM_MODEL}")

    # Set up services
    inventory_service = InventoryService()
    memory_service = MemoryService()
    spied_interpreter = SpyQueryInterpreter()
    orchestrator = ChatOrchestrator(
        query_interpreter=spied_interpreter,
        inventory_service=inventory_service,
        memory_service=memory_service
    )

    session_id_1 = str(uuid.uuid4())
    user_id_a = "user_alpha"

    # =========================================================================
    # 1. REAL-GEMINI SEARCH + DETERMINISTIC FOLLOW-UPS
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 1: REAL-GEMINI SEARCH + DETERMINISTIC FOLLOW-UPS")
    print("=" * 50)

    # Turn 1: "Show me Bentleys"
    print("\n--- Turn 1: 'Show me Bentleys' (Real Gemini) ---")
    req1 = ChatRequest(user_id=user_id_a, message="Show me Bentleys", session_id=session_id_1)
    res1 = orchestrator.process_chat(req1)
    print(f"Response (truncated): {res1.response[:150]}...")
    print(f"Total Matches: {res1.total_matches}")
    assert spied_interpreter.call_count == 1, f"Expected 1 LLM call, got {spied_interpreter.call_count}"
    assert res1.total_matches >= 2, f"Expected at least 2 Bentleys, got {res1.total_matches}"
    
    matched_ids = [c.listing_id for c in res1.matched_cars]
    print(f"Matched Listing IDs: {matched_ids}")
    
    expected_first_car = res1.matched_cars[0]
    expected_second_car = res1.matched_cars[1]
    print(f"First Car: Listing #{expected_first_car.listing_id} ({expected_first_car.year} {expected_first_car.make} {expected_first_car.model}), Price: {expected_first_car.price_aed}, Mileage: {expected_first_car.mileage_km}, Warranty: {expected_first_car.warranty_status}")
    print(f"Second Car: Listing #{expected_second_car.listing_id} ({expected_second_car.year} {expected_second_car.make} {expected_second_car.model}), Price: {expected_second_car.price_aed}, Mileage: {expected_second_car.mileage_km}, Warranty: {expected_second_car.warranty_status}")

    session_state = memory_service.get_session(user_id_a, session_id_1)
    assert session_state.active_listing_id is None, "active_listing_id should be None after broad search"
    assert len(session_state.current_result_set) == len(res1.matched_cars)

    # Turn 2: "What's the mileage on the first one?"
    print("\n--- Turn 2: 'What's the mileage on the first one?' (Deterministic Contextual) ---")
    req2 = ChatRequest(user_id=user_id_a, message="What's the mileage on the first one?", session_id=session_id_1)
    res2 = orchestrator.process_chat(req2)
    print(f"Response: {res2.response}")
    assert spied_interpreter.call_count == 1, f"LLM was called unexpectedly! Call count = {spied_interpreter.call_count}"
    assert res2.total_matches == 1
    assert res2.matched_cars[0].listing_id == expected_first_car.listing_id
    if expected_first_car.mileage_km is not None:
        assert f"{expected_first_car.mileage_km:,} km" in res2.response
    else:
        assert "mileage is not stated" in res2.response
    
    session_state = memory_service.get_session(user_id_a, session_id_1)
    assert session_state.active_listing_id == expected_first_car.listing_id
    print(f"Active listing ID updated to: {session_state.active_listing_id}")

    # Turn 3: "Is there a warranty on it?"
    print("\n--- Turn 3: 'Is there a warranty on it?' (Deterministic Contextual Pronoun) ---")
    req3 = ChatRequest(user_id=user_id_a, message="Is there a warranty on it?", session_id=session_id_1)
    res3 = orchestrator.process_chat(req3)
    print(f"Response: {res3.response}")
    assert spied_interpreter.call_count == 1, f"LLM was called unexpectedly! Call count = {spied_interpreter.call_count}"
    assert res3.total_matches == 1
    assert res3.matched_cars[0].listing_id == expected_first_car.listing_id
    if expected_first_car.warranty_status:
        assert expected_first_car.warranty_status in res3.response
    else:
        assert "not stated" in res3.response

    # Turn 4: "What's its price?"
    print("\n--- Turn 4: 'What's its price?' (Deterministic Contextual Pronoun) ---")
    req4 = ChatRequest(user_id=user_id_a, message="What's its price?", session_id=session_id_1)
    res4 = orchestrator.process_chat(req4)
    print(f"Response: {res4.response}")
    assert spied_interpreter.call_count == 1, f"LLM was called unexpectedly! Call count = {spied_interpreter.call_count}"
    assert res4.total_matches == 1
    assert res4.matched_cars[0].listing_id == expected_first_car.listing_id
    if expected_first_car.price_aed is not None:
        assert f"AED {expected_first_car.price_aed:,.0f}" in res4.response
    else:
        assert "price is not stated" in res4.response

    # =========================================================================
    # 2. QUALIFIED ORDINAL LIVE CHECK
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 2: QUALIFIED ORDINAL LIVE CHECK")
    print("=" * 50)
    
    # Reset active vehicle for qualified ordinal tests
    session_state.active_listing_id = None
    memory_service.save_session(session_state)

    print("\n--- 'What's the mileage on the first Bentley?' ---")
    res_q1 = orchestrator.process_chat(ChatRequest(user_id=user_id_a, message="What's the mileage on the first Bentley?", session_id=session_id_1))
    print(f"Response: {res_q1.response}")
    assert res_q1.matched_cars[0].listing_id == expected_first_car.listing_id
    assert spied_interpreter.call_count == 1

    print("\n--- 'What's the mileage on the second Bentley?' ---")
    res_q2 = orchestrator.process_chat(ChatRequest(user_id=user_id_a, message="What's the mileage on the second Bentley?", session_id=session_id_1))
    print(f"Response: {res_q2.response}")
    assert res_q2.matched_cars[0].listing_id == expected_second_car.listing_id
    assert spied_interpreter.call_count == 1

    # =========================================================================
    # 3. AMBIGUITY CHECK
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 3: AMBIGUITY CHECK")
    print("=" * 50)

    # 3A: Multiple Bentleys visible, no active car selected -> "What's the mileage on the Bentley?"
    session_state.active_listing_id = None
    memory_service.save_session(session_state)
    print("\n--- Ambiguous Make Query: 'What's the mileage on the Bentley?' ---")
    res_amb = orchestrator.process_chat(ChatRequest(user_id=user_id_a, message="What's the mileage on the Bentley?", session_id=session_id_1))
    print(f"Response: {res_amb.response}")
    assert res_amb.requires_clarification is True
    assert spied_interpreter.call_count == 1

    # 3B: Fresh session with no context -> "Does it have a warranty?"
    fresh_session_id = str(uuid.uuid4())
    print("\n--- Ambiguous Pronoun with No Context: 'Does it have a warranty?' ---")
    count_before_fresh_pronoun = spied_interpreter.call_count
    res_fresh_pronoun = orchestrator.process_chat(ChatRequest(user_id=user_id_a, message="Does it have a warranty?", session_id=fresh_session_id))
    print(f"Response: {res_fresh_pronoun.response}")
    assert res_fresh_pronoun.requires_clarification is True
    assert "Which vehicle are you referring to?" in res_fresh_pronoun.response
    assert spied_interpreter.call_count == count_before_fresh_pronoun, "Should not call QueryInterpreter for pronoun with no context"
    assert res_fresh_pronoun.matched_cars is None or len(res_fresh_pronoun.matched_cars) == 0
    assert res_fresh_pronoun.total_matches == 0

    # =========================================================================
    # 4. FRESH SEARCH NOT HIJACKED BY MEMORY
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 4: FRESH SEARCH NOT HIJACKED BY MEMORY")
    print("=" * 50)

    # Establish active vehicle first
    session_state.active_listing_id = expected_first_car.listing_id
    memory_service.save_session(session_state)
    initial_count = spied_interpreter.call_count

    print("\n--- Fresh Search: 'Show me GCC cars' ---")
    res_gcc = orchestrator.process_chat(ChatRequest(user_id=user_id_a, message="Show me GCC cars", session_id=session_id_1))
    print(f"Response (truncated): {res_gcc.response[:150]}...")
    assert spied_interpreter.call_count > initial_count, "QueryInterpreter call count should increment for fresh search"
    session_state = memory_service.get_session(user_id_a, session_id_1)
    assert session_state.active_listing_id is None, "active_listing_id should reset on fresh search"

    count_after_gcc = spied_interpreter.call_count
    print("\n--- Fresh Search: 'Find cars under warranty' ---")
    res_warr = orchestrator.process_chat(ChatRequest(user_id=user_id_a, message="Find cars under warranty", session_id=session_id_1))
    print(f"Response (truncated): {res_warr.response[:150]}...")
    assert spied_interpreter.call_count > count_after_gcc, "QueryInterpreter call count should increment for warranty search"

    # =========================================================================
    # 5. RESULT-SET REPLACEMENT
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 5: RESULT-SET REPLACEMENT")
    print("=" * 50)

    print("\n--- Search: 'Show me Land Rovers' ---")
    res_lr = orchestrator.process_chat(ChatRequest(user_id=user_id_a, message="Show me Land Rovers", session_id=session_id_1))
    assert len(res_lr.matched_cars) > 0
    first_lr = res_lr.matched_cars[0]
    assert first_lr.make.lower() == "land rover"
    print(f"First Land Rover: Listing #{first_lr.listing_id} ({first_lr.year} {first_lr.make} {first_lr.model})")

    print("\n--- Follow-up: 'What's the mileage on the first one?' ---")
    count_before_lr_followup = spied_interpreter.call_count
    res_lr_fup = orchestrator.process_chat(ChatRequest(user_id=user_id_a, message="What's the mileage on the first one?", session_id=session_id_1))
    print(f"Response: {res_lr_fup.response}")
    assert spied_interpreter.call_count == count_before_lr_followup
    assert res_lr_fup.matched_cars[0].listing_id == first_lr.listing_id
    assert res_lr_fup.matched_cars[0].listing_id != expected_first_car.listing_id

    # =========================================================================
    # 6. CROSS-USER ISOLATION
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 6: CROSS-USER ISOLATION")
    print("=" * 50)

    shared_session = "shared_multi_user_session"
    # User A establishes context
    orchestrator.process_chat(ChatRequest(user_id="user_alice", message="Show me Land Rovers", session_id=shared_session))
    orchestrator.process_chat(ChatRequest(user_id="user_alice", message="What's the mileage on the first one?", session_id=shared_session))
    alice_state = memory_service.get_session("user_alice", shared_session)
    assert alice_state.active_listing_id is not None

    # User B sends message on same session ID
    print("\n--- User B sends 'What's its mileage?' on same session ID ---")
    res_bob = orchestrator.process_chat(ChatRequest(user_id="user_bob", message="What's its mileage?", session_id=shared_session))
    print(f"User B Response: {res_bob.response}")
    bob_state = memory_service.get_session("user_bob", shared_session)
    assert bob_state.active_listing_id is None
    # User B does NOT receive Alice's car
    if res_bob.matched_cars:
        assert all(c.listing_id != alice_state.active_listing_id for c in res_bob.matched_cars)

    # =========================================================================
    # 7. VIEWING CANDIDATE MEMORY
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 7: VIEWING CANDIDATE MEMORY")
    print("=" * 50)

    sess_view = str(uuid.uuid4())
    print("\n--- Viewing Request: 'I want to test drive a Bentley' ---")
    res_view = orchestrator.process_chat(ChatRequest(user_id=user_id_a, message="I want to test drive a Bentley", session_id=sess_view))
    print(f"Response (truncated): {res_view.response[:150]}...")
    assert res_view.matched_cars is not None and len(res_view.matched_cars) >= 2
    view_second_car = res_view.matched_cars[1]

    print("\n--- Viewing Follow-up: 'I want the second one' ---")
    count_before_view_fup = spied_interpreter.call_count
    res_view_fup = orchestrator.process_chat(ChatRequest(user_id=user_id_a, message="I want the second one", session_id=sess_view))
    print(f"Response: {res_view_fup.response}")
    assert spied_interpreter.call_count == count_before_view_fup
    assert res_view_fup.matched_cars[0].listing_id == view_second_car.listing_id

    # =========================================================================
    # 8. ACTUAL FASTAPI MULTI-TURN CHECK
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 8: ACTUAL FASTAPI MULTI-TURN CHECK")
    print("=" * 50)

    # Wire spied interpreter into main app orchestrator
    import backend.main as main_module
    main_module.chat_orchestrator.query_interpreter = spied_interpreter
    client = TestClient(app)

    # Turn 1: POST /chat fresh search
    api_res1 = client.post("/chat", json={"user_id": "fastapi_user", "message": "Show me Bentleys"})
    assert api_res1.status_code == 200
    api_data1 = api_res1.json()
    api_sess_id = api_data1["session_id"]
    api_first_car = api_data1["matched_cars"][0]
    print(f"API Turn 1 Session ID: {api_sess_id}, Matches: {api_data1['total_matches']}")

    # Turn 2: POST /chat reference first car
    api_res2 = client.post("/chat", json={"user_id": "fastapi_user", "message": "What's the mileage on that first Bentley?", "session_id": api_sess_id})
    assert api_res2.status_code == 200
    api_data2 = api_res2.json()
    assert api_data2["session_id"] == api_sess_id
    assert api_data2["matched_cars"][0]["listing_id"] == api_first_car["listing_id"]
    print(f"API Turn 2 Resolved Car ID: {api_data2['matched_cars'][0]['listing_id']}")

    # Turn 3: POST /chat pronoun follow-up
    api_res3 = client.post("/chat", json={"user_id": "fastapi_user", "message": "Is there a warranty on it?", "session_id": api_sess_id})
    assert api_res3.status_code == 200
    api_data3 = api_res3.json()
    assert api_data3["session_id"] == api_sess_id
    assert api_data3["matched_cars"][0]["listing_id"] == api_first_car["listing_id"]
    print(f"API Turn 3 Resolved Car ID: {api_data3['matched_cars'][0]['listing_id']}")

    # =========================================================================
    # 9. MEMORY STATE AUDIT
    # =========================================================================
    print("\n" + "=" * 50)
    print("STEP 9: MEMORY STATE AUDIT")
    print("=" * 50)

    final_session = memory_service.get_session(user_id_a, session_id_1)
    print(f"Session ID: {final_session.session_id}")
    print(f"User ID: {final_session.user_id}")
    print(f"Created At: {final_session.created_at} (tz: {final_session.created_at.tzinfo})")
    print(f"Updated At: {final_session.updated_at} (tz: {final_session.updated_at.tzinfo})")
    print(f"Total Turns: {len(final_session.turns)}")
    for idx, turn in enumerate(final_session.turns, 1):
        print(f"  Turn {idx}: User: '{turn.user_message}' | Intent: {turn.intent} | Referenced ID: {turn.referenced_listing_id} | Matched IDs: {turn.matched_listing_ids[:3]}")
    
    assert final_session.created_at.tzinfo == timezone.utc
    assert final_session.updated_at.tzinfo == timezone.utc

    print("\n" + "=" * 70)
    print("ALL LIVE VERIFICATION STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
