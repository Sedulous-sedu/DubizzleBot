"""End-to-end smoke test verifying Phase 4B cross-session persistent memory with real SQLite and real Gemini."""

import os
import sys
import uuid
import tempfile
from typing import List
from fastapi.testclient import TestClient

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.config import settings
from backend.models.chat import ChatRequest
from backend.models.persistent_memory import PreferencePatch
from backend.services.inventory import InventoryService
from backend.services.memory import MemoryService
from backend.services.persistent_memory import PersistentMemoryService
from backend.services.query_interpreter import QueryInterpreter
from backend.services.orchestrator import ChatOrchestrator
from backend.main import create_app

class SpiedQueryInterpreter(QueryInterpreter):
    """Wrapper around QueryInterpreter that records invocation count for validation."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.call_count = 0

    def interpret(self, query: str):
        self.call_count += 1
        return super().interpret(query)

def run_smoke_test():
    print("=" * 70)
    print("DUBIZZLEBOT PHASE 4B: LIVE PERSISTENT CROSS-SESSION VERIFICATION")
    print("=" * 70)

    # Use isolated temporary SQLite database
    temp_dir = tempfile.mkdtemp()
    temp_db_path = os.path.join(temp_dir, "smoke_test_persistent.db")
    print(f"Temporary SQLite Database: {temp_db_path}")

    try:
        user_id_a = f"phase4b_user_a_{uuid.uuid4().hex[:6]}"
        session_id_a = f"session_a_{uuid.uuid4().hex[:6]}"

        inventory_service = InventoryService()
        spied_interpreter = SpiedQueryInterpreter()
        memory_service_a = MemoryService()
        persistent_memory_a = PersistentMemoryService(db_path=temp_db_path)

        orchestrator_a = ChatOrchestrator(
            query_interpreter=spied_interpreter,
            inventory_service=inventory_service,
            memory_service=memory_service_a,
            persistent_memory=persistent_memory_a
        )

        # =========================================================================
        # 1. SESSION A: LIVE SEARCH & DETERMINISTIC SAVE
        # =========================================================================
        print("\n" + "=" * 50)
        print("STEP 1: SESSION A - LIVE SEARCH & LIKE CAR")
        print("=" * 50)

        # Turn 1: Live Gemini Search for Bentleys
        print(f"\n--- Turn 1 (Session A): 'Show me Bentleys' (Live Gemini: {settings.LLM_MODEL}) ---")
        req1 = ChatRequest(user_id=user_id_a, message="Show me Bentleys", session_id=session_id_a)
        res1 = orchestrator_a.process_chat(req1)
        print(f"Response (truncated):\n{res1.response[:160]}...\n")
        assert res1.total_matches > 1, f"Expected > 1 Bentleys, got {res1.total_matches}"
        assert spied_interpreter.call_count == 1, "Expected 1 QueryInterpreter call for initial search"
        
        saved_listing_ids = [c.listing_id for c in res1.matched_cars]
        print(f"Session A returned Listing IDs: {saved_listing_ids}")
        second_car = res1.matched_cars[1]
        saved_listing_id = second_car.listing_id
        print(f"Exact saved second-car Listing_ID: #{saved_listing_id} ({second_car.year} {second_car.make} {second_car.model})")

        # Turn 2: Deterministic Like on Second Car
        print("\n--- Turn 2 (Session A): 'I like the second one.' (Deterministic Save) ---")
        req2 = ChatRequest(user_id=user_id_a, message="I like the second one.", session_id=session_id_a)
        res2 = orchestrator_a.process_chat(req2)
        print(f"Response: {res2.response}")
        assert res2.total_matches == 1
        assert res2.matched_cars[0].listing_id == saved_listing_id
        assert f"Listing #{saved_listing_id}" in res2.response
        assert spied_interpreter.call_count == 1, "Expected 0 additional LLM calls for saving car"

        # Duplicate save check (idempotency)
        persistent_memory_a.save_liked_car(user_id_a, saved_listing_id)
        assert persistent_memory_a.get_liked_listing_ids(user_id_a) == [saved_listing_id]

        # =========================================================================
        # 2. SESSION B: SIMULATED RESTART & RECALL LIKED CARS
        # =========================================================================
        print("\n" + "=" * 50)
        print("STEP 2: SIMULATED RESTART & RECALL LIKED CARS (SESSION B)")
        print("=" * 50)

        session_id_b = f"session_b_{uuid.uuid4().hex[:6]}"
        # Discard and recreate MemoryService, PersistentMemoryService, and ChatOrchestrator
        memory_service_b = MemoryService()
        persistent_memory_b = PersistentMemoryService(db_path=temp_db_path)

        orchestrator_b = ChatOrchestrator(
            query_interpreter=spied_interpreter,
            inventory_service=inventory_service,
            memory_service=memory_service_b,
            persistent_memory=persistent_memory_b
        )

        # Turn 3: Recall liked cars across session boundary
        print(f"\n--- Turn 3 (Session B): 'What cars did I like?' (Zero LLM Calls) ---")
        req3 = ChatRequest(user_id=user_id_a, message="What cars did I like?", session_id=session_id_b)
        res3 = orchestrator_b.process_chat(req3)
        print(f"Response:\n{res3.response}")
        assert res3.total_matches == 1
        assert res3.matched_cars[0].listing_id == saved_listing_id
        assert spied_interpreter.call_count == 1, "Recalling saved cars must require ZERO additional LLM calls"

        # =========================================================================
        # 3. LIVE PREFERENCE PERSISTENCE & PARTIAL UPDATES
        # =========================================================================
        print("\n" + "=" * 50)
        print("STEP 3: PREFERENCE PERSISTENCE & PARTIAL UPDATES")
        print("=" * 50)

        # Turn 4: Initial preferences (Budget & GCC)
        print("\n--- Turn 4 (Session B): 'My budget is under AED 120,000' and 'I prefer GCC cars' ---")
        res4a = orchestrator_b.process_chat(ChatRequest(user_id=user_id_a, message="My budget is under AED 120,000", session_id=session_id_b))
        res4b = orchestrator_b.process_chat(ChatRequest(user_id=user_id_a, message="I prefer GCC cars", session_id=session_id_b))
        prefs4 = persistent_memory_b.get_preferences(user_id_a)
        assert prefs4.max_price_aed == 120000.0
        assert prefs4.regional_specs == "GCC"
        print(f"Preferences after Turn 4: max_price_aed={prefs4.max_price_aed}, regional_specs={prefs4.regional_specs}")

        # Turn 5: Partial Budget Update (120k -> 150k, GCC must survive)
        print("\n--- Turn 5 (Session B): 'My budget is now AED 150,000' ---")
        res5 = orchestrator_b.process_chat(ChatRequest(user_id=user_id_a, message="My budget is now AED 150,000", session_id=session_id_b))
        prefs5 = persistent_memory_b.get_preferences(user_id_a)
        assert prefs5.max_price_aed == 150000.0
        assert prefs5.regional_specs == "GCC", "Regional specs MUST survive partial budget update!"
        print(f"Preferences after Turn 5: max_price_aed={prefs5.max_price_aed}, regional_specs={prefs5.regional_specs} (preserved)")

        # Turn 6: Clear warranty preference
        print("\n--- Turn 6 (Session B): 'I don't care about warranty anymore' ---")
        res6 = orchestrator_b.process_chat(ChatRequest(user_id=user_id_a, message="I don't care about warranty anymore", session_id=session_id_b))
        prefs6 = persistent_memory_b.get_preferences(user_id_a)
        assert prefs6.warranty_preference is None
        assert prefs6.max_price_aed == 150000.0
        assert prefs6.regional_specs == "GCC"
        print(f"Preferences after Turn 6: warranty_preference=None, budget and specs preserved")

        # =========================================================================
        # 4. PREFERENCE / LAST SEARCH DISTINCTION
        # =========================================================================
        print("\n" + "=" * 50)
        print("STEP 4: PREFERENCE / LAST SEARCH DISTINCTION")
        print("=" * 50)

        # Turn 7: Fresh Search (Land Rovers from 2018)
        print("\n--- Turn 7 (Session B): 'Show me Land Rovers from 2018' (Live Search) ---")
        req7 = ChatRequest(user_id=user_id_a, message="Show me Land Rovers from 2018", session_id=session_id_b)
        res7 = orchestrator_b.process_chat(req7)
        assert spied_interpreter.call_count == 2, "Expected QueryInterpreter call count = 2 after new search"
        
        prefs7 = persistent_memory_b.get_preferences(user_id_a)
        assert prefs7.last_search_filters is not None
        assert prefs7.last_search_filters.make.lower() == "land rover"
        assert prefs7.preferred_make is None, "Last search must NOT overwrite explicit preferred_make!"
        print(f"Last search recorded: make={prefs7.last_search_filters.make}, explicit preferred_make remains {prefs7.preferred_make}")

        # Turn 8: Transparency Check
        print("\n--- Turn 8 (Session B): 'What do you remember about me?' ---")
        req8 = ChatRequest(user_id=user_id_a, message="What do you remember about me?", session_id=session_id_b)
        res8 = orchestrator_b.process_chat(req8)
        print(f"Response:\n{res8.response}")
        assert "Budget: up to AED 150,000" in res8.response
        assert "Regional Specs: GCC" in res8.response
        assert "Land Rover" in res8.response
        assert "1 vehicle in your favorites" in res8.response
        assert "You prefer Land Rover" not in res8.response
        assert spied_interpreter.call_count == 2

        # =========================================================================
        # 5. MEMORY-ASSISTED SEARCH & CURRENT-TURN OVERRIDE
        # =========================================================================
        print("\n" + "=" * 50)
        print("STEP 5: MEMORY-ASSISTED SEARCH & OVERRIDE")
        print("=" * 50)

        # Turn 9: Generic search matching saved preferences
        print("\n--- Turn 9 (Session B): 'Show me cars matching my saved preferences' ---")
        req9 = ChatRequest(user_id=user_id_a, message="Show me cars matching my saved preferences", session_id=session_id_b)
        res9 = orchestrator_b.process_chat(req9)
        print(f"Response (truncated):\n{res9.response[:160]}...\n")
        assert res9.total_matches > 0
        for car in res9.matched_cars:
            if car.price_aed is not None:
                assert car.price_aed <= 150000.0
            if car.regional_specs is not None:
                assert car.regional_specs.upper() == "GCC"
        assert spied_interpreter.call_count == 2, "Memory-assisted search must require 0 LLM calls"

        # Turn 10: Current-turn make override
        print("\n--- Turn 10 (Session B): 'Show me Land Rovers matching my saved preferences' ---")
        # Temporarily save preferred_make = Toyota to verify override
        persistent_memory_b.save_preferences(user_id_a, PreferencePatch(preferred_make="Toyota", max_price_aed=200000.0))
        req10 = ChatRequest(user_id=user_id_a, message="Show me Land Rovers matching my saved preferences", session_id=session_id_b)
        res10 = orchestrator_b.process_chat(req10)
        print(f"Response (truncated):\n{res10.response[:160]}...\n")
        assert res10.total_matches > 0
        for car in res10.matched_cars:
            assert car.make.lower() == "land rover"
            if car.price_aed is not None:
                assert car.price_aed <= 200000.0
        assert spied_interpreter.call_count == 2, "Current-turn make override must require 0 LLM calls"

        # =========================================================================
        # 6. CLEAR / FORGET CHECKS
        # =========================================================================
        print("\n" + "=" * 50)
        print("STEP 6: CLEAR / FORGET CHECKS")
        print("=" * 50)

        # Turn 11: Forget saved preferences
        print("\n--- Turn 11 (Session B): 'Forget my saved preferences' ---")
        req11 = ChatRequest(user_id=user_id_a, message="Forget my saved preferences", session_id=session_id_b)
        res11 = orchestrator_b.process_chat(req11)
        print(f"Response: {res11.response}")
        prefs11 = persistent_memory_b.get_preferences(user_id_a)
        assert prefs11.has_explicit_preferences() is False
        assert prefs11.last_search_filters is not None, "last_search_filters must be preserved!"

        # Turn 12: Verify favorites still exist
        print("\n--- Turn 12 (Session B): 'What cars did I like?' ---")
        res12 = orchestrator_b.process_chat(ChatRequest(user_id=user_id_a, message="What cars did I like?", session_id=session_id_b))
        assert res12.total_matches == 1
        assert res12.matched_cars[0].listing_id == saved_listing_id
        print(f"Saved favorite #{saved_listing_id} survived preference clearing")

        # Turn 13: Clear saved cars
        print("\n--- Turn 13 (Session B): 'Clear my saved cars' ---")
        req13 = ChatRequest(user_id=user_id_a, message="Clear my saved cars", session_id=session_id_b)
        res13 = orchestrator_b.process_chat(req13)
        print(f"Response: {res13.response}")
        assert persistent_memory_b.get_liked_listing_ids(user_id_a) == []

        # Turn 14: Forget everything
        print("\n--- Turn 14 (Session B): 'Forget everything about me' ---")
        req14 = ChatRequest(user_id=user_id_a, message="Forget everything about me", session_id=session_id_b)
        res14 = orchestrator_b.process_chat(req14)
        print(f"Response: {res14.response}")
        assert persistent_memory_b.get_preferences(user_id_a) is None
        assert persistent_memory_b.get_liked_listing_ids(user_id_a) == []

        # =========================================================================
        # 7. CROSS-USER ISOLATION & STALE LISTING BEHAVIOR
        # =========================================================================
        print("\n" + "=" * 50)
        print("STEP 7: CROSS-USER ISOLATION & STALE LISTINGS")
        print("=" * 50)

        user_id_b = f"phase4b_user_b_{uuid.uuid4().hex[:6]}"
        # Populate User A with data
        persistent_memory_b.save_preferences(user_id_a, PreferencePatch(preferred_make="Ferrari"))
        persistent_memory_b.save_liked_car(user_id_a, 9)

        # User B queries
        res_ub_likes = orchestrator_b.process_chat(ChatRequest(user_id=user_id_b, message="What cars did I like?", session_id="sess_b"))
        assert res_ub_likes.total_matches == 0
        assert res_ub_likes.matched_cars is None
        print("User B has 0 liked cars (complete isolation from User A)")

        res_ub_mem = orchestrator_b.process_chat(ChatRequest(user_id=user_id_b, message="What do you remember about me?", session_id="sess_b"))
        assert "Ferrari" not in res_ub_mem.response
        print("User B memory transparency contains 0 leaked User A preferences")

        # Stale Listing Check (Listing 999999)
        persistent_memory_b.save_liked_car(user_id_b, 999999)
        res_stale = orchestrator_b.process_chat(ChatRequest(user_id=user_id_b, message="What cars did I like?", session_id="sess_b"))
        assert res_stale.total_matches == 0
        assert res_stale.matched_cars is None
        assert "no longer available in our active inventory" in res_stale.response
        print("Stale listing #999999 safely reported as unavailable without crashing")

        # =========================================================================
        # 8. ACTUAL FASTAPI CROSS-SESSION HTTP CHECK
        # =========================================================================
        print("\n" + "=" * 50)
        print("STEP 8: FASTAPI CROSS-SESSION HTTP CHECK")
        print("=" * 50)

        test_app = create_app(db_path=temp_db_path)
        client = TestClient(test_app)
        api_user = f"api_user_{uuid.uuid4().hex[:6]}"
        api_sess_1 = f"api_sess_1_{uuid.uuid4().hex[:6]}"
        api_sess_2 = f"api_sess_2_{uuid.uuid4().hex[:6]}"

        # Session 1: Like a car (Listing #9)
        persistent_memory_b.save_liked_car(api_user, 9)

        # Session 2: Recall via POST /chat
        http_res = client.post("/chat", json={"user_id": api_user, "message": "What cars did I like?", "session_id": api_sess_2})
        assert http_res.status_code == 200
        http_data = http_res.json()
        assert http_data["session_id"] == api_sess_2
        assert http_data["total_matches"] == 1
        assert http_data["matched_cars"][0]["listing_id"] == 9
        print(f"FastAPI /chat cross-session recall returned HTTP 200 with verified Listing #9")

        print("\n" + "=" * 70)
        print("PHASE 4B LIVE PERSISTENCE VERIFICATION PASSED")
        print("=" * 70)

    finally:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

if __name__ == "__main__":
    run_smoke_test()
