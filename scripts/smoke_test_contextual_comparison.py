"""Live multi-turn verification script for DubizzleBot Contextual Model-Year Comparisons."""

import uuid
from backend.models.chat import ChatRequest
from backend.services.orchestrator import ChatOrchestrator

def run_live_verification():
    orchestrator = ChatOrchestrator()
    session_id = str(uuid.uuid4())
    user_id = f"live_eval_{uuid.uuid4().hex[:6]}"

    print("==================================================")
    print("LIVE MULTI-TURN CONTEXTUAL COMPARISON VALIDATION")
    print(f"User: {user_id} | Session: {session_id}")
    print("==================================================")

    # ----------------------------------------------------
    # Flow 1: Search Fords
    # ----------------------------------------------------
    print("\n--- TURN 1: Search Fords ---")
    req1 = ChatRequest(user_id=user_id, session_id=session_id, message="Show me Fords")
    res1 = orchestrator.process_chat(req1)
    print(f"Response:\n{res1.response}\n")
    print(f"Matched count: {len(res1.matched_cars or [])} | Total: {res1.total_matches}")
    assert res1.matched_cars and len(res1.matched_cars) > 0, "Expected Ford search results"
    original_fords = res1.matched_cars
    ford_years = [c.year for c in original_fords]
    max_year = max(ford_years)
    min_year = min(ford_years)
    print(f"Visible Ford years: {ford_years} -> Max: {max_year}, Min: {min_year}")

    # ----------------------------------------------------
    # Flow 2: Which is the latest year model?
    # ----------------------------------------------------
    print("\n--- TURN 2: Which is the latest year model? ---")
    req2 = ChatRequest(user_id=user_id, session_id=session_id, message="Which is the latest year model?")
    res2 = orchestrator.process_chat(req2)
    print(f"Response:\n{res2.response}\n")
    assert str(max_year) in res2.response, f"Expected latest year {max_year} in response"
    expected_latest_cars = [c for c in original_fords if c.year == max_year]
    assert res2.total_matches == len(expected_latest_cars), f"Expected {len(expected_latest_cars)} winners"
    assert [c.listing_id for c in res2.matched_cars] == [c.listing_id for c in expected_latest_cars]
    print(f"✅ Latest comparison succeeded: Year {max_year} with {res2.total_matches} winner(s)")

    # ----------------------------------------------------
    # Flow 3: Which is the oldest?
    # ----------------------------------------------------
    print("\n--- TURN 3: Which is the oldest? ---")
    req3 = ChatRequest(user_id=user_id, session_id=session_id, message="Which is the oldest?")
    res3 = orchestrator.process_chat(req3)
    print(f"Response:\n{res3.response}\n")
    assert str(min_year) in res3.response, f"Expected oldest year {min_year} in response"
    expected_oldest_cars = [c for c in original_fords if c.year == min_year]
    assert res3.total_matches == len(expected_oldest_cars), f"Expected {len(expected_oldest_cars)} oldest winners"
    assert [c.listing_id for c in res3.matched_cars] == [c.listing_id for c in expected_oldest_cars]
    print(f"✅ Oldest comparison succeeded: Year {min_year} with {res3.total_matches} winner(s)")

    # ----------------------------------------------------
    # Flow 4: What's the mileage on the second one? (Ordinal preservation)
    # ----------------------------------------------------
    print("\n--- TURN 4: What's the mileage on the second one? ---")
    req4 = ChatRequest(user_id=user_id, session_id=session_id, message="What's the mileage on the second one?")
    res4 = orchestrator.process_chat(req4)
    print(f"Response:\n{res4.response}\n")
    assert len(original_fords) >= 2, "Expected at least 2 Fords for ordinal test"
    second_ford = original_fords[1]
    assert res4.matched_cars and res4.matched_cars[0].listing_id == second_ford.listing_id, (
        f"Expected second car #{second_ford.listing_id}, got #{res4.matched_cars[0].listing_id if res4.matched_cars else None}"
    )
    print(f"✅ Ordinal preservation verified: Resolved #{second_ford.listing_id} ({second_ford.year} {second_ford.make} {second_ford.model})")

    # ----------------------------------------------------
    # Flow 5: Search Bentleys + Save to Favorites + Recall
    # ----------------------------------------------------
    print("\n--- TURN 5: Show me Bentleys ---")
    req5 = ChatRequest(user_id=user_id, session_id=session_id, message="Show me Bentleys")
    res5 = orchestrator.process_chat(req5)
    print(f"Response:\n{res5.response}\n")
    assert res5.matched_cars and len(res5.matched_cars) > 0
    second_bentley = res5.matched_cars[1] if len(res5.matched_cars) > 1 else res5.matched_cars[0]

    print("\n--- TURN 6: I like the second one ---")
    req6 = ChatRequest(user_id=user_id, session_id=session_id, message="I like the second one")
    res6 = orchestrator.process_chat(req6)
    print(f"Response:\n{res6.response}\n")
    assert f"#{second_bentley.listing_id}" in res6.response

    print("\n--- TURN 7: What cars did I like? ---")
    req7 = ChatRequest(user_id=user_id, session_id=session_id, message="What cars did I like?")
    res7 = orchestrator.process_chat(req7)
    print(f"Response:\n{res7.response}\n")
    assert f"#{second_bentley.listing_id}" in res7.response
    print("✅ Persistent memory flow verified successfully.")

    # ----------------------------------------------------
    # Flow 6: Unsupported search confirmation flow
    # ----------------------------------------------------
    print("\n--- TURN 8: Unsupported search (5 cheapest Bentleys) ---")
    req8 = ChatRequest(user_id=user_id, session_id=session_id, message="Show me the 5 cheapest Bentleys")
    res8 = orchestrator.process_chat(req8)
    print(f"Response:\n{res8.response}\n")
    assert "reliably rank" in res8.response or "ranking" in res8.response or "cannot deterministically filter" in res8.response

    print("\n--- TURN 9: Yes (confirm supported criteria) ---")
    req9 = ChatRequest(user_id=user_id, session_id=session_id, message="Yes")
    res9 = orchestrator.process_chat(req9)
    print(f"Response:\n{res9.response}\n")
    assert res9.total_matches > 0 and all(c.make.lower() == "bentley" for c in (res9.matched_cars or []))
    print("✅ Unsupported search confirmation verified successfully.")

    # ----------------------------------------------------
    # Flow 7: Empty session comparison clarification
    # ----------------------------------------------------
    print("\n--- TURN 10: Empty session comparison ---")
    empty_session_id = str(uuid.uuid4())
    req10 = ChatRequest(user_id=f"empty_{uuid.uuid4().hex[:4]}", session_id=empty_session_id, message="Which is the latest?")
    res10 = orchestrator.process_chat(req10)
    print(f"Response:\n{res10.response}\n")
    assert res10.requires_clarification is True
    assert res10.total_matches == 0
    assert "Search for some cars first" in res10.response
    print("✅ Empty session comparison clarification verified.")

    print("\n==================================================")
    print("ALL LIVE VERIFICATION CHECKS PASSED PERFECTLY!")
    print("==================================================")

if __name__ == "__main__":
    run_live_verification()
