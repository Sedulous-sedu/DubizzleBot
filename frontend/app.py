"""Streamlit Web Client Application for DubizzleBot AI Assistant."""

import streamlit as st

from frontend.config import BACKEND_URL
from frontend.api_client import DubizzleAPIClient, DubizzleAPIError
from frontend.state import (
    ensure_initial_state,
    start_new_conversation,
    switch_user,
    queue_prompt,
    consume_queued_prompt,
    add_user_message,
    add_assistant_message,
)
from frontend.components import (
    render_matched_cars,
    render_empty_state,
)

def main():
    st.set_page_config(
        page_title="DubizzleBot — AI Used-Car Assistant",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize client-side state
    ensure_initial_state(st.session_state)
    api_client = DubizzleAPIClient(base_url=BACKEND_URL)

    # =========================================================================
    # SIDEBAR: EVALUATION CONTROLS & IDENTITY
    # =========================================================================
    st.sidebar.title("🚗 DubizzleBot")
    st.sidebar.caption("AI Used-Car Assistant — Evaluation Prototype")
    st.sidebar.markdown("---")

    # 1. User Identity Management (Phase 4B Persistence)
    st.sidebar.markdown(f"**Active User ID:** `{st.session_state['active_user_id']}`")
    user_input_val = st.sidebar.text_input(
        "Switch User Identity",
        value=st.session_state["user_id_input"],
        key="user_input_field",
        help="Simulates returning customer identity for long-term memory & preferences.",
    )
    if st.sidebar.button("👤 Switch User", use_container_width=True):
        if switch_user(st.session_state, user_input_val):
            st.rerun()

    st.sidebar.markdown("---")

    # 2. Session Context Actions (Phase 4A Conversational Context)
    short_session_id = st.session_state["session_id"][:8]
    st.sidebar.markdown(f"**Session ID:** `{short_session_id}...`")
    if st.sidebar.button("➕ New Conversation", use_container_width=True):
        start_new_conversation(st.session_state)
        st.rerun()

    st.sidebar.markdown("---")

    # 3. Backend Health Status
    is_healthy = api_client.health_check()
    if is_healthy:
        st.sidebar.success(f"🟢 Connected to Backend\n\n`{BACKEND_URL}`")
    else:
        st.sidebar.error(
            f"🔴 Backend Offline\n\n`{BACKEND_URL}`\n\n"
            "Run in terminal:\n`uv run uvicorn backend.main:app --reload`"
        )

    st.sidebar.markdown("---")

    # 4. Evaluation Guide
    with st.sidebar.expander("ℹ️ Evaluation Guide & Rules"):
        st.markdown(
            "• **Persistent Memory**: Saved cars and preferences follow your **User ID** across sessions.\n"
            "• **New Conversation**: Generates a new session while preserving saved user favorites.\n"
            "• **Test-Drive Booking**: Monday–Saturday, 8:00 AM to 8:00 PM (Asia/Dubai).\n"
            "• **Lead Qualification**: Requires contact + budget + automotive need + confirmation."
        )

    # 5. Developer Inspector
    with st.sidebar.expander("🛠️ Session Inspector"):
        st.json({
            "active_user_id": st.session_state["active_user_id"],
            "session_id": st.session_state["session_id"],
            "message_count": len(st.session_state["messages"]),
            "backend_url": BACKEND_URL,
            "backend_online": is_healthy,
        })

    # =========================================================================
    # MAIN CHAT AREA
    # =========================================================================
    st.title("🚗 DubizzleBot AI Assistant")
    st.caption("Grounded UAE Used-Car Inventory Search, Preferences, & Test-Drive Bookings")

    if not is_healthy:
        st.warning(
            f"⚠️ Cannot connect to the DubizzleBot FastAPI backend at `{BACKEND_URL}`. "
            "Please ensure the backend server is running: `uv run uvicorn backend.main:app --reload`"
        )

    def on_sample_prompt(prompt_text: str):
        queue_prompt(st.session_state, prompt_text)
        st.rerun()

    # Empty State (Welcome Banner + Starter Chips)
    if len(st.session_state["messages"]) == 0:
        render_empty_state(on_prompt_select=on_sample_prompt)

    # Render Existing Chat History
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("matched_cars"):
                total_matches = msg.get("total_matches", len(msg["matched_cars"]))
                render_matched_cars(
                    matched_cars=msg["matched_cars"],
                    total_matches=total_matches,
                    message_id=msg["id"],
                    state=st.session_state,
                )

    # Handle Input (from Chat Input or Clicked Starter Chip)
    queued_prompt = consume_queued_prompt(st.session_state)
    user_input = st.chat_input("Ask DubizzleBot about cars, preferences, bookings, or test drives...")
    active_message = queued_prompt or user_input

    if active_message:
        # 1. Optimistic User Bubble
        add_user_message(st.session_state, active_message)
        with st.chat_message("user"):
            st.markdown(active_message)

        # 2. Dispatch to Backend & Render Assistant Bubble
        with st.chat_message("assistant"):
            with st.spinner("DubizzleBot is processing your request..."):
                try:
                    response_obj = api_client.send_chat(
                        user_id=st.session_state["active_user_id"],
                        session_id=st.session_state["session_id"],
                        message=active_message,
                    )
                    st.markdown(response_obj.response)
                    if response_obj.matched_cars:
                        render_matched_cars(
                            matched_cars=[c.model_dump() for c in response_obj.matched_cars],
                            total_matches=response_obj.total_matches,
                            message_id="latest",
                            state=st.session_state,
                        )
                    add_assistant_message(st.session_state, response_obj)
                except DubizzleAPIError as e:
                    error_msg = f"⚠️ {e.message}"
                    st.error(error_msg)
                    # Preserve error turn in history to keep conversation synchronized
                    st.session_state["messages"].append({
                        "id": "err",
                        "role": "assistant",
                        "content": error_msg,
                        "matched_cars": None,
                        "intent": "error",
                        "total_matches": 0,
                        "requires_clarification": False,
                    })

        st.rerun()

if __name__ == "__main__":
    main()
