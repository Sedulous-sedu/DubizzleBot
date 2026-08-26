"""Pure state management module for DubizzleBot Streamlit client.
Operates on MutableMapping abstractions for seamless testability in pytest and runtime execution in Streamlit.
"""

import uuid
from typing import MutableMapping, Optional, List, Dict, Any

from frontend.api_client import FrontendChatResponse

DEFAULT_USER_ID = "demo_user"

def ensure_initial_state(state: MutableMapping) -> None:
    """Initializes client session state attributes if not already present."""
    if "active_user_id" not in state:
        state["active_user_id"] = DEFAULT_USER_ID
    if "user_id_input" not in state:
        state["user_id_input"] = DEFAULT_USER_ID
    if "session_id" not in state:
        state["session_id"] = str(uuid.uuid4())
    if "messages" not in state:
        state["messages"] = []
    if "queued_prompt" not in state:
        state["queued_prompt"] = None
    if "page_offsets" not in state:
        state["page_offsets"] = {}

def start_new_conversation(state: MutableMapping) -> str:
    """
    Resets the conversational session context while preserving the persistent user identity.
    Generates a new session_id, clears visible chat messages and UI pagination offsets.
    Returns the newly generated session_id.
    """
    ensure_initial_state(state)
    new_session_id = str(uuid.uuid4())
    state["session_id"] = new_session_id
    state["messages"] = []
    state["queued_prompt"] = None
    state["page_offsets"] = {}
    return new_session_id

def switch_user(state: MutableMapping, new_user_id: str) -> bool:
    """
    Switches active user identity.
    If the trimmed new_user_id is non-empty and differs from active_user_id:
    - Updates active_user_id and user_id_input
    - Generates a new session_id
    - Clears visible messages and per-conversation state
    Returns True if user was switched, False otherwise.
    """
    ensure_initial_state(state)
    clean_user_id = str(new_user_id).strip()
    if not clean_user_id:
        return False
    if clean_user_id == state.get("active_user_id"):
        return False

    state["active_user_id"] = clean_user_id
    state["user_id_input"] = clean_user_id
    state["session_id"] = str(uuid.uuid4())
    state["messages"] = []
    state["queued_prompt"] = None
    state["page_offsets"] = {}
    return True

def queue_prompt(state: MutableMapping, prompt: str) -> None:
    """Queues a sample starter prompt for one-shot consumption upon rerun."""
    ensure_initial_state(state)
    clean_prompt = str(prompt).strip()
    if clean_prompt:
        state["queued_prompt"] = clean_prompt

def consume_queued_prompt(state: MutableMapping) -> Optional[str]:
    """Atomically consumes and clears the queued starter prompt, preventing duplicate submissions."""
    ensure_initial_state(state)
    prompt = state.get("queued_prompt")
    state["queued_prompt"] = None
    return prompt

def add_user_message(state: MutableMapping, content: str) -> Dict[str, Any]:
    """Appends a user chat turn to message history and returns the message dict."""
    ensure_initial_state(state)
    clean_content = str(content).strip()
    msg = {
        "id": str(uuid.uuid4()),
        "role": "user",
        "content": clean_content,
    }
    state["messages"].append(msg)
    return msg

def add_assistant_message(
    state: MutableMapping,
    response: FrontendChatResponse,
) -> Dict[str, Any]:
    """
    Appends an assistant chat turn to message history, preserving complete received matched_cars.
    Returns the message dict.
    """
    ensure_initial_state(state)
    raw_cars = None
    if response.matched_cars is not None:
        raw_cars = [c.model_dump() for c in response.matched_cars]

    msg = {
        "id": str(uuid.uuid4()),
        "role": "assistant",
        "content": response.response,
        "matched_cars": raw_cars,
        "intent": response.intent,
        "total_matches": response.total_matches,
        "requires_clarification": response.requires_clarification,
    }
    state["messages"].append(msg)
    return msg
