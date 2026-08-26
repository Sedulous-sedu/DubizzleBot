"""Memory service managing in-memory thread-safe short-term conversation state."""

import threading
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from backend.models.car import CarListing
from backend.models.intent import UserIntentEnum
from backend.models.memory import SessionState, ConversationTurn

class MemoryService:
    """
    Thread-safe in-memory session memory service with LRU eviction and strict (user_id, session_id) isolation.
    """

    def __init__(self, max_sessions: int = 1000, max_turns_per_session: int = 50):
        self._max_sessions = max_sessions
        self._max_turns_per_session = max_turns_per_session
        self._sessions: OrderedDict[Tuple[str, str], SessionState] = OrderedDict()
        self._lock = threading.RLock()

    def get_or_create_session(self, user_id: str, session_id: str) -> SessionState:
        """
        Retrieves existing SessionState for (user_id, session_id) or initializes a new one.
        Updates LRU recency on access.
        """
        key = (user_id, session_id)
        with self._lock:
            if key in self._sessions:
                self._sessions.move_to_end(key)
                return self._sessions[key]

            # Evict oldest session if capacity reached
            if len(self._sessions) >= self._max_sessions:
                self._sessions.popitem(last=False)

            new_session = SessionState(
                session_id=session_id,
                user_id=user_id,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                turns=[],
                current_result_set=[],
                active_listing_id=None
            )
            self._sessions[key] = new_session
            return new_session

    def get_session(self, user_id: str, session_id: str) -> Optional[SessionState]:
        """
        Retrieves SessionState for (user_id, session_id) if it exists.
        Returns None if not found, enforcing user isolation.
        """
        key = (user_id, session_id)
        with self._lock:
            if key in self._sessions:
                self._sessions.move_to_end(key)
                return self._sessions[key]
            return None

    def save_session(self, session_state: SessionState) -> None:
        """Saves or updates SessionState."""
        key = (session_state.user_id, session_state.session_id)
        with self._lock:
            session_state.updated_at = datetime.now(timezone.utc)
            self._sessions[key] = session_state
            self._sessions.move_to_end(key)

    def record_turn(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        assistant_response: str,
        intent: UserIntentEnum,
        matched_cars: Optional[List[CarListing]] = None,
        referenced_listing_id: Optional[int] = None,
        replace_result_set: bool = False,
        active_listing_id: Optional[int] = None
    ) -> SessionState:
        """
        Records a completed conversation turn into the session state.
        Optionally replaces current_result_set and updates active_listing_id.
        Trims oldest turns if max_turns_per_session is exceeded.
        """
        with self._lock:
            session = self.get_or_create_session(user_id, session_id)
            matched_ids = [c.listing_id for c in matched_cars] if matched_cars else []

            turn = ConversationTurn(
                user_message=user_message,
                assistant_response=assistant_response,
                intent=intent,
                matched_listing_ids=matched_ids,
                referenced_listing_id=referenced_listing_id
            )

            session.turns.append(turn)
            if len(session.turns) > self._max_turns_per_session:
                session.turns = session.turns[-self._max_turns_per_session:]

            if replace_result_set:
                session.current_result_set = matched_cars if matched_cars is not None else []
                session.active_listing_id = active_listing_id
            else:
                if active_listing_id is not None:
                    session.active_listing_id = active_listing_id
                elif referenced_listing_id is not None:
                    session.active_listing_id = referenced_listing_id

            session.updated_at = datetime.now(timezone.utc)
            self.save_session(session)
            return session

    def clear_session(self, user_id: str, session_id: str) -> bool:
        """Clears a session from memory."""
        key = (user_id, session_id)
        with self._lock:
            if key in self._sessions:
                del self._sessions[key]
                return True
            return False

    def clear_all(self) -> None:
        """Clears all sessions (useful for test isolation)."""
        with self._lock:
            self._sessions.clear()
