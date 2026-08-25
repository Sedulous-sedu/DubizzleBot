"""Memory service handling short-term conversation state and SQLite long-term user preferences."""

from typing import Dict, Any, List, Optional

class MemoryService:
    """Service providing short-term session state and persistent long-term memory."""

    def get_session_history(self, session_id: str) -> List[Dict[str, str]]:
        """Skeleton method for fetching active turn history in session."""
        raise NotImplementedError("Short-term session memory will be implemented in subsequent task.")

    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Skeleton method for fetching long-term persistent user profile preferences."""
        raise NotImplementedError("Long-term persistent user memory will be implemented in subsequent task.")
