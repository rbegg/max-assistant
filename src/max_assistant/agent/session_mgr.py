# max_assistant/agent/sessions.py
import logging
from typing import Dict, Set, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from max_assistant.agent.agent import Agent

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Centralized registry to manage active user connections and agent sessions.
    Supports a 1-to-many relationship allowing a user to connect via multiple devices.
    """

    def __init__(self):
        # Maps user_id string to a set of active Agent connection instances
        self._sessions: Dict[str, Set["Agent"]] = {}

    def register(self, user_id: str, agent: "Agent") -> None:
        """Adds a specific connection's agent to the user's connection pool."""
        if not user_id:
            logger.warning("Attempted to register a session with an empty user_id.")
            return

        if user_id not in self._sessions:
            self._sessions[user_id] = set()

        self._sessions[user_id].add(agent)
        logger.info(f"Registered connection. Total live sessions for user {user_id}: {len(self._sessions[user_id])}")

    def unregister(self, user_id: str, agent: "Agent") -> None:
        """Removes a specific agent connection, cleaning up the user key if empty."""
        if not user_id or user_id not in self._sessions:
            return

        # Discard only this specific connection context
        self._sessions[user_id].discard(agent)
        logger.info(f"Unregistered a connection for user_id: {user_id}")

        # If no active connections remain for this user, clear the dictionary key
        if not self._sessions[user_id]:
            del self._sessions[user_id]
            logger.info(f"All connections severed. Cleared global key for user_id: {user_id}")

    def get_agents(self, user_id: str) -> Set["Agent"]:
        """Retrieves all active agent connection instances for a given user."""
        return self._sessions.get(user_id, set())

    def get_active_user_ids(self) -> list[str]:
        """Retrieves all active agent connection instances for a given user."""
        return list(self._sessions.keys())

    def get_all_sessions(self) -> Dict[str, Set["Agent"]]:
        """Exposes the internal registry map (useful for background pollers)."""
        return self._sessions


# Create a single global instance to maintain singleton access across the app
session_manager = SessionManager()