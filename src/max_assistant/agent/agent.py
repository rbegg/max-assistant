# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
This module defines the Agent class, which encapsulates the reasoning engine and
manages conversation state. It provides a clean interface for text-based interaction.
"""

import logging
from uuid import uuid4
from typing import Dict, Any, Optional, List

from langchain_core.messages import BaseMessage
from max_assistant.config import DEFAULT_USERNAME, TTS_VOICE
from max_assistant.agent.state import GraphState

logger = logging.getLogger(__name__)


class Agent:
    """Encapsulates the reasoning engine and conversation state management."""

    # FIX 1: Explicitly specify Optional mapping for incoming initialization dict parameters
    def __init__(self, reasoning_engine: Any, initial_user_info: Optional[Dict[str, Any]] = None) -> None:
        self.reasoning_engine = reasoning_engine

        # Safely assign default dictionaries without mutating optional reference frames
        runtime_user_info = initial_user_info if initial_user_info is not None else {}

        self.conversation_state: GraphState = {
            "messages": [],
            "userinfo": runtime_user_info,
            "thread_id": str(uuid4()),
            "transcribed_text": "",
            "voice": TTS_VOICE,
            "external_event": {},
            "is_background": False,
        }
        self.connection_manager = None  # Bound dynamically on client socket connection

        user_name = runtime_user_info.get("user", {}).get("firstName", DEFAULT_USERNAME)
        logger.info(f"Agent initialized for user: {user_name}")

    def set_thread_id(self, thread_id: str) -> None:
        self.conversation_state["thread_id"] = thread_id
        logger.info(f"Thread ID set to {thread_id}")

    def set_voice(self, voice: str) -> None:
        """Sets the TTS voice for the conversation."""
        self.conversation_state["voice"] = voice

    def get_voice(self) -> str:
        """Gets the current TTS voice."""
        return self.conversation_state.get("voice", TTS_VOICE)

    def set_user_info(self, user_info: Dict[str, Any]) -> None:
        self.conversation_state["userinfo"] = user_info
        user_name = user_info.get("user", {}).get("firstName", DEFAULT_USERNAME)
        logger.info(f"Agent user updated to: {user_name}")

    async def ainvoke(self, text_input: str) -> str:
        """Invokes the agent with text input and returns the text response."""
        # FIX 2: To appease missing key constraints while keeping LangGraph sequence lineages linked,
        # initialize the missing required structure key explicitly as an empty list container.
        inputs: GraphState = {
            "messages": [],
            "transcribed_text": text_input,
            "userinfo": self.conversation_state["userinfo"],
            "thread_id": self.conversation_state["thread_id"],
            "voice": self.conversation_state["voice"],
            "external_event": {},
            "is_background": False
        }

        config = {
            "configurable": {
                "thread_id": self.conversation_state["thread_id"],
                "checkpoint_ns": ""
            }
        }

        logger.info(f"Calling Reasoning engine with: {text_input}")
        return await self._execute_graph_turn(inputs, config)

    async def handle_push_event(self, reminder_payload: dict) -> str:
        """
        Invoked by the background reminder poller when a timer expires.
        Injects the reminder payload into the graph state.
        """
        inputs: GraphState = {
            "messages": [],
            "transcribed_text": "",
            "userinfo": self.conversation_state["userinfo"],
            "thread_id": self.conversation_state["thread_id"],
            "voice": self.conversation_state["voice"],
            "external_event": reminder_payload,
            "is_background": True,
        }

        config = {
            "configurable": {
                "thread_id": self.conversation_state["thread_id"],
                "checkpoint_ns": ""
            }
        }

        logger.info(f"Poller event triggered for thread {inputs.get('thread_id')}")
        return await self._execute_graph_turn(inputs, config)

    # FIX 3: Update type mapping bounds to explicitly accept GraphState containers
    async def _execute_graph_turn(self, inputs: GraphState, config: dict[str, Any]) -> str:
        """Internal helper to execute LangGraph and sync state cleanly."""
        try:
            # Type safe replication of existing in-memory history elements
            local_history: List[BaseMessage] = list(self.conversation_state.get("messages", []))

            # Execute the graph turn smoothly
            final_state = await self.reasoning_engine.ainvoke(inputs, config=config)

            if isinstance(final_state, dict):
                new_messages = final_state.get("messages", [])

                # Merge flat configuration parameters
                self.conversation_state.update(final_state)

                # FIX: Streamlined list comprehension clears the unreachable code warning
                # while keeping strict type verification for List[BaseMessage]
                resolved_turn_messages: List[BaseMessage] = [
                    msg for msg in new_messages if isinstance(msg, BaseMessage)
                ]

                # Merge the history tracking arrays cleanly
                self.conversation_state["messages"] = local_history + resolved_turn_messages

                # Merge the history tracking arrays cleanly without mathematical concatenation issues
                self.conversation_state["messages"] = local_history + resolved_turn_messages
            else:
                logger.error(f"Engine returned invalid state type: {type(final_state)}")
                return "" if inputs.get("is_background") else "I encountered an internal logic error."

            llm_response = ""
            current_messages = self.conversation_state.get("messages")
            if current_messages:
                last_message = current_messages[-1]
                if isinstance(last_message.content, str):
                    llm_response = last_message.content

            return llm_response

        except Exception as e:
            logger.error(f"Reasoning engine crashed during execution: {e}", exc_info=True)
            return "" if inputs.get("is_background") else "I'm sorry, I encountered an error."