# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
This module defines the Agent class, which encapsulates the reasoning engine and
manages conversation state. It provides a clean interface for text-based interaction.
"""

import logging
from uuid import uuid4
from typing import Dict, Any

from max_assistant.config import DEFAULT_USERNAME, TTS_VOICE
from max_assistant.agent.state import GraphState

logger = logging.getLogger(__name__)

class Agent:
    """Encapsulates the reasoning engine and conversation state management."""

    def __init__(self, reasoning_engine, initial_user_info: Dict[str, Any]):
        self.reasoning_engine = reasoning_engine
        self.conversation_state: GraphState = {
            "messages": [],
            "userinfo": initial_user_info,
            "thread_id": str(uuid4()),
            "transcribed_text": "",
            "voice": TTS_VOICE,
            "external_event": {},
            "is_background": False,
        }
        self.connection_manager = None  # Bound dynamically on client socket connection
        user_name = initial_user_info.get("user", {}).get("firstName", DEFAULT_USERNAME)
        logger.info(f"Agent initialized for user: {user_name}")


    def set_thread_id(self, thread_id: str):
        self.conversation_state["thread_id"] = thread_id
        logger.info(f"Thread ID set to {thread_id}")


    def set_voice(self, voice: str):
        """Sets the TTS voice for the conversation."""
        self.conversation_state["voice"] = voice


    def get_voice(self) -> str:
        """Gets the current TTS voice."""
        return self.conversation_state.get("voice", TTS_VOICE)


    async def ainvoke(self, text_input: str) -> str:
        """Invokes the agent with text input and returns the text response."""
        inputs: GraphState = {
            "transcribed_text": text_input,
            "messages": self.conversation_state["messages"],
            "userinfo": self.conversation_state["userinfo"],
            "thread_id": self.conversation_state["thread_id"],
            "voice": self.conversation_state["voice"],
        }
        logger.info(f"Calling Reasoning engine with: {text_input}")
        return await self._execute_graph_turn(inputs)


    async def handle_push_event(self, reminder_payload: dict) -> str:
        """
        Invoked by the background reminder poller when a timer expires.
        Injects the reminder payload into the graph state.
        """
        inputs: GraphState = {
            "transcribed_text": "",
            "messages": self.conversation_state["messages"],
            "userinfo": self.conversation_state["userinfo"],
            "thread_id": self.conversation_state["thread_id"],
            "voice": self.conversation_state["voice"],
            "external_event": reminder_payload,
            "is_background": True,
        }
        logger.info(f"Poller event triggered for thread {inputs.get('thread_id')}")
        return await self._execute_graph_turn(inputs)


    async def _execute_graph_turn(self, inputs: GraphState) -> str:
        """
        Internal helper to execute the LangGraph workflow, safely merge the
        resulting state, and extract the final string response.
        """
        try:
            final_state = await self.reasoning_engine.ainvoke(inputs)

            # Defensive validation and merging
            if isinstance(final_state, dict):
                self.conversation_state.update(final_state)
            else:
                logger.error(f"Engine returned invalid state type: {type(final_state)}")
                return "" if inputs.get("is_background") else "I encountered an internal logic error."

            llm_response = ""
            if self.conversation_state.get("messages") and len(self.conversation_state["messages"]) > 0:
                last_message = self.conversation_state["messages"][-1]
                raw_content = last_message.content

                # Coerce LangChain's potentially complex content into a flat string
                if isinstance(raw_content, str):
                    llm_response = raw_content
                elif isinstance(raw_content, list):
                    logger.warning("Received list content from LLM, flattening to string.")
                    # Extract text from dict blocks or cast to string if it's a list of strings
                    string_blocks = [
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in raw_content
                    ]
                    llm_response = " ".join(string_blocks)

            return llm_response

        except Exception as e:
            logger.error(f"Reasoning engine crashed during execution: {e}", exc_info=True)
            return "" if inputs.get("is_background") else "I'm sorry, I encountered a systemic error while thinking."