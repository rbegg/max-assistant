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
            "voice": TTS_VOICE
        }
        user_name = initial_user_info.get("user", {}).get("firstName", DEFAULT_USERNAME)
        logger.info(f"Agent initialized for user: {user_name}")

    async def ainvoke(self, text_input: str) -> str:
        """Invokes the agent with text input and returns the text response."""

        inputs: GraphState = {
            "transcribed_text": text_input,
            "messages": self.conversation_state.get("messages", []),
            "userinfo": self.conversation_state.get("userinfo", {}),
            "thread_id": self.conversation_state.get("thread_id"),
            "voice": self.conversation_state.get("voice", TTS_VOICE)
        }
        logger.info(f"Calling Reasoning engine with: {text_input}")
        final_state = await self.reasoning_engine.ainvoke(inputs)
        self.conversation_state = final_state

        llm_response = ""
        if final_state.get("messages") and len(final_state["messages"]) > 0:
            last_message = final_state["messages"][-1]
            llm_response = last_message.content

        return llm_response


    def set_thread_id(self, thread_id: str):
        self.conversation_state["thread_id"] = thread_id
        logger.info(f"Thread ID set to {thread_id}")

    def set_voice(self, voice: str):
        """Sets the TTS voice for the conversation."""
        self.conversation_state["voice"] = voice

    def get_voice(self) -> str:
        """Gets the current TTS voice."""
        return self.conversation_state.get("voice", TTS_VOICE)

