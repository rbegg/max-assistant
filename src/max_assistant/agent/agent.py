# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
This module defines the Agent class, which encapsulates the reasoning engine and
manages conversation state. It provides a clean interface for text-based interaction.
"""

import logging
from uuid import uuid4

from max_assistant.config import DEFAULT_USERNAME, TTS_VOICE
from max_assistant.agent.state import GraphState
from max_assistant.tools.person_tools import PersonTools

logger = logging.getLogger(__name__)

class Agent:
    """Encapsulates the reasoning engine and conversation state management."""

    def __init__(self, reasoning_engine, person_tools: PersonTools):
        self.reasoning_engine = reasoning_engine
        self.person_tools = person_tools
        self.conversation_state: GraphState = {
            "messages": [],
            "userinfo": {},
            "thread_id": "",
            "transcribed_text": "",
            "voice": TTS_VOICE
        }
        self.user_info_dict = None


    async def initialize_session(self):
        """
        Fetches the user's data to initialize the session.
        This should be called once after the Agent is created.
        """
        logger.info("Initializing agent session: fetching user info...")
        user_data = await self.person_tools.get_user_info_internal()
        self.conversation_state["userinfo"] = user_data

        # Try to get the user's name from the loaded data
        user_name = user_data.get("user", {}).get("firstName", DEFAULT_USERNAME)

        logger.info(f"User info loaded for: {user_name}")
        # Set a new thread ID for this new session
        self.set_thread_id(str(uuid4()))


    async def ainvoke(self, text_input: str) -> str:
        """Invokes the agent with text input and returns the text response."""

        inputs: GraphState = {
            "transcribed_text": text_input,
            "messages": self.conversation_state.get("messages", []),
            "userinfo": self.conversation_state.get("userinfo", {}),
            "thread_id": self.conversation_state.get("thread_id", str(uuid4())),
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

