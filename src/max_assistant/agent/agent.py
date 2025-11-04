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


class Agent:
    """Encapsulates the reasoning engine and conversation state management."""

    def __init__(self, reasoning_engine):
        self.reasoning_engine = reasoning_engine
        self.conversation_state: GraphState = {
            "messages": [],
            "username": DEFAULT_USERNAME,
            "thread_id": "",
            "transcribed_text": "",
            "voice": TTS_VOICE
        }

    async def ainvoke(self, text_input: str) -> str:
        """Invokes the agent with text input and returns the text response."""
        inputs: GraphState = {
            "transcribed_text": text_input,
            "messages": self.conversation_state.get("messages", []),
            "username": self.conversation_state.get("username", DEFAULT_USERNAME),
            "thread_id": self.conversation_state.get("thread_id", str(uuid4())),
            "voice": self.conversation_state.get("voice", TTS_VOICE)
        }
        logging.info(f"Calling Reasoning engine with: {text_input}")
        final_state = await self.reasoning_engine.ainvoke(inputs)
        self.conversation_state = final_state

        llm_response = ""
        if final_state.get("messages") and len(final_state["messages"]) > 0:
            last_message = final_state["messages"][-1]
            llm_response = last_message.content

        return llm_response


    def set_thread_id(self, thread_id: str):
        self.conversation_state["thread_id"] = thread_id
        logging.info(f"Thread ID set to {thread_id}")

    def set_username(self, username: str):
        """Sets the username for the conversation."""
        self.conversation_state["username"] = username
        thread_id = str(uuid4())
        self.set_thread_id(thread_id)

    def set_voice(self, voice: str):
        """Sets the TTS voice for the conversation."""
        self.conversation_state["voice"] = voice

    def get_voice(self) -> str:
        """Gets the current TTS voice."""
        return self.conversation_state.get("voice", TTS_VOICE)

