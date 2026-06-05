# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
Defines the state for the langgraph graph.
"""
from typing import Annotated
import operator
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict, NotRequired


class GraphState(TypedDict):
    """
    Represents the state of the graph.

    Attributes:
        transcribed_text: The user's transcribed text for the current turn.
        userinfo: The user's cached profile and location data.
        thread_id: Unique identifier for the conversation thread.
        messages: The full conversation history, which will be pruned.
        voice: The current TTS voice configuration.
        external_event: Optional payload containing background timer/reminder data.
    """
    transcribed_text: NotRequired[str]
    userinfo: dict
    thread_id: str
    messages: Annotated[list[BaseMessage], operator.add]
    voice: str
    external_event: NotRequired[dict]
    is_background: NotRequired[bool]