# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
Defines the state for the langgraph graph.
"""
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import BaseMessage


class GraphState(TypedDict):
    """
    Represents the state of the graph.

    Attributes:
        transcribed_text: The user's transcribed text for the current turn.
        username: The user's name.
        messages: The full conversation history, which will be pruned.
    """
    transcribed_text: str
    username: str
    messages: Annotated[list[BaseMessage], operator.add]
    voice: str