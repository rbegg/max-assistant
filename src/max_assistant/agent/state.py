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
        plan: A list of steps for the agent to execute.
        past_steps: A list of already executed steps and their results.
        response: The final response to the user.
    """
    transcribed_text: str
    userinfo: dict
    thread_id: str
    messages: Annotated[list[BaseMessage], operator.add]
    voice: str
    plan: list[str]
    past_steps: Annotated[list[tuple], operator.add]
    response: str