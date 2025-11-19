# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
This module defines a framework for managing conversation history and invoking a large language model (LLM)
to generate responses. It includes mechanisms for pruning conversation history to ensure efficient interaction
with the LLM, as well as a configurable reasoning engine implemented as a state graph.

The module initializes an LLM using the ChatOllama model, constructs conversation nodes for pruning messages
and generating AI responses, and builds an execution graph workflow for multi-node communication.
The reasoning engine incorporates both stateful and asynchronous operations to handle conversational data.

Classes and functions are structured to allow seamless integration of the reasoning engine into
chat applications or AI-powered assistants.
"""

import logging
import json
from typing import Literal, List
import uuid

from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, ToolCall
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama

from max_assistant.agent.prompts import senior_assistant_prompt
from max_assistant.agent.state import GraphState
from max_assistant.tools.registry import ToolRegistry
from max_assistant.tools.time_tools import get_current_datetime
from max_assistant.config import MESSAGE_PRUNING_LIMIT
from max_assistant.utils.datetime_utils import current_datetime

logger = logging.getLogger(__name__)

def prune_messages(state: GraphState):
    """
    Node to prune the history, keeping only the last K messages.
    """
    messages = state["messages"]
    if len(messages) > MESSAGE_PRUNING_LIMIT:
        logger.info(f"--- Pruning messages from {len(messages)} down to {MESSAGE_PRUNING_LIMIT} ---")
        # This overwrites the 'messages' key in the state with the pruned list
        return {"messages": messages[-MESSAGE_PRUNING_LIMIT:]}

    # If no pruning is needed, we don't need to modify the state
    return {}


# --- Build the Graph ---
async def create_reasoning_engine(
        llm: ChatOllama,
        tool_registry: ToolRegistry,):
    """Builds the graph with pruning, model calls, and tool execution."""

    # 1. Initialize Tools from the registry
    logger.info("Collecting tools from registry...")
    tools = tool_registry.get_all_tools()
    tools.append(get_current_datetime)  # Add standalone tools
    llm_with_tools = llm.bind_tools(tools)
    logger.info(f"Reasoning engine configured with {len(tools)} tools.")

    # 2. Define Nodes that will be part of the graph

    def prepare_input(state: GraphState):
        """
        """
        logger.info("Node: prepare_input")
        # We only add the user's input if it's a new turn.
        # If the last message is a ToolMessage, we are in a tool-calling loop
        # and should not add the user's input again.
        last_message = state["messages"][-1] if state["messages"] else None
        if not isinstance(last_message, ToolMessage):
            return {"messages": [HumanMessage(content=state["transcribed_text"])]}
        return {}

    async def call_model(state: GraphState):
        """
        Node to invoke the LLM with the current state. The user's input is already
        in the message history.
        """
        logger.info("Calling model with current history.")

        # The prompt and LLM with tools are combined to form the chain
        chain = senior_assistant_prompt | llm_with_tools

        # The 'messages' in the state now contains the user's latest input.
        response = await chain.ainvoke({
            "user_info": state["userinfo"],
            "current_datetime": current_datetime(),
            "messages": state["messages"],
        })

        logger.info(f"Model produced: {response.content}")

        if response.tool_calls:
            # It's a standard tool call, just return it
            return {"messages": [response]}

        try:
            # Check if the *content* is a JSON tool call
            content_json = json.loads(response.content)
            if isinstance(content_json, dict) and "name" in content_json:
                logger.warning("Raw JSON tool call detected. Re-formatting message.")

                # Create a proper AIMessage with a tool_calls attribute
                tool_call_obj = ToolCall(
                    name=content_json["name"],
                    args=content_json.get("parameters", {}),
                    id=str(uuid.uuid4())  # Create a new ID
                )

                # Create a new message that has the 'tool_calls'
                # attribute that should_continue is looking for.
                new_response = AIMessage(
                    content="",  # Content is now empty
                    tool_calls=[tool_call_obj],
                    id=response.id
                )
                return {"messages": [new_response]}

        except (json.JSONDecodeError, TypeError):
            # It's just a regular text response, not JSON
            pass

            # It's a regular text response, return it as-is
        return {"messages": [response]}

    def should_continue(state: GraphState) -> Literal["execute_tools", "end"]:
        """Conditional node to decide whether to execute tools or end."""

        last_message = state["messages"][-1]
        if last_message.tool_calls:
            logger.info("Node: should_continue - Return= execute_tools")
            return "execute_tools"
        logger.info("Node: should_continue - Return= end")
        return "end"

    # 3. Build the workflow
    workflow = StateGraph(GraphState)

    workflow.add_node("prepare_input", prepare_input)
    workflow.add_node("prune", prune_messages)
    workflow.add_node("agent", call_model)
    workflow.add_node("execute_tools", ToolNode(tools))

    # 4. Add edges
    workflow.set_entry_point("prepare_input")
    workflow.add_edge("prepare_input", "prune")
    workflow.add_edge("prune", "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "execute_tools": "execute_tools",
            "end": END,
        },
    )
    workflow.add_edge("execute_tools", "agent")

    # 5. Compile and return
    return workflow.compile()