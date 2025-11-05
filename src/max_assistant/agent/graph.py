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
from typing import Literal

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


from max_assistant.agent.prompts import senior_assistant_prompt
from max_assistant.agent.state import GraphState
from max_assistant.clients.ollama_preloader import warm_up_ollama_async
from max_assistant.clients.neo4j_client import Neo4jClient
from max_assistant.clients import neo4j_client as neo4j_client_module
from max_assistant.tools.schedule_tools import (
    get_full_schedule, get_appointments_for_date, get_routines_for_date, get_activities_info )
from max_assistant.tools.time_tools import get_current_datetime
from max_assistant.config import OLLAMA_MODEL_NAME, OLLAMA_BASE_URL, MESSAGE_PRUNING_LIMIT
from max_assistant.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
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
async def create_reasoning_engine():
    """Builds the graph with pruning, model calls, and tool execution."""


    logger.info(f"Ollama Base URL = {OLLAMA_BASE_URL} model = {OLLAMA_MODEL_NAME}")

    # 1. Initialize LLM and Tools
    llm = await warm_up_ollama_async(OLLAMA_MODEL_NAME, OLLAMA_BASE_URL, temperature=0)
    if not llm:
        raise RuntimeError("Failed to initialize the LLM.")

    neo4j_client_module.client = await Neo4jClient.create(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
    if not neo4j_client_module.client:
        raise RuntimeError("Failed to initialize the Neo4j client.")

    # 1.4. Create the final tools list from the corrected objects
    tools = [get_full_schedule, get_appointments_for_date, get_routines_for_date, get_activities_info, get_current_datetime]

    # 1.55. Bind the tools
    llm_with_tools = llm.bind_tools(tools)

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
            "user_name": state["username"],
            "location": "Not available",
            "current_datetime": current_datetime(),
            "messages": state["messages"],
        })

        logger.info(f"Model produced: {response.content}")

        # Return only the AI's response to be appended to the state
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