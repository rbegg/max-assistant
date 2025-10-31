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
from datetime import datetime

from langchain_core.messages import HumanMessage, BaseMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from src.config import OLLAMA_MODEL_NAME, OLLAMA_BASE_URL, MESSAGE_PRUNING_LIMIT
from src.agent.prompts import senior_assistant_prompt
from src.agent.state import GraphState
from src.api.ollama_preloader import warm_up_ollama_async
from src.tools.neo4j_tools import get_schedule_summary
from src.tools.time_tools import get_current_time

logging.info(f"Ollama Base URL = {OLLAMA_BASE_URL} model = {OLLAMA_MODEL_NAME}")



def prune_messages(state: GraphState):
    """
    Node to prune the history, keeping only the last K messages.
    """
    messages = state["messages"]
    if len(messages) > MESSAGE_PRUNING_LIMIT:
        logging.info(f"--- Pruning messages from {len(messages)} down to {MESSAGE_PRUNING_LIMIT} ---")
        # This overwrites the 'messages' key in the state with the pruned list
        return {"messages": messages[-MESSAGE_PRUNING_LIMIT:]}

    # If no pruning is needed, we don't need to modify the state
    return {}


# --- Build the Graph ---
async def create_reasoning_engine():
    """Builds the graph with pruning, model calls, and tool execution."""

    # 1. Initialize LLM and Tools
    llm = await warm_up_ollama_async(OLLAMA_MODEL_NAME, OLLAMA_BASE_URL, temperature=0)
    if not llm:
        raise RuntimeError("Failed to initialize the LLM.")

    tools = [get_schedule_summary, get_current_time]
    llm_with_tools = llm.bind_tools(tools)

    # 2. Define Nodes that will be part of the graph

    def prepare_input(state: GraphState):
        """
        Takes the user's transcribed text and adds it to the message history
        as a HumanMessage. This is the first step in the graph.
        """
        logging.info("Node: prepare_input")
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
        logging.info("Calling model with current history.")

        # The prompt and LLM with tools are combined to form the chain
        chain = senior_assistant_prompt | llm_with_tools

        # The 'messages' in the state now contains the user's latest input.
        response = await chain.ainvoke({
            "user_name": state["username"],
            "location": "Not available",
            "messages": state["messages"],
        })

        logging.info(f"Model produced: {response.content}")

        # Return only the AI's response to be appended to the state
        return {"messages": [response]}

    def should_continue(state: GraphState) -> Literal["execute_tools", "end"]:
        """Conditional node to decide whether to execute tools or end."""

        last_message = state["messages"][-1]
        if last_message.tool_calls:
            logging.info("Node: should_continue - Return= execute_tools")
            return "execute_tools"
        logging.info("Node: should_continue - Return= end")
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