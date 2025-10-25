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

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from src.config import OLLAMA_MODEL_NAME, OLLAMA_BASE_URL, MESSAGE_PRUNING_LIMIT
from src.agent.prompts import senior_assistant_prompt
from src.agent.state import GraphState
from src.api.ollama_preloader import warm_up_ollama_async
from src.context.protocol import get_dynamic_context

logging.info(f"Ollama Base URL = {OLLAMA_BASE_URL} model = {OLLAMA_MODEL_NAME}")

# This global variable will hold the llm chain after async initialization
llm_chain = None

location = "Guelph, Ontario, Canada"
schedule_summary = """
Breakfast: 8:30am
Exercise: 10 am
Mid-day Medication: 12 pm 
Lunch: 12:30pm
Dinner: 5:30pm
Bingo: 7 pm
Evening Medication: 9 pm
"""

# --- Define Graph Nodes ---
async def invoke_llm(state: GraphState):
    """
    Node to get a response from the LLM based on the conversation history.
    """
    global llm_chain
    if not llm_chain:
        logging.error("LLM chain not initialized!")
        # Return an empty message or handle the error appropriately
        return {"messages": [HumanMessage(content=state["transcribed_text"])]}

    logging.info(f"Reasoning engine received: {state['transcribed_text']}")
    logging.info(f"Current message count: {len(state['messages'])}")
    logging.info(f"Username: {state['username']}")

    # Gather context using the Model Context Protocol
    dynamic_context = await get_dynamic_context(state["username"])

    # Invoke the LLM with the (potentially pruned) message history and the new user input
    response = await llm_chain.ainvoke({
        **dynamic_context,
        "messages": state["messages"],
        "input": state["transcribed_text"]
    })

    logging.info(f"Reasoning engine produced: {response.content}")

    # The node returns the new user message and the AI's response to be added to the state
    return {"messages": [HumanMessage(content=state["transcribed_text"]), response]}


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
    """Builds the graph with a pruning step before the LLM call."""
    global llm_chain

    # Asynchronously warm up the LLM
    llm = await warm_up_ollama_async(OLLAMA_MODEL_NAME, OLLAMA_BASE_URL, temperature=0)

    if not llm:
        raise RuntimeError("Failed to initialize the LLM.")

    # Create the chain with the initialized LLM
    llm_chain = senior_assistant_prompt | llm
    workflow = StateGraph(GraphState)

    # Add the nodes
    workflow.add_node("prune", prune_messages)
    workflow.add_node("llm", invoke_llm)

    # Set the entry point to the new pruning node
    workflow.set_entry_point("prune")

    # Define the flow: prune -> llm -> end
    workflow.add_edge("prune", "llm")
    workflow.add_edge("llm", END)

    return workflow.compile()