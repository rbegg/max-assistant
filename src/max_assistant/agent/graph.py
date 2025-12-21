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
from typing import Literal
import uuid
import re

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_ollama import ChatOllama

from max_assistant.agent.prompts import senior_assistant_prompt, planner_prompt, replanner_prompt
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
    """Builds the graph with planning, execution, and human-in-the-loop."""

    # 1. Initialize Tools and LLMs
    logger.info("Collecting tools from registry...")
    tools = tool_registry.get_all_tools()
    tools.append(get_current_datetime)
    llm_with_tools = llm.bind_tools(tools)
    logger.info(f"Reasoning engine configured with {len(tools)} tools.")

    # 2. Define Nodes

    def prepare_input(state: GraphState):
        """Adds the user's input to the message history."""
        logger.info("Node: prepare_input")
        last_message = state["messages"][-1] if state["messages"] else None
        # Don't add user input again if we are in a tool-use loop
        if not isinstance(last_message, ToolMessage):
            return {"messages": [HumanMessage(content=state["transcribed_text"])]}
        return {}

    async def planner(state: GraphState):
        """
        Generates a plan based on the user's latest input.
        Always replans at the start of a turn to handle context switches.
        """
        logger.info("Node: planner")
        
        # REMOVED: The check that skips planning if a plan exists.
        # We always want to evaluate the plan against the new user input.
        
        logger.info("Generating a new plan based on user input.")
        
        # Create a plain-text description of the tools for the planner's context.
        tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])

        # Use the original LLM, not the one with tools bound.
        planner_chain = planner_prompt | llm
        
        # We pass the *existing* plan as context, so the planner can decide to keep it if relevant,
        # but usually, it should generate a new one for the new request.
        response = await planner_chain.ainvoke({
            "user_info": json.dumps(state["userinfo"], indent=2),
            "messages": state["messages"],
            "tools": tool_descriptions, # Pass the tool descriptions as text.
            "request": state["transcribed_text"]
        })
        plan_str = response.content.strip()

        # If the LLM returns nothing or explicitly says no plan is needed, create a default step.
        if not plan_str or "NO_PLAN_NEEDED" in plan_str:
            logger.info("Planner decided no plan is needed or returned an empty response.")
            return {"plan": ["Respond to the user."]}

        # First, try to parse a numbered list.
        plan = re.findall(r"^\d+\.\s*(.*)", plan_str, re.MULTILINE)

        # If no numbered list is found, treat each non-empty line as a step.
        if not plan:
            logger.info("No numbered list found in plan. Parsing line by line.")
            plan = [line.strip() for line in plan_str.split('\n') if line.strip()]

        # If after all that, the plan is still empty, use a safe default.
        if not plan:
            logger.warning("LLM failed to generate a parseable plan. Defaulting to a single response step.")
            return {"plan": ["Respond to the user."]}

        logger.info(f"Generated plan: {plan}")
        return {"plan": plan}


    async def replan_step(state: GraphState):
        """
        updates the plan based on tool results.
        Replaces 'record_tool_results'.
        """
        logger.info("Node: replan_step")
        
        # 1. Get the result of the tool
        tool_result = state["messages"][-1].content
        current_step = state["plan"][0] if state["plan"] else "Unknown step"
        
        logger.info(f"Step '{current_step}' executed. Result preview: {tool_result[:50]}...")

        # 2. Update History
        new_past_steps = state.get("past_steps", []) + [(current_step, tool_result)]

        # 3. Replan
        # Create context for the replanner
        tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
        
        replanner_chain = replanner_prompt | llm
        
        response = await replanner_chain.ainvoke({
            "user_info": json.dumps(state["userinfo"], indent=2),
            "messages": state["messages"],
            "tools": tool_descriptions,
            "request": state["transcribed_text"],
            "plan": state["plan"],
            "last_step": current_step,
            "last_result": tool_result
        })

        plan_str = response.content.strip()
        
        # Parse the new plan (same logic as the original planner)
        new_plan = re.findall(r"^\d+\.\s*(.*)", plan_str, re.MULTILINE)
        
        if not new_plan:
             # Fallback: if replanner fails to return a list, assumes it's just text explaining the next step
             # or that we should just finish.
             logger.info("Replanner returned unstructured text. Defaulting to response.")
             new_plan = ["Respond to the user."]

        logger.info(f"Replanned. Old plan len: {len(state['plan'])}. New plan len: {len(new_plan)}")
        logger.debug(f"New Plan: {new_plan}")

        return {
            "past_steps": new_past_steps,
            "plan": new_plan 
        }


    async def agent_executor(state: GraphState):
        """
        Executes the next step in the plan. This could be a tool call,
        a question to the user, or the final response.
        """
        logger.info("Node: agent_executor")
        logger.info(f"Executing step '{state['plan'][0]}'...")
        chain = senior_assistant_prompt | llm_with_tools

        response = await chain.ainvoke({
            "user_info": state["userinfo"],
            "current_datetime": current_datetime(),
            "messages": state["messages"],
            "plan": state.get("plan", []),
            "past_steps": state.get("past_steps", []),
        })

        # The response can be a tool call or a message to the user (question/final answer)
        return {"messages": [response]}


    def human_in_the_loop_or_final_response(state: GraphState):
        """
        This node runs after the agent_executor if no tools were called.
        It checks if the plan is done. If so, it sets the final response.
        If not, it assumes the agent asked a question, sets that as the
        response, and prepares for the next turn (human-in-the-loop).
        """
        logger.info("Node: human_in_the_loop_or_final_response")
        agent_response_content = state["messages"][-1].content
        plan = state.get("plan", [])

        # If the plan is empty or has the final 'respond' step, this is the end.
        if not plan or len(plan) == 1:
            logger.info("Plan is complete. Setting final response.")
            step = plan[0] if plan else "Final response"
            return {
                "response": agent_response_content,
                "past_steps": [(step, agent_response_content)],
                "plan": [] # Clear the plan
            }
        else: # The agent is asking a question for human-in-the-loop
            logger.info("Human in the loop: Pausing for user input.")
            current_step = plan[0]
            return {
                "response": agent_response_content, # This is the question for the user
                "past_steps": [(current_step, "Asked user for info.")], # Log that we asked
                "plan": plan[1:] # Consume the 'ask' step
            }

    def record_tool_results(state: GraphState):
        """Records the outcome of a tool call into 'past_steps'."""
        logger.info("Node: record_tool_results")
        tool_result = state["messages"][-1].content
        current_step = state["plan"][0]
        logger.info(f"Step '{current_step}' executed via tool. Result: {tool_result[:100]}...")
        return {
            "past_steps": [(current_step, tool_result)],
            "plan": state["plan"][1:] # Consume the tool step
        }


    # 3. Build the workflow
    workflow = StateGraph(GraphState)

    workflow.add_node("prepare_input", prepare_input)
    workflow.add_node("prune", prune_messages)
    workflow.add_node("planner", planner)
    workflow.add_node("agent", agent_executor)
    tool_node = ToolNode(tools)
    workflow.add_node("execute_tools", tool_node)
    
    # Changed from 'record_tools' to 'replanner'
    workflow.add_node("replanner", replan_step)
    
    workflow.add_node("responder", human_in_the_loop_or_final_response)

    # 4. Add edges
    workflow.set_entry_point("prepare_input")
    workflow.add_edge("prepare_input", "prune")
    workflow.add_edge("prune", "planner")
    workflow.add_edge("planner", "agent")

    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "execute_tools",
            "__end__": "responder"
        },
    )
    # Changed edge to point to replanner
    workflow.add_edge("execute_tools", "replanner")
    workflow.add_edge("replanner", "agent") 
    workflow.add_edge("responder", END) # End of turn


    # 5. Compile and return
    return workflow.compile()