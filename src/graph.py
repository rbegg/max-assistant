# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
   Implementation of the langgraph state graph for the Max Assistant.
"""

import os
from typing import TypedDict, Annotated
import operator
import asyncio
import datetime

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END

from .prompts import senior_assistant_prompt

# --- Configuration ---
# Set the maximum number of messages to retain in the history (user + AI = 2 messages per turn)
# A value of 10 retains the last 5 turns of the conversation.
MESSAGE_PRUNING_LIMIT = 10


# --- Define State ---
# The state is updated to manage a list of messages for conversation history.
# 'operator.add' ensures that new messages are appended to the existing list.
class GraphState(TypedDict):
    # The user's transcribed text for the current turn
    transcribed_text: str
    # The full conversation history, which will be pruned
    messages: Annotated[list[BaseMessage], operator.add]


# --- LLM and Prompt Initialization ---
model_name = os.getenv("OLLAMA_MODEL_NAME", "llama3")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

print(f"Ollama Base URL = {ollama_base_url} model = {model_name}")

llm = ChatOllama(base_url=ollama_base_url, model=model_name, temperature=0)


chain = senior_assistant_prompt | llm

user_name = "Robert"
location = "Guelph, Ontario, Canada"
schedule_summary = """
Breakfast: 8:30am
Exercise: 10:00am
Mid-day Medication: 12:00pm 
Lunch: 12:30pm
Dinner: 5:30pm
Bingo: 7:00pm
Evening Medication: 9:00pm
"""

# --- Define Graph Nodes ---
async def invoke_llm(state: GraphState):
    """
    Node to get a response from the LLM based on the conversation history.
    """
    print(f"Reasoning engine received: {state['transcribed_text']}")

    current_time = datetime.datetime.now().strftime("%A, %B %d, %Y, %I:%M:%S %p")

    # Invoke the LLM with the (potentially pruned) message history and the new user input
    response = await chain.ainvoke({
        "user_name": user_name,
        "location": location,
        "current_time": current_time,
        "schedule_summary": schedule_summary,
        "messages": state["messages"],
        "input": state["transcribed_text"]
    })

    print(f"Reasoning engine produced: {response.content}")

    # The node returns the new user message and the AI's response to be added to the state
    return {"messages": [HumanMessage(content=state["transcribed_text"]), response]}


def prune_messages(state: GraphState):
    """
    Node to prune the history, keeping only the last K messages.
    """
    messages = state["messages"]
    if len(messages) > MESSAGE_PRUNING_LIMIT:
        print(f"--- Pruning messages from {len(messages)} down to {MESSAGE_PRUNING_LIMIT} ---")
        # This overwrites the 'messages' key in the state with the pruned list
        return {"messages": messages[-MESSAGE_PRUNING_LIMIT:]}

    # If no pruning is needed, we don't need to modify the state
    return {}


# --- Build the Graph ---
def create_reasoning_engine():
    """Builds the graph with a pruning step before the LLM call."""
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


# --- Example Usage ---
async def main():
    """Shows how the graph prunes the conversation history."""
    app = create_reasoning_engine()

    # Initial state for the conversation
    conversation_state = {"messages": []}

    # A helper function to run a turn
    async def run_turn(text: str):
        nonlocal conversation_state
        # The input for a run includes the new text and the current message history
        inputs = {"transcribed_text": text, "messages": conversation_state["messages"]}
        response_state = await app.ainvoke(inputs)
        # The output state contains the updated, complete history, which we save for the next turn
        conversation_state = response_state

        print(f"AI Response: {conversation_state['messages'][-1].content}")
        print(f"Current message count: {len(conversation_state['messages'])}")
        print("-" * 20)

    # Run several turns to demonstrate the pruning
    await run_turn("My name is Arthur and I live in Toronto.")
    await run_turn("My favorite color is blue.")
    await run_turn("My dog's name is Sparky.")
    await run_turn("I enjoy gardening.")
    await run_turn("I'm planning a trip to Italy.")  # Message count will be 10 here

    # This turn will trigger the pruning
    print("\n>>> THIS TURN WILL TRIGGER PRUNING <<<\n")
    await run_turn("What was the first thing I told you?")

    # The AI should no longer know my name because that message was pruned
    await run_turn("What is my name?")


if __name__ == "__main__":
    asyncio.run(main())
