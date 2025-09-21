# app/main.py

import os
from typing import Annotated, Dict
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel
from typing_extensions import TypedDict

# Load environment variables from .env file
load_dotenv()


# --- Graph State Definition ---
# This defines the structure of the data that flows through the graph.
# 'messages' will hold the conversation history.
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]


# --- Model and Graph Setup ---
def create_graph():
    """Creates a new LangGraph instance for a conversation."""
    # Initialize the Ollama model from environment variables
    model_name = os.getenv("OLLAMA_MODEL", "llama3")
    llm = ChatOllama(model=model_name, temperature=0)

    # Define the primary node of the graph
    def call_model(state: GraphState):
        """Calls the LLM with the current conversation state."""
        messages = state["messages"]
        response = llm.invoke(messages)
        # The response is added back to the state, updating the conversation
        return {"messages": [response]}

    # Build the graph
    workflow = StateGraph(GraphState)
    workflow.add_node("llm", call_model)
    workflow.set_entry_point("llm")
    workflow.set_finish_point("llm")

    return workflow.compile()


# --- FastAPI Application ---
app = FastAPI(
    title="LangGraph Chat Service",
    description="A simple chat service using LangGraph and Ollama.",
    version="1.0.0",
)

# In-memory storage for conversation graphs.
# For production, use a more persistent storage like Redis or a database.
conversations: Dict[str, any] = {}


# Pydantic models for API request validation
class ChatRequest(BaseModel):
    user_input: str
    session_id: str | None = None  # Optional session_id


# --- API Endpoint ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Handles a chat request, maintains context, and streams the LLM response.
    """
    session_id = request.session_id or str(uuid4())

    # Get or create a new graph for the session
    if session_id not in conversations:
        # Add a system prompt for a new conversation
        system_message = SystemMessage(content="You are a helpful AI assistant.")
        conversations[session_id] = {
            "graph": create_graph(),
            "initial_messages": [system_message]
        }

    # Retrieve the session's graph and initial messages
    session_data = conversations[session_id]
    graph = session_data["graph"]
    initial_messages = session_data["initial_messages"]

    async def stream_response():
        """Generator function to stream the response chunks."""
        # The 'stream' method processes the input through the graph
        # and yields the output chunks as they are generated.
        async for chunk in graph.astream(
                {"messages": [HumanMessage(content=request.user_input)]},
                config={"configurable": {"thread_id": session_id}}  # This is key for state management
        ):
            if "llm" in chunk:
                ai_message_chunk = chunk["llm"]["messages"][-1].content
                yield ai_message_chunk

    return StreamingResponse(stream_response(), media_type="text/plain")


@app.get("/")
def read_root():
    return {"status": "LangGraph chat service is running."}