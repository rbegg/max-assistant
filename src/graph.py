import os
from typing import TypedDict

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END


# --- Define State ---
# The state is simplified to only manage the text conversation.
class GraphState(TypedDict):
    transcribed_text: str
    llm_response: str


# --- LLM and Prompt Initialization ---
model_name = os.getenv("OLLAMA_MODEL_NAME", "llama3")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

print(f"Ollama Base URL = {ollama_base_url} model = {model_name}")

llm = ChatOllama(base_url=ollama_base_url, model=model_name, temperature=0)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a helpful assistant for a senior adult.  Keep your answers short and concise with a goal to engage in a conversation."),
        ("user", "{input}"),
    ]
)
chain = prompt | llm


# --- Define Graph Node ---
async def invoke_llm(state: GraphState):
    """Node to get a response from the LLM based on the transcribed text."""
    print(f"Reasoning engine received: {state['transcribed_text']}")
    response = await chain.ainvoke({"input": state["transcribed_text"]})
    print(f"Reasoning engine produced: {response.content}")
    return {"llm_response": response.content}


# --- Build the Graph ---
def create_reasoning_engine():
    """Builds the graph, which is now just a single-node reasoning engine."""
    workflow = StateGraph(GraphState)
    workflow.add_node("llm", invoke_llm)
    workflow.set_entry_point("llm")
    workflow.add_edge("llm", END)
    return workflow.compile()
