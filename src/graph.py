import httpx
from typing import TypedDict
from langgraph.graph import StateGraph, END


# --- Define State ---
class GraphState(TypedDict):
    transcribed_text: str
    llm_response: str
    output_audio: bytes


# --- Define Graph Nodes ---
async def invoke_llm(state: GraphState):
    """Node to get a response from an LLM."""
    print(f"User said: {state['transcribed_text']}")
    # Replace with your actual LLM API call (e.g., to OpenAI, Anthropic, etc.)
    response_text = f"You said '{state['transcribed_text']}'. This is a mock response from the assistant."
    return {"llm_response": response_text}


async def synthesize_speech(state: GraphState):
    """Node to convert the LLM's text response to speech using a TTS API."""
    print(f"Assistant says: {state['llm_response']}")
    # Replace with your actual TTS API call (e.g., ElevenLabs, OpenAI TTS)
    mock_audio = b"dummy_audio_bytes_for_" + state['llm_response'].encode('utf-8')
    return {"output_audio": mock_audio}


# --- Build the Graph ---
def create_assistant_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("llm", invoke_llm)
    workflow.add_node("tts", synthesize_speech)

    workflow.set_entry_point("llm")
    workflow.add_edge("llm", "tts")
    workflow.add_edge("tts", END)

    return workflow.compile()