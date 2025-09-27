import httpx
import math
import wave
import struct
import os
from io import BytesIO
from typing import TypedDict

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
# from onnxruntime.quantization.quant_utils import model_has_external_data


# --- Define State ---
class GraphState(TypedDict):
    transcribed_text: str
    llm_response: str
    output_audio: bytes


# --- LLM and Prompt Initialization ---
model_name = os.getenv("OLLAMA_MODEL_NAME", "llama3")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

print(f"Ollama Base URL = {ollama_base_url} model = {model_name}")

llm = ChatOllama(base_url=ollama_base_url, model=model_name, temperature=0)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("user", "{input}"),
    ]
)
chain = prompt | llm


# --- Define Graph Nodes ---
async def invoke_llm(state: GraphState):
    """Node to get a response from an LLM."""
    print(f"User said: {state['transcribed_text']}")

    # Invoke the pre-initialized chain
    response = await chain.ainvoke({"input": state["transcribed_text"]})

    return {"llm_response": response.content}


async def synthesize_speech(state: GraphState):
    """Node to convert the LLM's text response to speech using a TTS API."""
    print(f"Assistant says: {state['llm_response']}")
    # For testing, generate a short audible tone as the audio response.
    mock_audio = _create_test_sound()
    return {"output_audio": mock_audio}


def _create_test_sound(duration: float = 0.5, frequency: int = 440) -> bytes:
    """
    Generates a sine wave and returns it as WAV-formatted bytes in memory.
    """
    sample_rate = 44100
    num_samples = int(sample_rate * duration)
    amplitude = 16000  # For 16-bit audio

    # Generate PCM samples for a sine wave
    samples = [int(amplitude * math.sin(2 * math.pi * i * frequency / sample_rate)) for i in range(num_samples)]

    # Use an in-memory buffer to write the WAV file
    with BytesIO() as wav_buffer:
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes = 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(struct.pack(f'<{len(samples)}h', *samples))
        return wav_buffer.getvalue()


# --- Build the Graph ---
def create_assistant_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("llm", invoke_llm)
    workflow.add_node("tts", synthesize_speech)

    workflow.set_entry_point("llm")
    workflow.add_edge("llm", "tts")
    workflow.add_edge("tts", END)

    return workflow.compile()