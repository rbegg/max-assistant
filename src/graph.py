import httpx
import math
import wave
import struct
from io import BytesIO
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