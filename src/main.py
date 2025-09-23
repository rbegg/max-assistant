import uvicorn
import asyncio
import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from websockets.client import connect as websocket_connect

# Import the local graph builder
from .graph import create_assistant_graph

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# The URL for the STT service, accessed via its Docker service name
STT_WEBSOCKET_URL = "ws://stt-service:8000/ws"

app = FastAPI()

# Compile the LangGraph workflow when the application starts
assistant_graph = create_assistant_graph()


@app.websocket("/ws")
async def websocket_endpoint(client_ws: WebSocket):
    """
    Handles the primary client connection. This function IS the LangGraph application's
    entry point, managing both the workflow and the real-time communication.
    """
    await client_ws.accept()
    logging.info("Client connected to Assistant.")

    try:
        # Establish a persistent WebSocket connection to the STT service
        async with websocket_connect(STT_WEBSOCKET_URL) as stt_ws:
            logging.info("Assistant connected to STT service.")

            async def forward_audio_to_stt():
                """A dedicated task to stream audio from the client to the STT service."""
                while True:
                    audio_chunk = await client_ws.receive_bytes()
                    await stt_ws.send(audio_chunk)

            async def handle_responses():
                """
                A task to listen for STT results, forward them to the client,
                and invoke the LOCAL LangGraph workflow upon receiving a final transcript.
                """
                while True:
                    message = await stt_ws.recv()
                    stt_response = json.loads(message)

                    # 1. Immediately forward the live transcript to the client for UI feedback
                    client_message = {"type": "transcript", "data": stt_response}
                    await client_ws.send_text(json.dumps(client_message))

                    # 2. If the transcript is final, trigger the local graph
                    if stt_response.get("final"):
                        final_transcript = stt_response["transcript"].strip()
                        if not final_transcript:
                            continue

                        logging.info(f"Final transcript: '{final_transcript}'. Invoking local graph.")

                        # --- LOCAL LANGGRAPH INVOCATION ---
                        initial_state = {"transcribed_text": final_transcript}
                        final_state = await assistant_graph.ainvoke(initial_state)

                        # Send the final TTS audio response back to the client
                        if final_state.get("output_audio"):
                            await client_ws.send_bytes(final_state["output_audio"])

            # Run both concurrent tasks
            await asyncio.gather(forward_audio_to_stt(), handle_responses())

    except WebSocketDisconnect:
        logging.info("Client disconnected from Assistant.")
    except Exception as e:
        logging.error(f"Assistant Error: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000)