import uvicorn
import asyncio
import logging
import os
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from websockets.client import connect as websocket_connect
from websockets.exceptions import ConnectionClosed

# Import the local graph builder
from .graph import create_assistant_graph

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# The URL for the STT service, configurable via environment variable
STT_WEBSOCKET_URL = os.environ.get("STT_WEBSOCKET_URL", "ws://stt/ws")

app = FastAPI()

# Compile the LangGraph workflow when the application starts
assistant_graph = create_assistant_graph()


async def forward_audio_to_stt(client_ws: WebSocket, stt_ws):
    """
    Receives audio from the client and forwards it to the STT service
    as a continuous stream. This runs as a concurrent task.
    """
    try:
        while True:
            audio_chunk = await client_ws.receive_bytes()
            await stt_ws.send(audio_chunk)
    except WebSocketDisconnect:
        logging.info("Client disconnected. Audio forwarding task is stopping.")
    except ConnectionClosed:
        logging.warning("STT connection closed. Audio forwarding task is stopping.")
    except Exception as e:
        logging.error(f"Error in audio forwarding task: {e}", exc_info=True)


async def handle_stt_and_graph_responses(client_ws: WebSocket, stt_ws):
    """
    Receives JSON messages from STT, streams them to the client, and
    invokes the assistant graph on "final" transcripts.
    """
    try:
        while True:
            # 1. Wait for a message from the STT service
            message_str = await stt_ws.recv()

            try:
                stt_response = json.loads(message_str)
                transcript = stt_response.get("transcript", "").strip()

                if not stt_response["transcript"]:
                    continue

                # 2. Immediately forward the json  to the client for UI feedback
                await client_ws.send_text(message_str)
                logging.info(f"Forwarded json to client: '{message_str}'")

                # 3. If the transcript is final, trigger the assistant graph
                if stt_response.get("is_final"):
                    logging.info("Invoking assistant graph with final transcript.")
                    initial_state = {"transcribed_text": transcript}
                    final_state = await assistant_graph.ainvoke(initial_state)

                    # 4. Send the final TTS audio response back to the client
                    if final_state and final_state.get("output_audio"):
                        logging.info("Sending synthesized audio response to client.")
                        await client_ws.send_bytes(final_state["output_audio"])

            except (json.JSONDecodeError, AttributeError):
                # This could happen if the STT service sends a non-JSON message or malformed data
                logging.warning(f"Could not decode or parse STT message: {message_str}")
                continue

    except ConnectionClosed:
        logging.info("STT connection closed. Response handler is stopping.")
    except Exception as e:
        logging.error(f"Error in STT response handler: {e}", exc_info=True)


@app.websocket("/ws")
async def websocket_endpoint(client_ws: WebSocket):
    """
    Manages the bidirectional WebSocket bridge between the client and STT service.
    """
    await client_ws.accept()
    logging.info("Client connected.")

    stt_ws = None
    try:
        async with websocket_connect(STT_WEBSOCKET_URL) as stt_ws:
            logging.info(f"Connected to STT service at {STT_WEBSOCKET_URL}.")

            # Run concurrent tasks for bidirectional streaming
            audio_forwarder_task = asyncio.create_task(forward_audio_to_stt(client_ws, stt_ws))
            stt_handler_task = asyncio.create_task(handle_stt_and_graph_responses(client_ws, stt_ws))

            # Wait for either task to complete (indicating a disconnection)
            done, pending = await asyncio.wait(
                [audio_forwarder_task, stt_handler_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Clean up by canceling the other running task
            for task in pending:
                task.cancel()
            for task in done:
                if task.exception():
                    raise task.exception()

    except Exception as e:
        logging.error(f"An error occurred in the main WebSocket endpoint: {e}", exc_info=True)
    finally:
        logging.info("Client connection handler finished.")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000)