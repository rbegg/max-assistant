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
                transcript = stt_response.get("data", "").strip()

                if not transcript:
                    continue

                # 2. Immediately forward the json  to the client for UI feedback
                await client_ws.send_text(message_str)
                logging.info(f"Forwarded json to client: '{message_str}'")

                logging.info("Invoking assistant graph with transcript.")
                initial_state = {"transcribed_text": transcript}

                llm_response_sent = False
                output_audio = None

                async for step in assistant_graph.astream(initial_state):
                    # Each step is a dictionary with a single key: the node name.
                    # The value is the dictionary returned by that node.
                    node_name = list(step.keys())[0]

                    # The '__end__' key is yielded last, we don't need to parse it here.
                    if node_name == "__end__":
                        break

                    node_output = step[node_name]

                    # Check if this is the output from the 'llm' node and stream it.
                    if not llm_response_sent and "llm_response" in node_output:
                        llm_response = node_output["llm_response"]
                        logging.info(f"Streaming LLM response to client: {llm_response}")

                        response_payload = {
                            "data": llm_response,
                            "source": "assistant",
                        }
                        await client_ws.send_text(json.dumps(response_payload))
                        llm_response_sent = True

                    # Check if this is the output from the 'tts' node and capture the audio.
                    if "output_audio" in node_output:
                        output_audio = node_output["output_audio"]

                # After the graph is finished, send the audio if we captured it.
                if output_audio:
                    logging.info("Sending synthesized audio response to client.")
                    await client_ws.send_bytes(output_audio)

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