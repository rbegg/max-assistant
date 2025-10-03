# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
Provides a WebSocket-based API for handling client interactions with a conversational AI agent, including speech-to-text,
text-based reasoning, and synthesized speech output.
This module integrates various components such as STT, TTS, and reasoning through queues to decouple WebSocket
communications from processing logic. It includes the lifecycle management of background tasks and handles graceful
shutdowns for better resource management.
Attributes:
    app (FastAPI): The main FastAPI application instance.
    binary_input_queue (asyncio.Queue): Queue for handling binary input data, primarily for audio data.
    text_input_queue (asyncio.Queue): Queue for handling incoming text messages.
    client_output_queue (asyncio.Queue): Queue for sending messages to the WebSocket client.
    shutdown_event (asyncio.Event): Event signaling tasks to stop during shutdown.
    conversation_state (GraphState): Shared state of the conversation including transcribed text, username, and messages.
    reasoning_engine: The reasoning engine, which processes the state and provides responses to client queries.
Functions:
    lifespan(app): Manages application startup and shutdown, launching and stopping background tasks.
    text_input_handler_loop(): Processes text input data from the client, and updates conversation state.
    agent_loop(): Main conversational loop handling STT output, reasoning engine invocation, TTS generation,
                  and client output responses.
    websocket_endpoint(client_ws): Handles WebSocket connections, bridging client communications with internal queues.
"""

import uvicorn
import asyncio
import logging
import json
from contextlib import asynccontextmanager  # 1. Import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from asyncio import Queue

from .graph import create_reasoning_engine
from .stt_service import transcript_generator
from .tts_service import synthesize_speech
from .state import GraphState
from .config import DEFAULT_USERNAME, TTS_VOICE, PORT, LOG_LEVEL, LOG_FORMAT, SHUTDOWN_TIMEOUT, QUEUE_GET_TIMEOUT

# --- Configuration ---
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

# --- Queues for decoupling WebSocket I/O from agent logic ---
binary_input_queue = Queue()
text_input_queue = Queue()
client_output_queue = Queue()

# --- Graceful Shutdown Setup ---
shutdown_event = asyncio.Event()
# Global state is still needed for the agent and text handler loops
conversation_state: GraphState = {
    "messages": [],
    "username": DEFAULT_USERNAME,
    "transcribed_text": "",
    "voice": TTS_VOICE }


# 2. Create the new lifespan context manager to replace on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's startup and shutdown logic, handling background tasks.
    """
    logging.info("Launching agent and text handler background tasks...")
    # Startup logic: create background tasks
    agent_task = asyncio.create_task(agent_loop())
    text_handler_task = asyncio.create_task(text_input_handler_loop())

    # The 'yield' pauses the function; the application runs while it's paused.
    yield

    # Shutdown logic: runs after the application is shutting down
    logging.info("Shutdown event received. Signaling tasks to stop.")
    shutdown_event.set()

    tasks_to_await = [task for task in (agent_task, text_handler_task) if task]
    if tasks_to_await:
        try:
            await asyncio.wait_for(asyncio.gather(*tasks_to_await), timeout=SHUTDOWN_TIMEOUT)
        except asyncio.TimeoutError:
            logging.warning("Tasks did not finish in time, cancelling.")
            for task in tasks_to_await:
                task.cancel()
        except asyncio.CancelledError:
            logging.info("Tasks were cancelled during shutdown.")
    logging.info("Shutdown complete.")


# 3. Instantiate FastAPI with the lifespan manager
app = FastAPI(lifespan=lifespan)

# NOTE: The global variables 'agent_task' and 'text_handler_task' are removed.
# Their scope is now contained entirely within the 'lifespan' function.

# 4. DELETED the deprecated @app.on_event("startup") and @app.on_event("shutdown") functions


# --- Text Input Handler ---
async def text_input_handler_loop():
    while not shutdown_event.is_set():
        try:
            # Use a timeout to allow the loop to check for the shutdown event
            text_data = await asyncio.wait_for(text_input_queue.get(), timeout=QUEUE_GET_TIMEOUT)
            logging.info(f"TEXT_HANDLER: Received text from client: {text_data}")
            client_dict = json.loads(text_data)
            if "username" in client_dict:
                conversation_state["username"] = client_dict.get("username", conversation_state["username"])
            if "voice" in client_dict:
                conversation_state["voice"] = client_dict.get("voice", conversation_state["voice"])
            # This is a placeholder for future text-based command handling
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            logging.info("Text input handler loop cancelled.")
            break
    logging.info("Text input handler loop has stopped.")


# --- Background Agent Task ---
async def agent_loop():
    global conversation_state

    # The transcript_generator now reads from our decoupled queue
    async for stt_message_str in transcript_generator(binary_input_queue):
        if shutdown_event.is_set():
            break
        try:
            stt_response = json.loads(stt_message_str)
            transcript = stt_response.get("data", "").strip()
            if not transcript:
                continue

            # 1. Send transcribed text back to the client via output queue
            await client_output_queue.put(stt_message_str)

            # 2. Send transcribed text to the reasoning engine
            inputs: GraphState = {
                "transcribed_text": transcript,
                "messages": conversation_state.get("messages", []),
                "username": conversation_state.get("username", DEFAULT_USERNAME),
                "voice": conversation_state.get("voice", TTS_VOICE)
            }
            final_state = await reasoning_engine.ainvoke(inputs)
            conversation_state = final_state

            # 3. Extract the response and send to client (as text)
            llm_response = ""
            if final_state.get("messages") and len(final_state["messages"]) > 0:
                last_message = final_state["messages"][-1]
                llm_response = last_message.content

            response_payload = {"data": llm_response, "source": "assistant"}
            await client_output_queue.put(json.dumps(response_payload))

            # 4. Send reasoning response to TTS to get audio
            output_audio = await synthesize_speech(llm_response, conversation_state.get("voice", TTS_VOICE))

            # 5. Send synthesized audio to the client
            await client_output_queue.put(output_audio)

        except (json.JSONDecodeError, AttributeError) as e:
            logging.warning(f"Could not parse STT message: {stt_message_str} ({e})")
            continue
        except asyncio.CancelledError:
            logging.info("Agent loop cancelled.")
            break
    logging.info("Agent loop has stopped.")


# --- WebSocket Endpoint ---
@app.websocket("/ws")
async def websocket_endpoint(client_ws: WebSocket):
    await client_ws.accept()
    logging.info("Client connected.")

    # These tasks bridge the websocket and the queues
    async def client_reader():
        try:
            while True:
                message = await client_ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                if 'text' in message:
                    await text_input_queue.put(message['text'])
                elif 'bytes' in message:
                    await binary_input_queue.put(message['bytes'])
        except WebSocketDisconnect:
            logging.info("Client disconnected (reader).")
        except Exception as e:
            logging.error(f"WS reader error: {e}", exc_info=True)

    async def client_writer():
        try:
            while True:
                message = await client_output_queue.get()
                if isinstance(message, bytes):
                    await client_ws.send_bytes(message)
                else:  # str
                    await client_ws.send_text(str(message))
        except WebSocketDisconnect:
            logging.info("Client disconnected (writer).")
        except Exception as e:
            logging.error(f"WS writer error: {e}", exc_info=True)

    reader_task = asyncio.create_task(client_reader())
    writer_task = asyncio.create_task(client_writer())

    try:
        # Keep the connection alive until one of the tasks finishes
        done, pending = await asyncio.wait(
            [reader_task, writer_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
    finally:
        logging.info("Client connection handler finished.")


# --- Reasoning Engine ---
reasoning_engine = create_reasoning_engine()

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)