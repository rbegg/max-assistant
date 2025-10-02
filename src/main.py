import uvicorn
import asyncio
import logging
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from asyncio import Queue

from .graph import create_reasoning_engine
from .stt_service import transcript_generator
from .tts_service import synthesize_speech

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = FastAPI()

# --- Queues for decoupling WebSocket I/O from agent logic ---
binary_input_queue = Queue()
text_input_queue = Queue()
client_output_queue = Queue()


# --- Graceful Shutdown Setup ---
shutdown_event = asyncio.Event()
agent_task = None
text_handler_task = None


@app.on_event("startup")
async def on_startup():
    global agent_task, text_handler_task
    logging.info("Launching agent and text handler background tasks...")
    agent_task = asyncio.create_task(agent_loop())
    text_handler_task = asyncio.create_task(text_input_handler_loop())


@app.on_event("shutdown")
async def on_shutdown():
    global agent_task, text_handler_task
    logging.info("Shutdown event received. Signaling agent to stop.")
    shutdown_event.set()
    
    tasks_to_await = [task for task in (agent_task, text_handler_task) if task]
    if tasks_to_await:
        try:
            await asyncio.wait_for(asyncio.gather(*tasks_to_await), timeout=5.0)
        except asyncio.TimeoutError:
            logging.warning("Tasks did not finish in time, cancelling.")
            for task in tasks_to_await:
                task.cancel()
        except asyncio.CancelledError:
            logging.info("Tasks were cancelled during shutdown.")
    logging.info("Shutdown complete.")


# --- Text Input Handler ---
async def text_input_handler_loop():
    while not shutdown_event.is_set():
        try:
            # Use a timeout to allow the loop to check for the shutdown event
            text_data = await asyncio.wait_for(text_input_queue.get(), timeout=1.0)
            logging.info(f"TEXT_HANDLER: Received text from client: {text_data}")
            # This is a placeholder for future text-based command handling
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            logging.info("Text input handler loop cancelled.")
            break
    logging.info("Text input handler loop has stopped.")


# --- Background Agent Task (Refactored) ---
async def agent_loop():
    conversation_state = {"messages": []}

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
            inputs = {
                "transcribed_text": transcript,
                "messages": conversation_state.get("messages", [])
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
            output_audio = await synthesize_speech(llm_response)

            # 5. Send synthesized audio to the client
            await client_output_queue.put(output_audio)

        except (json.JSONDecodeError, AttributeError) as e:
            logging.warning(f"Could not parse STT message: {stt_message_str} ({e})")
            continue
        except asyncio.CancelledError:
            logging.info("Agent loop cancelled.")
            break
    logging.info("Agent loop has stopped.")


# --- WebSocket Endpoint (Refactored) ---
@app.websocket("/ws")
async def websocket_endpoint(client_ws: WebSocket):
    await client_ws.accept()
    logging.info("Client connected.")

    # These tasks bridge the websocket and the queues
    async def client_reader():
        try:
            while True:
                # For now, we only process binary audio data as per original implementation.
                # To handle text, the client would need to send it, and we could use
                # `message = await client_ws.receive()` to inspect message type.
                audio_chunk = await client_ws.receive_bytes()
                await binary_input_queue.put(audio_chunk)
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


# --- Reasoning Engine (Unchanged) ---
reasoning_engine = create_reasoning_engine()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000)