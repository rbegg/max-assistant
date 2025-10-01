import uvicorn
import asyncio
import logging
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .graph import create_reasoning_engine
from .stt_service import transcript_generator
from .tts_service import synthesize_speech

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = FastAPI()

# --- Graceful Shutdown Setup ---
# 1. Create a global event to signal shutdown
shutdown_event = asyncio.Event()
# 2. A global placeholder to hold our background task
agent_task = None


# --- App Events (Startup and Shutdown) ---
@app.on_event("startup")
async def on_startup():
    """Launches the agent loop as a background task."""
    global agent_task
    logging.info("Launching agent background task...")
    # 3. Create and store the background task
    agent_task = asyncio.create_task(agent_loop())


@app.on_event("shutdown")
async def on_shutdown():
    """Gracefully stops the background agent task."""
    global agent_task
    logging.info("Shutdown event received. Signaling agent to stop.")

    # 4. Signal the agent loop to stop
    shutdown_event.set()

    # 5. Wait for the task to finish or cancel it
    if agent_task:
        try:
            # Give the task a moment to finish its current work
            await asyncio.wait_for(agent_task, timeout=5.0)
        except asyncio.TimeoutError:
            logging.warning("Agent task did not finish in time, cancelling.")
            agent_task.cancel()
        except asyncio.CancelledError:
            logging.info("Agent task was cancelled.")
    logging.info("Shutdown complete.")


# --- Background Agent Task ---
async def agent_loop():
    """The agent's main loop, now aware of the shutdown signal."""
    # 6. The loop now checks the shutdown event on each iteration
    while not shutdown_event.is_set():
        try:
            # This is a placeholder for your agent's actual work cycle.
            # In a real scenario, this would involve waiting on a queue.
            # We add a small sleep to prevent this loop from pegging the CPU.
            await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            # This allows the task to exit cleanly if it's cancelled
            logging.info("Agent loop cancelled.")
            break
    logging.info("Agent loop has stopped.")


# --- WebSocket Endpoint (Unchanged) ---
@app.websocket("/ws")
async def websocket_endpoint(client_ws: WebSocket):
    await client_ws.accept()
    logging.info("Client connected.")
    try:
        async for stt_message_str in transcript_generator(client_ws):
            # ... (rest of your WebSocket logic remains the same)
            try:
                stt_response = json.loads(stt_message_str)
                transcript = stt_response.get("data", "").strip()
                if not transcript: continue

                await client_ws.send_text(stt_message_str)
                initial_state = {"transcribed_text": transcript}
                final_state = await reasoning_engine.ainvoke(initial_state)
                llm_response = final_state.get("llm_response", "")

                response_payload = {"data": llm_response, "source": "assistant"}
                await client_ws.send_text(json.dumps(response_payload))

                output_audio = await synthesize_speech(llm_response)
                await client_ws.send_bytes(output_audio)

            except (json.JSONDecodeError, AttributeError) as e:
                logging.warning(f"Could not parse STT message: {stt_message_str} ({e})")
                continue
    except WebSocketDisconnect:
        logging.info("Client disconnected.")
    except Exception as e:
        logging.error(f"Main WebSocket endpoint error: {e}", exc_info=True)
    finally:
        logging.info("Client connection handler finished.")


# --- Reasoning Engine (Unchanged) ---
reasoning_engine = create_reasoning_engine()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000)