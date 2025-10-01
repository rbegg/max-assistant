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

# --- Graceful Shutdown Setup (Unchanged) ---
shutdown_event = asyncio.Event()
agent_task = None


@app.on_event("startup")
async def on_startup():
    global agent_task
    logging.info("Launching agent background task...")
    agent_task = asyncio.create_task(agent_loop())


@app.on_event("shutdown")
async def on_shutdown():
    global agent_task
    logging.info("Shutdown event received. Signaling agent to stop.")
    shutdown_event.set()
    if agent_task:
        try:
            await asyncio.wait_for(agent_task, timeout=5.0)
        except asyncio.TimeoutError:
            logging.warning("Agent task did not finish in time, cancelling.")
            agent_task.cancel()
        except asyncio.CancelledError:
            logging.info("Agent task was cancelled.")
    logging.info("Shutdown complete.")


# --- Background Agent Task (Unchanged) ---
async def agent_loop():
    while not shutdown_event.is_set():
        try:
            await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            logging.info("Agent loop cancelled.")
            break
    logging.info("Agent loop has stopped.")


# --- WebSocket Endpoint (Updated) ---
@app.websocket("/ws")
async def websocket_endpoint(client_ws: WebSocket):
    await client_ws.accept()
    logging.info("Client connected.")

    # CHANGE 1: Initialize conversation state for the duration of this connection
    conversation_state = {"messages": []}

    try:
        async for stt_message_str in transcript_generator(client_ws):
            try:
                stt_response = json.loads(stt_message_str)
                transcript = stt_response.get("data", "").strip()
                if not transcript:
                    continue

                await client_ws.send_text(stt_message_str)

                # CHANGE 2: Update the input to include the current conversation state
                inputs = {
                    "transcribed_text": transcript,
                    "messages": conversation_state.get("messages", [])
                }

                # The engine now receives the history and returns the updated state
                final_state = await reasoning_engine.ainvoke(inputs)

                # CHANGE 3: Persist the updated state for the next turn
                conversation_state = final_state

                # CHANGE 4: Extract the response from the last message in the list
                llm_response = ""
                if final_state.get("messages") and len(final_state["messages"]) > 0:
                    last_message = final_state["messages"][-1]
                    llm_response = last_message.content

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