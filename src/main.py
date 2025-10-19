import uvicorn
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket
from .config import PORT, LOG_LEVEL, LOG_FORMAT, HOST
from .connection_manager import ConnectionManager
from .graph import create_reasoning_engine

# --- Configuration ---
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)


# Compiled reasoning engine.
reasoning_engine = None




@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's startup logic.
    """
    global reasoning_engine
    logging.info("Application startup...")

    logging.info("Initializing the reasoning engine...")
    reasoning_engine = await create_reasoning_engine()
    if not reasoning_engine:
        raise RuntimeError("Failed to initialize the reasoning engine.")
    logging.info("Reasoning engine is ready.")

    yield

    logging.info("Application shutdown.")


app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health_check():
    """
    Checks if the application is healthy.
    """
    return {"status": "healthy"}


@app.websocket("/ws")
async def websocket_endpoint(client_ws: WebSocket):
    await client_ws.accept()
    logging.info("Client connected.")

    if not reasoning_engine:
        logging.error("Reasoning engine not available.")
        await client_ws.close(code=1011, reason="Server error: Reasoning engine not initialized.")
        return

    manager = ConnectionManager(reasoning_engine, client_ws)
    try:
        await manager.handle_connection()
    except Exception as e:
        logging.error(f"Connection handler failed: {e}", exc_info=True)
    finally:
        logging.info("Client connection handler finished.")


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)