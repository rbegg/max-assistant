import json
import os
import uvicorn
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket
from .config import PORT, LOG_LEVEL, LOG_FORMAT, HOST
from max_assistant.connection_manager import ConnectionManager
from max_assistant.agent.graph import create_reasoning_engine
from max_assistant.clients.neo4j_client import Neo4jClient

# Compiled reasoning engine.
reasoning_engine = None
reasoning_engine_preload_task = None


def setup_logging(config_path='log_config.json'):
    """Loads logging configuration from a JSON file."""

    # Make sure the 'logs' directory exists (if using a file handler)
    os.makedirs('logs', exist_ok=True)

    try:
        with open(config_path, 'rt') as f:
            config = json.load(f)

        # Apply the configuration
        logging.config.dictConfig(config)

    except FileNotFoundError:
        print(f"Error: Config file '{config_path}' not found. Using basic config.")
        logging.basicConfig(level=logging.INFO)
    except json.JSONDecodeError:
        print(f"Error: Could not parse '{config_path}'. Using basic config.")
        logging.basicConfig(level=logging.INFO)
    except Exception as e:
        print(f"An error occurred during logging setup: {e}")
        logging.basicConfig(level=logging.INFO)


setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's startup logic.
    """
    global reasoning_engine_preload_task
    logging.info("Application startup...")

    logging.info("Initializing the reasoning engine...")
    reasoning_engine_preload_task = asyncio.create_task(create_reasoning_engine())

    yield

    logging.info("Closing Neo4j client connection...")
    await Neo4jClient.close()
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

    global reasoning_engine
    if not reasoning_engine:
        logging.info("Waiting for reasoning engine to be ready...")
        reasoning_engine = await reasoning_engine_preload_task
        if not reasoning_engine:
            logging.error("Reasoning engine not available.")
            await client_ws.close(code=1011, reason="Server error: Reasoning engine not initialized.")
            return
        logging.info("Reasoning engine is ready.")

    manager = ConnectionManager(reasoning_engine, client_ws)
    try:
        await manager.handle_connection()
    except Exception as e:
        logging.error(f"Connection handler failed: {e}", exc_info=True)
    finally:
        logging.info("Client connection handler finished.")


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
