import json
import os
import uvicorn
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket

from max_assistant.config import (
    PORT, LOG_LEVEL, LOG_FORMAT, HOST,
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,
    OLLAMA_MODEL_NAME, OLLAMA_BASE_URL
)
from max_assistant.clients.ollama_preloader import warm_up_ollama_async
from max_assistant.connection_manager import ConnectionManager
from max_assistant.agent.graph import create_reasoning_engine
from max_assistant.clients.neo4j_client import Neo4jClient

# Compiled reasoning engine.
reasoning_engine = None
reasoning_engine_preload_task = None
db_client = None


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
    global reasoning_engine, db_client
    logging.info("Application startup...")

    try:
        logging.info("Initializing the neo4j client and the llm ...")

        # Run both async functions at the same time
        results = await asyncio.gather(
            Neo4jClient.create(
                NEO4J_URI,
                NEO4J_USERNAME,
                NEO4J_PASSWORD
            ),
            warm_up_ollama_async(
                OLLAMA_MODEL_NAME,
                OLLAMA_BASE_URL,
                temperature=0
            )
        )

        # Unpack the results
        db_client = results[0]
        llm = results[1]

        if not db_client:
            raise RuntimeError("Failed to initialize Neo4j client.")
        if not llm:
            raise RuntimeError("Failed to initialize the LLM.")

        logging.info("Successfully initialized Neo4j client and LLM.")

        # 3. Create the reasoning engine with the dependencies
        logging.info("Initializing the reasoning engine...")
        reasoning_engine = await create_reasoning_engine(db_client, llm)

    except Exception as e:
        logging.critical(f"Failed to initialize application: {e}", exc_info=True)
        # This will prevent the app from starting if init fails
        raise e

    yield

    # Shutdown logic: This code runs after the server is stopped
    logging.info("Closing Neo4j client connection...")
    if db_client:
        await db_client.close()
    logging.info("Application shutdown complete.")

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
