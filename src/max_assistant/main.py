import json
import os
import uvicorn
import asyncio
import logging
import logging.config
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket

from max_assistant.config import (
    PORT, HOST,
)

from max_assistant.app_services import AppServices
from max_assistant.connection_manager import ConnectionManager

logger = logging.getLogger(__name__)

app_services: AppServices = None


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
async def lifespan(_: FastAPI):
    """
    Manages the application's startup logic.
    """
    global app_services
    logger.info("Application startup...")

    try:
        app_services = await AppServices.create()
        logger.info("Application services successfully initialized.")

    except Exception as e:
        logger.critical(f"Failed to initialize application: {e}", exc_info=True)
        # This will prevent the app from starting if init fails
        raise e


    yield

    # Shutdown logic: This code runs after the server is stopped
    logger.info("Closing Neo4j client connection...")
    if app_services and app_services.db_client:
        await app_services.db_client.close()
    logger.info("Application shutdown complete.")


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
    logger.info("Client connected.")

    global app_services

    if not app_services or not app_services.reasoning_engine:
        logger.error("Server not fully initialized: missing services.")
        await client_ws.close(code=1011, reason="Server error: Not initialized.")
        return
    logger.info("Reasoning engine and services are ready.")

    manager = ConnectionManager(
        app_services,
        client_ws
    )
    try:
        await manager.handle_connection()
    except Exception as e:
        logger.error(f"Connection handler failed: {e}", exc_info=True)
    finally:
        logger.info("Client connection handler finished.")


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
