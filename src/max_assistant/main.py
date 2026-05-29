import json
import os
import uvicorn
import asyncio
import logging
import logging.config
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket

from max_assistant.agent.session_mgr import session_manager
from max_assistant.config import PORT, HOST
from max_assistant.app_services import AppServices
from max_assistant.connection_manager import ConnectionManager
from max_assistant.tools.reminder_tools import ReminderTools

logger = logging.getLogger(__name__)


def setup_logging(config_path='log_config.json'):
    """Loads logging configuration from a JSON file."""
    os.makedirs('logs', exist_ok=True)
    try:
        with open(config_path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        print(f"Logging setup error ({type(e).__name__}): Fallback to basic configuration.")
        logging.basicConfig(level=logging.INFO)


# Trigger logging before building anything
setup_logging()


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """
    Manages the application's startup and shutdown logic safely.
    Uses app.state to avoid volatile global scope mutations.
    """
    logger.info("Application startup initiating...")
    fastapi_app.state.app_services = None
    fastapi_app.state.poller_task = None

    try:
        # 1. Initialize Service Singletons
        services = await AppServices.create()
        fastapi_app.state.app_services = services

        # 2. Extract and Bind Registry Providers
        reminder_tools_instance: ReminderTools | None = services.get_tool_provider(ReminderTools)

        if reminder_tools_instance:
            logger.info("Spawning background reminder poller task...")
            fastapi_app.state.poller_task = asyncio.create_task(
                reminder_tools_instance.start_reminder_poller_dynamic(
                    get_sessions_fn=session_manager.get_all_sessions,
                    get_active_user_ids_fn=session_manager.get_active_user_ids,
                    poll_interval_seconds=20
                )
            )
        else:
            logger.error("ReminderTools provider missing from registry! Background tracking unavailable.")

    except Exception as e:
        logger.critical(f"CRITICAL: Application initialization crashed: {e}", exc_info=True)
        # Flush log buffers before exiting process abnormally
        await asyncio.sleep(0.1)
        sys.exit(1)

    yield  # --- Application Execution Phase ---

    logger.info("Application shutting down, executing resource cleanup...")

    # 3. Clean and Safe Background Poller Disassembly
    if fastapi_app.state.poller_task:
        task: asyncio.Task = fastapi_app.state.poller_task
        if not task.done():
            logger.info("Cancelling active background reminder poller...")
            task.cancel()

        try:
            # Await completion or cancellation to harvest latent task exceptions
            await task
        except asyncio.CancelledError:
            logger.info("Reminder poller background task successfully cancelled.")
        except Exception as e:
            logger.error(f"Captured latent unhandled exception from reminder poller during shutdown: {e}",
                         exc_info=True)

    # 4. Defensive Database Client Termination
    if fastapi_app.state.app_services and fastapi_app.state.app_services.db_client:
        logger.info("Closing active Neo4j client connection pooling...")
        try:
            await fastapi_app.state.app_services.db_client.close()
        except Exception as e:
            logger.error(f"Error closing Neo4j connectivity pool safely: {e}", exc_info=True)

    logger.info("Application shutdown complete.")


# Pass the lifespan context safely into the application builder instance
app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health_check():
    """Validates the health and routing accessibility of the server instance."""
    return {"status": "healthy"}


@app.websocket("/ws")
async def websocket_endpoint(client_ws: WebSocket):
    """Handles WebSocket transport layer upgrades and provisions request contexts."""
    # Directly access the property. Type checkers accept this clean guard sequence.
    services: AppServices | None = app.state.app_services

    await client_ws.accept()
    logger.info("Incoming WebSocket connection accepted.")

    # This type guard narrows the type down from 'AppServices | None' to just 'AppServices'
    if services is None or not services.reasoning_engine:
        logger.error("Rejecting connection: Core application singletons are uninitialized.")
        await client_ws.close(code=1011, reason="Server error: Services not initialized.")
        return

    logger.info("Provisioning isolated ConnectionManager infrastructure for client session.")
    # The IDE will now be 100% happy here!
    manager = ConnectionManager(services, client_ws)

    try:
        await manager.handle_connection()
    except Exception as e:
        logger.error(f"Fatal connection handler trap exception: {e}", exc_info=True)
    finally:
        logger.info("Client WebSocket transport cleanup sequence complete.")


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_config=None)