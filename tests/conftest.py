import json
import logging.config
import os
from pathlib import Path
import pytest


# Resolve the absolute path to log_config.json based on this conftest.py file location
# Path(__file__).parent is services/max-assistant/tests/
# Path(__file__).parent.parent is services/max-assistant/
ASSISTANT_DIR = Path(__file__).parent.parent
LOG_CONFIG_PATH = ASSISTANT_DIR / "log_config.json"


@pytest.fixture(scope="session", autouse=True)
def configure_logging_from_json():
    """Programmatically applies log_config.json to the entire pytest session with path resolution."""
    if LOG_CONFIG_PATH.exists():
        try:
            with open(LOG_CONFIG_PATH, "r") as f:
                config = json.load(f)

            # --- DYNAMIC PATH RESOLUTION ---
            # Resolve relative filenames to absolute paths relative to the assistant directory
            for handler_name, handler_config in config.get("handlers", {}).items():
                if "filename" in handler_config:
                    relative_filename = handler_config["filename"]
                    absolute_filename = ASSISTANT_DIR / relative_filename

                    # Convert to string path
                    handler_config["filename"] = str(absolute_filename.resolve())

                    # Ensure the parent directory (e.g. services/max-assistant/logs/) exists
                    os.makedirs(absolute_filename.parent, exist_ok=True)

            # Apply the fully resolved dictConfig
            logging.config.dictConfig(config)
            print(f"\n[TEST SETUP] Successfully applied logging configuration from {LOG_CONFIG_PATH}")
            print(f"[TEST SETUP] Log files will be written to: {ASSISTANT_DIR / 'logs'}")
        except Exception as e:
            print(f"\n[TEST SETUP] Warning: Failed to parse logging config: {e}")
    else:
        print(f"\n[TEST SETUP] Warning: log_config.json not found at {LOG_CONFIG_PATH}. Using standard fallback.")