# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
This module sets and manages configuration settings for Max-Assistant.
Strictly maps and validates environment bindings without operational side effects.
"""
import os
import logging

# --- Helper Parsers ---

def _parse_int_env(key: str, default: int) -> int:
    """Safely converts environment variable to an integer with validation fallbacks."""
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        # Log to root logger directly since configuration isn't fully established yet
        logging.warning(f"Invalid integer configuration for '{key}'='{val}'. Defaulting to {default}.")
        return default


def _get_required_env(key: str, local_fallback: str) -> str:
    """
    Returns the environment variable. Optionally logs a notice if relying
    on local-only loops to flag production environment hazards.
    """
    val = os.getenv(key)
    if not val:
        # You can toggle this to raise a KeyError in strict production environments
        return local_fallback
    return val


def _parse_float_env(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        logging.warning(f"Invalid float configuration for '{key}'='{val}'. Defaulting to {default}.")
        return default

# --- Server Configuration ---

HOST = os.getenv("HOST", "127.0.0.1")
PORT = _parse_int_env("PORT", 9000)

# --- Dynamic Log Level Computation ---

log_level_str = os.getenv('LOG_LEVEL', 'INFO').upper()
LOG_LEVEL = logging.getLevelName(log_level_str)

if not isinstance(LOG_LEVEL, int):
    LOG_LEVEL = logging.INFO

# Note: BasicConfig was removed here to prevent clashing with main.py's dictConfig hook.

# --- Application Hyperparameters ---

TTS_VOICE = os.getenv("TTS_VOICE", "en_US-hfc_female-medium")
DEFAULT_USERNAME = os.getenv("DEFAULT_USERNAME", "User")
SHUTDOWN_TIMEOUT = _parse_float_env("SHUTDOWN_TIMEOUT", 5.0)
QUEUE_GET_TIMEOUT = _parse_float_env("QUEUE_GET_TIMEOUT", 1.0)
MESSAGE_PRUNING_LIMIT = _parse_int_env("MESSAGE_PRUNING_LIMIT", 10)

# --- Core Service Topologies (Ollama, Neo4j, STT) ---

OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3")
OLLAMA_BASE_URL = _get_required_env("OLLAMA_BASE_URL", "http://localhost:11434")

NEO4J_URI = _get_required_env("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

STT_WEBSOCKET_URL = os.getenv("STT_WEBSOCKET_URL", "ws://stt/ws")

# --- Integrations (OAuth / Credentials) ---

GOOGLE_SENDER_EMAIL = os.getenv("GOOGLE_SENDER_EMAIL", "")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")