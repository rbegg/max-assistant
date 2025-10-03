# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
This module sets and manages configuration settings for Max-Assistant.
"""
import os
import logging

# --- Server Configuration for local development ---

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "9000"))

# --- Logging Configuration ---

LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# Standard levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
# Default level: INFO
log_level_str = os.getenv('LOG_LEVEL', 'info').upper()
LOG_LEVEL = logging.getLevelName(log_level_str)

# Validate LOG_LEVEL
if not isinstance(LOG_LEVEL, int):
    logging.warning(f"Invalid log level '{log_level_str}'. Defaulting to INFO.")
    LOG_LEVEL = logging.INFO

# --- Application Configuration ---

TTS_VOICE = os.environ.get("TTS_VOICE", "en_US-hfc_female-medium")
DEFAULT_USERNAME = os.getenv("DEFAULT_USERNAME", "User")
SHUTDOWN_TIMEOUT = float(os.getenv("SHUTDOWN_TIMEOUT", "5.0"))
QUEUE_GET_TIMEOUT = float(os.getenv("QUEUE_GET_TIMEOUT", "1.0"))

# Validate MESSAGE_PRUNING_LIMIT
limit_str = (os.getenv("MESSAGE_PRUNING_LIMIT", 10))
MESSAGE_PRUNING_LIMIT = int(limit_str)
if not isinstance(MESSAGE_PRUNING_LIMIT, int):
    MESSAGE_PRUNING_LIMIT = 10
    logging.warning(f"Invalid Message Pruning Limit '{limit_str}'. Defaulting to {MESSAGE_PRUNING_LIMIT}")
