# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
This module implements the Model Context Protocol for gathering dynamic context for the LLM.
"""

import datetime
import logging
from src.tools.neo4j import neo4j_client


async def get_dynamic_context(username: str) -> dict:
    """
    Gathers all dynamic context required by the LLM prompt based on the Model Context Protocol.
    """
    logging.info(f"MCP: Gathering context for {username}")

    current_time = datetime.datetime.now().strftime("%A, %B %d, %Y, %I:%M:%S %p")
    schedule_summary = await neo4j_client.get_schedule_summary(username)
    location = "Guelph, Ontario, Canada"  # This could also be fetched dynamically

    context = {
        "user_name": username,
        "location": location,
        "current_time": current_time,
        "schedule_summary": schedule_summary,
    }
    logging.info("MCP: Context gathered.")
    return context
