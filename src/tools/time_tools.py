# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
This module defines a tool for getting the current time.
"""
import logging
from langchain_core.tools import tool
from pydantic.v1 import BaseModel
from datetime import datetime

class GetCurrentTimeInput(BaseModel):
    """Input schema for get_current_time."""
    pass

@tool(args_schema=GetCurrentTimeInput)
def get_current_time() -> str:
    """
    Returns the current date and time as a formatted string.
    Use this tool whenever the user asks for the current time or date.
    This tool does not take any parameters.
    """
    logging.info("Fetching current time")
    return datetime.now().strftime("%I:%M %p on %A, %B %d, %Y")