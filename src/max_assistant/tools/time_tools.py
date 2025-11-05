# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
This module defines a tool for getting the current time.
"""
import logging

from langchain_core.tools import tool
from pydantic import BaseModel

from max_assistant.utils.datetime_utils import current_datetime


class GetCurrentDateTimeInput(BaseModel):
    """Input schema for get_current_time."""
    pass


@tool(args_schema=GetCurrentDateTimeInput)
async def get_current_datetime() -> dict:
    """
    Returns the current date and time as an ISO formatted string, without seconds.
    Use this tool whenever the user asks for the current time, date or queries about today, tomorrow, etc.
    This tool does not take any parameters.
    """
    logging.info("Fetching current time")
    return current_datetime()
