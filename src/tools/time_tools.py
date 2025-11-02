# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
This module defines a tool for getting the current time.
"""
import logging
from langchain_core.tools import tool
from pydantic import BaseModel
from datetime import datetime

class GetCurrentDateTimeInput(BaseModel):
    """Input schema for get_current_time."""
    pass

@tool(args_schema=GetCurrentDateTimeInput)
async def get_current_datetime() -> dict:
    """
    Returns the current date and time as an ISO formatted string, without seconds.
    Use this tool whenever the user asks for the current time or date.
    This tool does not take any parameters.
    """
    logging.info("Fetching current time")
    dt_str = datetime.now().strftime("%Y-%m-%dT%H:%M")
    day = datetime.now().isoweekday()
    month = datetime.now().strftime("%B")
    year = datetime.now().year
    return {'ISODateTime': dt_str, 'Day': day, 'Month': month, 'Year': year}
