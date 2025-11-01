# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
This module defines tools for interacting with the Neo4j database.
"""
import logging
from typing import Any, Coroutine

from langchain_core.tools import tool
from pydantic.v1 import BaseModel, Field
from src.clients.neo4j_client import neo4j_client


# Define the input schema for the tool using Pydantic for validation and clarity.
class GetScheduleSummaryInput(BaseModel):
    """Input schema for get_schedule_summary."""
    userid: str = Field(description="The id of user for whom to fetch the schedule summary.")
    date_str: str = Field(description="The date in YYYY-MM-DD format for which to fetch the schedule summary.")


@tool(args_schema=GetScheduleSummaryInput)
async def get_schedule_summary(userid: str, date_str: str) -> list[str]:
    """
    Fetches the daily schedule summary for a given user.
    If a schedule is found, it will return the schedule summary as a string.
    If no schedule is found, it will return the string 'No schedule found.'.
    When you receive this, you MUST inform the user that they do not have a schedule.
    """
    logging.info(f"Fetching schedule summary for userID: {userid}")
    logging.info("Getting Appointments")
    appointments = await neo4j_client.get_appointments(userid, date_str)
    logging.info(f"Appointments: {appointments}")
    logging.info("Getting Daily Routines")
    daily_routines = await neo4j_client.get_daily_routines(userid, date_str)
    logging.info(f"Daily Routines: {daily_routines}")

    schedule = []
    for item in appointments:
        schedule.append(f" Appointment for: {item["title"]} at {item["startTime"]}")

    for item in daily_routines["routines"]:
        schedule.append(f"{item["title"]} at {item["startTime"]}")

    return schedule
