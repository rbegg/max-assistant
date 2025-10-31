# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
This module defines tools for interacting with the Neo4j database.
"""
import logging
from langchain_core.tools import tool
from pydantic.v1 import BaseModel, Field
from src.clients.neo4j_client import neo4j_client


# Define the input schema for the tool using Pydantic for validation and clarity.
class GetScheduleSummaryInput(BaseModel):
    """Input schema for get_schedule_summary."""
    username: str = Field(description="The name of the user for whom to fetch the schedule summary.")


@tool(args_schema=GetScheduleSummaryInput)
async def get_schedule_summary(username: str) -> str:
    """
    Fetches the daily schedule summary for a given user.
    If a schedule is found, it will return the schedule summary as a string.
    If no schedule is found, it will return the string 'No schedule found.'.
    When you receive this, you MUST inform the user that they do not have a schedule.
    """
    logging.info(f"Fetching schedule summary for user: {username}")
    return await neo4j_client.get_schedule_summary(username)