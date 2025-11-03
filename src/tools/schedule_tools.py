# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
Defines Pydantic models for the schedule-related tools and neo4j nodes
"""
import json
import asyncio
from typing import Optional, Type

from langchain_core.tools import tool
from pydantic import ValidationError, BaseModel

from src.clients import neo4j_client
# Import our new Pydantic models
from src.models.schedule_models import (
    Appointment,
    DailyRoutine,
    CreateAppointmentArgs
)

import logging
logger = logging.getLogger(__name__)


async def _query_and_validate_nodes(
        query: str,
        params: dict,
        model_class: Type[BaseModel],
        result_key: str
) -> str:
    """
    Private helper to execute a query, validate results against a
    Pydantic model, and return a JSON string.
    """
    logger.debug(f"Executing query with params: {params} for model: {model_class.__name__}")
    result = await neo4j_client.client.execute_query(query, params)

    # --- Pydantic Validation Step ---
    if "error" in result:
        return json.dumps(result)

    try:
        # Generic: extract nodes using the provided result_key
        raw_nodes = [item[result_key] for item in result.get("data", [])]

        # Generic: validate against the provided model_class
        validated_nodes = [model_class.model_validate(node) for node in raw_nodes]

        # Convert validated models to JSON
        return json.dumps([node.model_dump(mode='json') for node in validated_nodes], indent=2)

    except ValidationError as e:
        logger.error(f"Validation error for {model_class.__name__}: {e.errors()}")
        return json.dumps({"error": "Data validation failed", "details": e.errors()})
    except KeyError:
        logger.error(f"Validation: Unexpected data structure from DB. Expected key '{result_key}'.")
        return json.dumps({"error": "Data parsing failed",
                           "details": f"Unexpected data structure from DB. Missing key: {result_key}"})
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return json.dumps({"error": "Data parsing failed", "details": "Unexpected error."})

@tool
async def get_appointments_for_date(target_date: str) -> str:
    """
    Use this tool if the user asks specifically for 'appointments'
    and NOT for their general 'schedule'. For general schedule
    questions, use 'get_full_schedule'.
    The target_date arg must be a valid iso date string.
    """
    logger.info(f"Tool: get_appointments_for_date for {target_date}")
    query = """
            WITH datetime($targetDate) AS dt
                OPTIONAL MATCH (d: Day {year : dt.year, month : dt.month, day : dt.day})
                OPTIONAL MATCH (d)-[:HAS_APPOINTMENT]->(appt:Appointment)
            WITH appt 
            WHERE appt IS NOT NULL
                RETURN properties(appt) AS appointment 
            """

    params = {"targetDate": target_date}

    # Call the same helper with a different model and key
    return await _query_and_validate_nodes(
        query,
        params,
        model_class=Appointment,
        result_key="appointment"
    )


@tool
async def get_routines_for_date(target_date: str) -> str:
    """
    Use this tool if the user asks specifically for 'activities', 'daily routines',
    'meal times', or 'medication times'. For general schedule
    questions, use 'get_full_schedule'. The target_date arg must be a valid iso date string.
    """
    logger.info(f"Tool: get_routines_for_date for {target_date}")
    query = """
            WITH datetime($targetDate) AS dt
            WITH CASE dt.dayOfWeek
                WHEN 1 THEN 'Monday' WHEN 2 THEN 'Tuesday' WHEN 3 THEN 'Wednesday'
                WHEN 4 THEN 'Thursday' WHEN 5 THEN 'Friday' WHEN 6 THEN 'Saturday'
                ELSE 'Sunday' \
            END AS dowString
            MATCH (u:User)
            OPTIONAL MATCH (u)-[:ATTENDS]->(routine:DailyRoutine)
            WHERE dowString IN routine.dayOfWeek
            WITH routine WHERE routine IS NOT NULL
            RETURN properties(routine) AS routine \
            """

    params = {"targetDate": target_date}

    # Call the helper
    return await _query_and_validate_nodes(
        query,
        params,
        model_class=DailyRoutine,
        result_key="routine"
    )


@tool
async def get_full_schedule(target_date: str) -> str:
    """
    Use this as the DEFAULT tool for any general schedule question,
    like 'What's my schedule?', 'What's on today?', or 'Am I busy tomorrow?'.
    It combines appointments and daily routines into a single, sorted list.
    The target_date arg must be a valid iso date string.
    """
    logger.info(f"Tool: get_full_schedule for {target_date}")

    try:
        # Run both tools at the same time
        results = await asyncio.gather(
            get_appointments_for_date.ainvoke(target_date),
            get_routines_for_date.ainvoke(target_date,)
        )

        appts_json_str, routines_json_str = results

        # Parse the JSON results from both tools
        appts = json.loads(appts_json_str)
        routines = json.loads(routines_json_str)

        # Check if either tool returned an error
        if isinstance(appts, dict) and 'error' in appts:
            logger.error(f"Error from get_appointments: {appts}")
            return json.dumps(appts)
        if isinstance(routines, dict) and 'error' in routines:
            logger.error(f"Error from get_routines: {routines}")
            return json.dumps(routines)

        # Combine the two lists
        combined_list = []
        for item in appts + routines:
            combined_list.append({key: item[key] for key in ['time', 'title', 'duration', 'details']})

        # Sort the combined list by 'time'
        # We use .get() with a default to prevent errors if a key is missing
        sorted_list = sorted(
            combined_list,
            key=lambda x: x.get('time', '23:59:59')
        )

        logger.info(f"Returning combined schedule with {len(sorted_list)} items.")
        return json.dumps(sorted_list, indent=2)

    except Exception as e:
        # Catch-all for any other errors (like JSON parsing)
        error_type = e.__class__.__name__
        logger.error(f"Unexpected error in get_full_schedule: {e}", exc_info=True)
        return json.dumps({"error": f"Internal error combining schedule: {error_type}"})

@tool(args_schema=CreateAppointmentArgs)
async def create_appointment(
        title: str,
        date: str,
        details: Optional[str] = None,
        duration: Optional[int] = None
) -> str:
    """
    Creates a new appointment for the user.
    """
    # LangChain/LangGraph will automatically validate the LLM's
    # input using the 'CreateAppointmentArgs' type hint on the decorator.

    print(f"Tool: Calling create_appointment with args: {title}")

    # We can reliably build the params dict for our query
    params = {
        "title": title,
        "date": date,
        "details": details,
        "duration": duration
    }

    query = """
            WITH date($date) AS dt
            MERGE (d:Day {year: dt.year, month: dt.month, day: dt.day})
            CREATE (a:Appointment {
                id: randomUUID(),
                title: $title,
                date: $date,
                details: $details,
                duration: $duration
            })
            MERGE (d)-[:HAS_APPOINTMENT]->(a)
            RETURN a.id AS new_appointment_id
            """

    # The client returns a dict, so we just dump it to a string for the LLM
    result = await neo4j_client.client.execute_query(query, params)
    return json.dumps(result, indent=2)

@tool
async def get_activities_info( ) -> str:
    """
    Use this tool to get details about available Activities.
    The rating is an indicator of how much the user dislikes or enjoys an activity, favorites should include enjoy and
    ok ratings but exclude dislike.  If responding with details about a disliked activity, the users rating should be
    gently acknowledged, for example:  "There is bingo at 7 pm, but I know you dislike it."
    Pay attention to the day of the week the activity is offered and the current date.
    Use this to answer questions about activities like 'What are my favorite activities?' or to suggest activities.
    """
    logger.info(f"Tool: get_routine_info for type")

    query = """
            MATCH (u:User)
            OPTIONAL MATCH (u)-[:ATTENDS]->(routine:DailyRoutine)
            WHERE routine.type in $type_list
            WITH routine WHERE routine IS NOT NULL
            RETURN properties(routine) AS routine 
            """

    params = {"type_list": ['activity', 'exercise']}

    return await _query_and_validate_nodes(
        query,
        params,
        model_class=DailyRoutine,
        result_key="routine"
    )
