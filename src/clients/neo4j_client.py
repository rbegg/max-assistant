# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
This module provides a client for interacting with the Neo4j database.
"""
import logging
from neo4j import AsyncGraphDatabase
from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD


class Neo4jClient:
    """A client for interacting with a Neo4j database."""

    def __init__(self, uri, user, password):
        print(f"URI = {uri}, USER = {user}")
        self._driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def close(self):
        """Closes the database connection."""
        await self._driver.close()

    async def get_appointments(self, userid: int, date_str: str):
        logging.info(f"Fetching appointments for date: {date_str}")
        query = """
            WITH date($targetDate) AS dt
            OPTIONAL MATCH (d:Day {year: dt.year, month: dt.month, day: dt.day})
            // If the day was found, optionally find all appointments for that day.
            OPTIONAL MATCH (d)-[:HAS_APPOINTMENT]->(appt:Appointment)
            WITH appt
            WHERE appt IS NOT NULL
            // Now, collect the properties of the non-null appointments
            WITH collect(properties(appt)) AS appointmentsList          
            // Return the list wrapped in a single JSON object
            RETURN { appointments: appointmentsList } AS jsonData
            """
        try:
            records, _, _ = await self._driver.execute_query(query, targetDate=date_str)
            if records:
                logging.debug(f"Appointments: {records[0]['jsonData']['appointments']}")
                return records[0]["jsonData"]["appointments"]
            return []
        except Exception as e:
            logging.error(f"Could not retrieve appointments for {date_str}: {e}", exc_info=True)
            return []

    async def get_daily_routines(self, userid: int, date_str: str) -> str:
        """Fetches a user's schedule from Neo4j and returns it as a string."""
        logging.info(f"Fetching schedule for user: {userid}")
        query = """
            WITH date($targetDate) AS dt
            WITH CASE dt.dayOfWeek
                   WHEN 1 THEN 'Monday'
                   WHEN 2 THEN 'Tuesday'
                   WHEN 3 THEN 'Wednesday'
                   WHEN 4 THEN 'Thursday'
                   WHEN 5 THEN 'Friday'
                   WHEN 6 THEN 'Saturday'
                   WHEN 7 THEN 'Sunday'
                 END AS dowString
            MATCH (u:User)
            OPTIONAL MATCH (u)-[:ATTENDS]->(routine:DailyRoutine)
            WHERE dowString IN routine.dayOfWeek
            WITH routine
            WHERE routine IS NOT NULL
            WITH collect(properties(routine)) AS routinesList
            RETURN { routines: routinesList } AS jsonData
        """
        try:
            result = await self._driver.execute_query(query, targetDate=date_str)
            if result.records:
                json_data = result.records[0]["jsonData"]
                logging.debug(f"Daily Routines {json_data}")
                return json_data
            return "No schedule found."
        except Exception as e:
            logging.error(f"Could not retrieve Daily Routines for {userid}: {e}", exc_info=True)
            return "No schedule found."


# Singleton instance of the client
neo4j_client = Neo4jClient(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
