# Copyright (c) 2026, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
Defines LangGraph tools for scheduling reminders using task nodes and saving them to Neo4j.
"""
from datetime import datetime, timedelta
import json
import asyncio
import logging
from typing import Annotated

from langchain_core.tools import StructuredTool
from langgraph.prebuilt import InjectedState

from max_assistant.tools.registry import BaseToolProvider
from max_assistant.clients.neo4j_client import Neo4jClientError, Neo4jCircuitBreakerError
from max_assistant.utils.decorators import requires_db


logger = logging.getLogger(__name__)

class ReminderTools(BaseToolProvider):
    """
    A class that encapsulates reminder and task scheduling tools
    backed by a Neo4j client instance.
    """

    def __init__(self, db_client, llm=None):
        super().__init__(db_client, llm)
        logger.debug("ReminderTools initialized with a Neo4j client.")

    @requires_db
    async def schedule_reminder(
            self,
            message: str,
            delay_minutes: float,
            user_info: Annotated[dict, InjectedState("userinfo")]
        ) -> str:
        """
        Use this tool to schedule a reminder message after a certain number of minutes.
        """
        logger.info(f"Tool: schedule_reminder - message='{message}', delay={delay_minutes}m")

        user_id = self._get_verified_user_id(user_info)

        try:
            # Calculate the absolute execution time
            due_datetime = datetime.now() + timedelta(minutes=delay_minutes)
            due_iso = due_datetime.isoformat()

            query = """
                MATCH (u:User {id: $user_id}) 
                WITH u, datetime($due_time) AS d
                CREATE (t:Task {
                    id: randomUUID(),
                    text: $text,
                    due_time: d,
                    type: 'REMINDER',
                    status: 'PENDING'
                })
                MERGE (u)-[:HAS_YEAR]->(y:Year {year: d.year})
                MERGE (y)-[:HAS_MONTH]->(m:Month {month: d.month, year: d.year})
                ON CREATE SET m.name = format(d, 'MMM') 
                MERGE (m)-[:HAS_DAY]->(day:Day {day: d.day, month: d.month, year: d.year})
                MERGE (day)-[:HAS_TASK]->(t)
                RETURN t.id AS task_id
            """
            params = {
                "text": message,
                "due_time": due_iso,
                "user_id": user_id,
            }

            result = await self.db_client.execute_query(query, params)

            data = result.get("data", [])
            if not data:
                logger.error("Failed to schedule reminder: Could not match User ID in database.")
                return json.dumps({
                    "error": "User_Not_Found",
                    "instruction": "Could not attach the reminder because your user profile was not found in the database. Please inform the user."
                })

            # 4. Extract and use the task_id
            task_id = data[0].get("task_id")

            return json.dumps({
                "success": True,
                "task_id": task_id,
                "message": f"Successfully scheduled reminder: '{message}' for {delay_minutes} minutes from now.",
                "due_time": due_iso
            }, indent=2)

        except Neo4jCircuitBreakerError as e:
            logger.warning(f"Circuit Breaker blocked reminder scheduling: {e}")
            return json.dumps({
                "error": "Database_Offline_Circuit_Open",
                "instruction": "The system database is currently offline. The reminder could NOT be saved. Apologize and inform the user.",
                "details": str(e)
            })

        except Neo4jClientError as e:
            logger.error(f"Database error in schedule_reminder: {e}")
            return json.dumps({
                "error": "Database_Unavailable",
                "instruction": "The database connection failed abruptly. Apologize to the user and inform them that the reminder could not be saved right now.",
                "details": str(e)
            })

        except Exception as e:
            logger.error(f"Unexpected error in schedule_reminder tool: {e}")
            return json.dumps({"error": "Failed to schedule reminder", "details": str(e)})

    def get_tools(self) -> list:
        """
        Returns a list of all tool methods bound to this instance.
        """
        return [
            StructuredTool.from_function(
                func=None,
                coroutine=self.schedule_reminder,
                name="schedule_reminder",
                description=self.schedule_reminder.__doc__,
                handle_tool_error=self.format_system_tool_error,
            )
        ]

    async def start_reminder_poller_dynamic(self, get_sessions_fn, get_active_user_ids_fn, poll_interval_seconds: int = 20):
        """
        A continuous, non-blocking background loop that polls Neo4j for pending reminders for all users
        and routes them to the correct user session.
        """
        logger.info("Background reminder poller service activated.")

        # Language=cypher
        check_query = """
                      WITH datetime($now) AS now
                          MATCH (u: User)-[:HAS_YEAR]->()-[:HAS_MONTH]->()-[:HAS_DAY]->()-[:HAS_TASK]->(t:Task)
                      WHERE t.due_time <= now 
                          AND u.id in $user_id_list
                          AND t.type = 'REMINDER'
                          AND t.status = 'PENDING'
                      SET t.status = 'COMPLETED'
                      RETURN t.text AS text, t.id AS id, u.id AS user_id
                      """

        while True:
            try:
                active_sessions = get_sessions_fn()
                active_user_ids = get_active_user_ids_fn()

                if not active_user_ids:
                    logger.debug("No active sessions online. Skipping reminder database poll.")
                    await asyncio.sleep(poll_interval_seconds)
                    continue

                params = {"now": datetime.now().isoformat(),
                          "user_id_list": active_user_ids
                          }

                # Execute the query to find pending tasks
                result = await self.db_client.execute_query(check_query, params)

                tasks_due = result.get("data", [])
                for task in tasks_due:
                    reminder_text = task.get("text")
                    task_id = task.get("id")
                    user_id = task.get("user_id")  # Get the owner's ID

                    logger.info(f"Poller detected matured timer [{task_id}] for user [{user_id}]")

                    payload = {"task_id": task_id, "text": reminder_text}

                    for target_agent in active_sessions.get(user_id):
                        # Route the event to the correct connection manager
                        if target_agent and target_agent.connection_manager:
                            await target_agent.connection_manager.submit_external_event(payload)
                        else:
                            logger.warning(f"Reminder triggered for user [{user_id}], but no active connection found.")

            except Neo4jCircuitBreakerError:
                logger.debug("Reminder poller paused: Database circuit is OPEN.")
            except Neo4jClientError as e:
                logger.error(f"Database error in reminder poller: {e}. Backing off.")
                await asyncio.sleep(60)
                continue
            except asyncio.CancelledError:
                logger.info("Poller loop received cancellation signal.")
                raise
            except Exception as e:
                logger.error(f"Unexpected exception in reminder poller: {e}", exc_info=True)

            await asyncio.sleep(poll_interval_seconds)