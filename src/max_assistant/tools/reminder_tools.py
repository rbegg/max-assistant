# Copyright (c) 2026, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
Defines LangGraph tools for scheduling reminders using task nodes and saving them to Neo4j.
"""
from datetime import datetime, timedelta
import json
import asyncio
import logging

from langchain_core.tools import StructuredTool
from max_assistant.tools.registry import BaseToolProvider
from max_assistant.models.reminder_models import ScheduleReminderArgs
from max_assistant.clients.neo4j_client import Neo4jClientError, Neo4jCircuitBreakerError


logger = logging.getLogger(__name__)

class ReminderTools(BaseToolProvider):
    """
    A class that encapsulates reminder and task scheduling tools
    backed by a Neo4j client instance.
    """

    def __init__(self, db_client, llm=None):
        super().__init__(db_client, llm)
        logger.info("ReminderTools initialized with a Neo4j client.")

    async def schedule_reminder(self, message: str, delay_minutes: float) -> str:
        """
        Use this tool to schedule a reminder message after a certain number of minutes.
        """
        logger.info(f"Tool: schedule_reminder - message='{message}', delay={delay_minutes}m")

        try:
            # Calculate the absolute execution time
            due_datetime = datetime.now() + timedelta(minutes=delay_minutes)
            due_iso = due_datetime.isoformat()

            query = """
                MATCH (u:User) 
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
                "due_time": due_iso
            }

            result = await self.db_client.execute_query(query, params)

            return json.dumps({
                "success": True,
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
        def sync_fallback(*args, **kwargs):
            raise NotImplementedError("This tool is asynchronous and must be invoked using invoke_async or waited.")

        return [
            StructuredTool.from_function(
                func=sync_fallback,
                coroutine=self.schedule_reminder,
                name="schedule_reminder",
                description=self.schedule_reminder.__doc__,
                args_schema=ScheduleReminderArgs
            )
        ]

    async def start_reminder_poller_dynamic(self, get_agent_fn, poll_interval_seconds: int = 20):
        """
        A continuous, non-blocking background loop that polls Neo4j for pending reminders.
        """
        logger.info("Background reminder poller service activated via localized date-tree matching.")

        check_query = """
                      WITH datetime($now) AS now
                          MATCH (t:Task {status: 'PENDING', type : 'REMINDER'})
                      WHERE t.due_time <= now
                      SET t.status = 'COMPLETED'
                          RETURN t.text AS text, t.id AS id
                      """

        while True:
            try:
                agent = get_agent_fn()
                if not agent:
                    await asyncio.sleep(poll_interval_seconds)
                    continue

                params = {"now": datetime.now().isoformat()}

                # The execution is now safely inside a try/except block
                result = await self.db_client.execute_query(check_query, params)

                tasks_due = result.get("data", [])
                for task in tasks_due:
                    reminder_text = task.get("text")
                    task_id = task.get("id")

                    logger.info(f"Poller detected matured timer [{task_id}]: '{reminder_text}'")

                    payload = {"task_id": task_id, "text": reminder_text}

                    if agent and agent.connection_manager:
                        await agent.connection_manager.submit_external_event(payload)
                    else:
                        logger.warning("Reminder triggered, but no active connection is available.")

            except Neo4jCircuitBreakerError:
                # SILENT CATCH: The DB is offline. Log it at DEBUG level so we don't spam
                # the terminal every 20 seconds, and just wait for the next tick.
                logger.debug("Reminder poller paused: Database circuit is OPEN.")

            except Neo4jClientError as e:
                logger.error(f"Database error in reminder poller: {e}. Backing off for 60 seconds.")
                await asyncio.sleep(60)  # Extended backoff to prevent log spam
                continue  # Skip the standard 20-second sleep at the bottom of the loop

            except asyncio.CancelledError:
                logger.info("Poller loop received explicit cancellation signal. Shutting down gracefully.")
                raise

            except Exception as e:
                logger.error(f"Unexpected exception in reminder poller: {e}", exc_info=True)

            await asyncio.sleep(poll_interval_seconds)