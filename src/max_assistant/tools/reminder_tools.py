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
        Args:
            message: The reminder message to be scheduled.
            delay_minutes: The duration in minutes after which the reminder should be triggered.
        Examples: 'Remind me to go for lunch in 10 minutes', 'in 15 minutes remind me to call mom'.
        If a time is provided, get the current time and convert to minutes.
        Example: 'remind me to go for lunch at 11:45'
        Saves the task with a PENDING status into the Neo4j graph database.
        """
        logger.info(f"Tool: schedule_reminder - message='{message}', delay={delay_minutes}m")

        try:
            # 1. Calculate the absolute execution time in the local timezone
            due_datetime = datetime.now() + timedelta(minutes=delay_minutes)
            due_iso = due_datetime.isoformat()

            # 2. Persist the task node and map it to the primary user
            query = """
                // 1. Match the existing user
                MATCH (u:User ) // {id: $userId} Recommended to anchor your user with an ID or property
                
                // 2. Initialize the datetime variable from the parameter
                WITH u, datetime($due_time) AS d
                
                // 3. Create the new Task
                CREATE (t:Task {
                    id: randomUUID(),
                    text: $text,
                    due_time: d,
                    type: 'REMINDER',
                    status: 'PENDING'
                })
                
                // 4. Build/Merge the Time Tree step-by-step
                MERGE (u)-[:HAS_YEAR]->(y:Year {year: d.year})
                MERGE (y)-[:HAS_MONTH]->(m:Month {month: d.month, year: d.year})
                ON CREATE SET m.name = format(d, 'MMM') 
                
                MERGE (m)-[:HAS_DAY]->(day:Day {day: d.day, month: d.month, year: d.year})
                
                // 5. Link the Task to both the Day node and the User node
                MERGE (day)-[:HAS_TASK]->(t)
                
                // 6. Return the newly created task ID
                RETURN t.id AS task_id
            """
            params = {
                "text": message,
                "due_time": due_iso
            }

            result = await self.db_client.execute_query(query, params)

            if "error" in result:
                logger.error(f"Neo4j write failed for reminder: {result['error']}")
                return json.dumps(result)

            return json.dumps({
                "success": True,
                "message": f"Successfully scheduled reminder: '{message}' for {delay_minutes} minutes from now.",
                "due_time": due_iso
            }, indent=2)

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
        A continuous, non-blocking background loop that polls Neo4j for pending reminders
        anchored specifically to the current date's time-tree node branch.
        """
        logger.info("Background reminder poller service activated via localized date-tree matching.")

        now: str

        # This optimized query starts by grabbing today's exact date metrics.
        check_query = """
                        WITH datetime($now) AS now
                        MATCH (t:Task {status: 'PENDING', type: 'REMINDER'})
                        WHERE t.due_time <= now
                        SET t.status = 'COMPLETED'
                        RETURN t.text AS text, t.id AS id
                      """

        while True:
            try:
                # Resolve the active conversational agent processing instance
                agent = get_agent_fn()

                # If no user is currently connected to the websocket endpoint, wait for connection
                if not agent:
                    await asyncio.sleep(poll_interval_seconds)
                    continue

                params = {"now": datetime.now().isoformat()}
                result = await self.db_client.execute_query(check_query, params )

                if "error" in result:
                    logger.error(f"Poller date-tree query encountered an error: {result['error']}")
                else:
                    tasks_due = result.get("data", [])
                    for task in tasks_due:
                        reminder_text = task.get("text")
                        task_id = task.get("id")

                        logger.info(f"Poller detected matured timer [{task_id}] on today's tree: '{reminder_text}'")

                        payload = {
                            "task_id": task_id,
                            "text": reminder_text
                        }

                        if agent and agent.connection_manager:
                            # Delegate the event to the active connection's event loop
                            await agent.connection_manager.submit_external_event(payload)
                        else:
                            logger.warning("Reminder triggered, but no active connection is available.")

            except asyncio.CancelledError:
                logger.info("Poller loop received explicit cancellation signal. Shutting down gracefully.")
                raise
            except Exception as e:
                logger.error(f"Unexpected exception tracked inside reminder poller execution: {e}", exc_info=True)

            await asyncio.sleep(poll_interval_seconds)