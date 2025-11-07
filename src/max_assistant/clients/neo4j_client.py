# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
from typing import cast, LiteralString, Any, Dict
from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import Neo4jError, DriverError, ServiceUnavailable
import asyncio

import logging
logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    A native asyncio client for Neo4j.
    It uses AsyncGraphDatabase for non-blocking I/O.
    """

    def __init__(self, driver: AsyncDriver, database: str):
        """
        Private constructor. Use .create() to instantiate.
        """
        self.driver = driver
        self.database = database

    @classmethod
    async def create(
            cls,
            uri,
            user,
            password,
            database="neo4j",
            max_retries=5,
            initial_delay=3,
            backoff_factor=2
    ):
        """
        Asynchronous factory method to create and verify a client.
        Includes retry-with-backoff logic for startup.
        """
        delay = initial_delay

        # Range starts at 1 and includes max_retries
        for attempt in range(1, max_retries + 1):
            try:
                logger.debug(
                    f"Attempt {attempt}/{max_retries}: Connecting to "
                    f"Neo4j Async Driver URI: {uri} User: {user}..."
                )
                driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
                await driver.verify_connectivity()

                logger.info("Neo4j Async Driver connected successfully.")
                return cls(driver, database)

            except (ServiceUnavailable, DriverError, OSError) as e:
                logger.warning(
                    f"Attempt {attempt} failed for Neo4j connection: "
                    f"{e.__class__.__name__}: {e}"
                )

                # --- MODIFICATION 3 ---
                # Exit condition is now simpler
                if attempt == max_retries:
                    logger.error(f"All {max_retries} attempts failed to connect to Neo4j.")
                    raise

                logger.info(f"Retrying Neo4j connection in {delay} seconds...")
                await asyncio.sleep(delay)
                delay *= backoff_factor

            except Exception as e:
                logger.error(f"An unexpected error occurred during Neo4j connection: {e}", exc_info=True)
                raise
        raise RuntimeError("Failed to connect to Neo4j after all retries.")


    async def close(self):
        """Asynchronously closes the driver connection."""
        if self.driver:
            await self.driver.close()
            logger.info("Neo4j Async Driver connection closed.")

    async def execute_query(self, query: str, params: dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Executes a query using the native async driver.
        """
        if not self.driver:
            return{"error": "Neo4j is not connected"}

        logger.debug(f"Executing query: {query}")

        try:
            result = await self.driver.execute_query(
                cast(LiteralString, query),
                parameters_=(params or {}),
                database_=self.database
            )

            records = result.records
            summary = result.summary

            response: Dict[str, Any] = {"data": [record.data() for record in records]}

            # Include summary data if there are any database updates
            counters = summary.counters
            if (counters.nodes_created > 0 or
                    counters.nodes_deleted > 0 or
                    counters.relationships_created > 0 or
                    counters.relationships_deleted > 0 or
                    counters.properties_set > 0):
                response["summary"] = {
                    "nodes_created": counters.nodes_created,
                    "nodes_deleted": counters.nodes_deleted,
                    "relationships_created": counters.relationships_created,
                    "relationships_deleted": counters.relationships_deleted,
                    "properties_set": counters.properties_set,
                }

            logger.debug(f"Query returned: {response}")
            return response

        # --- 4. Query failed ---
        except (Neo4jError, DriverError) as e:
            return {"error": e.__class__.__name__, "message": str(e)}
        except Exception as e:
            return {"error": e.__class__.__name__, "message": str(e)}

