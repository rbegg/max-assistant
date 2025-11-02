# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
from typing import cast, LiteralString, Any, Dict, List
from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import Neo4jError, DriverError

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
    async def create(cls, uri, user, password, database="neo4j"):
        """
        Asynchronous factory method to create and verify a client.
        """
        try:
            logger.debug(f"Connecting to Neo4j Async Driver URI: {uri} User: {user}...")
            driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
            # 2. Verify connectivity asynchronously
            await driver.verify_connectivity()
            logger.info("Neo4j Async Driver connected successfully.")
            return cls(driver, database)
        except Exception as e:
            logger.error(f"Error connecting to Neo4j: {e}")
            raise

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

            # --- 1. Query returned data ---
            if records:
                data = {"data": [record.data() for record in records]}
                logger.debug(f"Query returned data: {data}")
                return data

            # --- 2. Query was a successful write ---
            counters = summary.counters
            if (counters.nodes_created > 0 or
                    counters.nodes_deleted > 0 or
                    counters.relationships_created > 0 or
                    counters.relationships_deleted > 0 or
                    counters.properties_set > 0):
                summary_data = {"summary": counters.summary()}
                logger.debug(f"Query was a successful write: {summary_data}")
                return summary_data

            # --- 3. Query was a read that returned no results ---
            logger.debug("Query was a read that returned no results.")
            return {"data": []}

        # --- 4. Query failed ---
        except (Neo4jError, DriverError) as e:
            return {"error": e.__class__.__name__, "message": str(e)}
        except Exception as e:
            return {"error": e.__class__.__name__, "message": str(e)}


# Global client instance, will be initialized in the main graph entrypoint
client: "Neo4jClient" = None
