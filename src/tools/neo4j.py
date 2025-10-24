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

    async def get_schedule_summary(self, username: str) -> str:
        """Fetches a user's schedule from Neo4j and returns it as a string."""
        logging.info(f"Fetching schedule for user: {username}")
        query = """
        MATCH (u:User {name: $username})-[:HAS_SCHEDULE]->(s:Schedule)
        RETURN s.summary
        """
        try:
            records, _, _ = await self._driver.execute_query(query, username=username)
            if records:
                return records[0]["s.summary"]
            return "No schedule found."
        except Exception as e:
            logging.error(f"Could not retrieve schedule for {username}: {e}", exc_info=True)
            return "No schedule found."


# Singleton instance of the client
neo4j_client = Neo4jClient(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
