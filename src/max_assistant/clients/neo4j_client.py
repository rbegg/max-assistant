# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
from typing import cast, LiteralString, Any, Dict
import asyncio
import json

from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import Neo4jError, DriverError, ServiceUnavailable

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
        self._schema_cache: str | None = None # Add schema cache property

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

    async def get_schema(self) -> str:
        """
        Fetches a comprehensive schema from Neo4j using the APOC library
        and caches it.
        This is used to provide context to the LLM for Cypher generation.
        """
        # Return from cache if available
        if self._schema_cache:
            logger.debug("Returning cached Neo4j schema.")
            return self._schema_cache

        logger.info("Fetching and caching Neo4j schema using APOC...")
        try:
            # --- MODIFICATION: Use the APOC procedure ---
            # This is the most reliable way to get the schema.
            # We use {sample: 1000} to limit the scan on large graphs.
            # You can remove the config map for a full (slower) scan.
            schema_result = await self.driver.execute_query(
                "CALL apoc.meta.schema({sample: 1000})",
                database_=self.database
            )

            if not schema_result.records:
                logger.error("APOC schema query returned no records.")
                return json.dumps({"error": "Failed to fetch schema", "message": "APOC query returned no records."})

            # apoc.meta.schema() returns a single record with a 'value' key
            # which is a large dictionary.
            apoc_schema = schema_result.records[0].data().get("value", {})
            if not apoc_schema:
                logger.error("APOC schema query returned no 'value' in record.")
                return json.dumps({"error": "Failed to fetch schema", "message": "APOC query returned empty value."})

            # --- Parsing Logic for apoc.meta.schema() output ---
            node_labels = []
            node_properties = {}
            relationship_types = []
            relationship_properties = {}
            relationship_structure = []

            for key, info in apoc_schema.items():
                item_type = info.get("type")

                if item_type == "node":
                    node_labels.append(key)

                    # Get node properties
                    props = {}
                    for prop_name, prop_data in info.get("properties", {}).items():
                        props[prop_name] = prop_data.get("type", "UNKNOWN")
                    if props:
                        node_properties[key] = [f"{k} ({v})" for k, v in props.items()]

                    # Get relationship structures from the node
                    for rel_name, rel_data in info.get("relationships", {}).items():
                        direction = rel_data.get("direction", "out")
                        target_labels = rel_data.get("labels", [])

                        for target_label in target_labels:
                            if direction == "out":
                                relationship_structure.append(f"(:{key})-[:{rel_name}]->(:{target_label})")
                            elif direction == "in":
                                relationship_structure.append(f"(:{target_label})-[:{rel_name}]->(:{key})")

                elif item_type == "relationship":
                    relationship_types.append(key)

                    # Get relationship properties
                    props = {}
                    for prop_name, prop_data in info.get("properties", {}).items():
                        props[prop_name] = prop_data.get("type", "UNKNOWN")
                    if props:
                        relationship_properties[key] = [f"{k} ({v})" for k, v in props.items()]

            # Format for the LLM
            schema = {
                "node_labels": sorted(list(set(node_labels))),
                "node_properties": node_properties,
                "relationship_types": sorted(list(set(relationship_types))),
                "relationship_properties": relationship_properties,
                "relationship_structure": sorted(list(set(relationship_structure)))
            }

            # Cache and return the JSON string
            self._schema_cache = json.dumps(schema, indent=2)
            return self._schema_cache

        except Neo4jError as e:
            if "There is no procedure with the name `apoc.meta.schema`" in str(e.message):
                logger.error("APOC procedures not found. Please ensure APOC is installed on Neo4j.")
                return json.dumps({"error": "APOC not installed", "message": str(e.message)})
            if "is restricted" in str(e.message):
                logger.error(
                    "APOC procedure is restricted. Add 'apoc.meta.schema' to dbms.security.procedures.allowlist in neo4j.conf")
                return json.dumps({"error": "APOC procedure restricted", "message": str(e.message)})
            logger.error(f"Failed to fetch Neo4j schema: {e}", exc_info=True)
            return json.dumps({"error": e.__class__.__name__, "message": str(e.message)})

        except Exception as e:
            logger.error(f"Failed to fetch Neo4j schema: {e}", exc_info=True)
            return json.dumps({"error": "Failed to fetch schema", "message": str(e)})


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

