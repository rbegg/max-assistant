import json
import logging
import asyncio
import time
from typing import cast, LiteralString, Any, Dict

from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import Neo4jError, DriverError, ServiceUnavailable


logger = logging.getLogger(__name__)


class Neo4jClientError(Exception):
    """Custom exception raised for errors within the Neo4jClient."""
    pass

class Neo4jCircuitBreakerError(Neo4jClientError):
    """Raised when the circuit breaker is OPEN and fast-failing requests."""
    pass

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
        self._schema_cache: str | None = None

        # --- Circuit Breaker State ---
        self._cb_state = "CLOSED"
        self._cb_failures = 0
        self._cb_failure_threshold = 3  # Open circuit after 3 consecutive failures
        self._cb_recovery_timeout = 60.0  # Wait 60 seconds before testing recovery
        self._cb_last_failure_time = 0.0

    @classmethod
    async def create(
            cls,
            uri: str,
            user: str,
            password: str,
            database: str = "neo4j",
            max_retries: int = 5,
            initial_delay: int = 3,
            backoff_factor: int = 2
    ) -> "Neo4jClient":
        """
        Asynchronous factory method to create and verify a client.
        Includes retry-with-backoff logic for startup.
        """
        delay = initial_delay

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    f"Attempt {attempt}/{max_retries}: Connecting to "
                    f"Neo4j Async Driver URI: {uri} User: {user}..."
                )
                driver = AsyncGraphDatabase.driver(
                    uri,
                    auth=(user, password),
                    max_transaction_retry_time=3.0,  # Give up on retries after 3 seconds
                    connection_timeout=2.0,  # Fail fast if the socket doesn't respond
                    max_connection_lifetime=3600  # Recycle connections to avoid stale sockets
                )
                await driver.verify_connectivity()

                logger.info("Neo4j Async Driver connected successfully.")
                return cls(driver, database)

            except (ServiceUnavailable, DriverError, OSError) as e:
                logger.warning(
                    f"Attempt {attempt} failed for Neo4j connection: "
                    f"{e.__class__.__name__}: {e}"
                )

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

        Raises:
            Neo4jClientError: If the APOC query fails, returns no data, or is restricted.
        """
        if self._schema_cache:
            logger.debug("Returning cached Neo4j schema.")
            return self._schema_cache

        if not self.driver:
            raise Neo4jClientError("Cannot execute query: Neo4j Async Driver is not initialized.")

        self._check_circuit()

        logger.info("Fetching and caching Neo4j schema using APOC...")

        try:
            schema_result = await self.driver.execute_query(
                "CALL apoc.meta.schema({sample: 1000})",
                database_=self.database
            )

            self._record_success()

            if not schema_result.records:
                raise Neo4jClientError("APOC schema query executed but returned no records.")

            apoc_schema = schema_result.records[0].data().get("value", {})
            if not apoc_schema:
                raise Neo4jClientError("APOC schema query returned an empty 'value' payload.")

            # --- Parsing Logic ---
            node_labels = []
            node_properties = {}
            relationship_types = []
            relationship_properties = {}
            relationship_structure = []

            for key, info in apoc_schema.items():
                item_type = info.get("type")

                if item_type == "node":
                    node_labels.append(key)

                    props = {}
                    for prop_name, prop_data in info.get("properties", {}).items():
                        props[prop_name] = prop_data.get("type", "UNKNOWN")
                    if props:
                        node_properties[key] = [f"{k} ({v})" for k, v in props.items()]

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

                    props = {}
                    for prop_name, prop_data in info.get("properties", {}).items():
                        props[prop_name] = prop_data.get("type", "UNKNOWN")
                    if props:
                        relationship_properties[key] = [f"{k} ({v})" for k, v in props.items()]

            schema = {
                "node_labels": sorted(list(set(node_labels))),
                "node_properties": node_properties,
                "relationship_types": sorted(list(set(relationship_types))),
                "relationship_properties": relationship_properties,
                "relationship_structure": sorted(list(set(relationship_structure)))
            }

            self._schema_cache = json.dumps(schema, indent=2)
            return self._schema_cache


        except ServiceUnavailable as e:
            # Expected network drop (e.g., container offline, DNS failure).
            # Record failure, log the string, but suppress the traceback.
            self._record_failure()
            logger.error(f"Database connection offline: {e}")
            raise Neo4jClientError(f"Connection lost while executing query: {str(e)}") from e

        except (DriverError, OSError) as e:
            # Unexpected driver or OS issue.
            # Record failure, and KEEP the traceback for debugging.
            self._record_failure()
            logger.error(f"Unexpected network/driver execution failed: {e}", exc_info=True)
            raise Neo4jClientError(f"Unexpected connectivity error: {str(e)}") from e

        except Neo4jError as e:
            if "There is no procedure with the name `apoc.meta.schema`" in str(e.message):
                raise Neo4jClientError("APOC not installed. Ensure APOC plugins are present in Neo4j.") from e
            if "is restricted" in str(e.message):
                raise Neo4jClientError(
                    "APOC restricted. Add 'apoc.meta.schema' to dbms.security.procedures.allowlist in neo4j.conf") from e

            logger.error(f"Neo4jError during schema fetch: {e}", exc_info=True)
            raise Neo4jClientError(f"Database error while fetching schema: {e.message}") from e

        except Exception as e:
            logger.error(f"Unexpected error fetching Neo4j schema: {e}", exc_info=True)
            raise Neo4jClientError(f"Unexpected error fetching schema: {str(e)}") from e

    def _check_circuit(self):
        """Verifies if a request is allowed to proceed."""
        if self._cb_state == "CLOSED":
            return

        if self._cb_state == "OPEN":
            elapsed = time.monotonic() - self._cb_last_failure_time
            if elapsed > self._cb_recovery_timeout:
                logger.info("Circuit Breaker: Cooldown elapsed. Entering HALF-OPEN state.")
                self._cb_state = "HALF-OPEN"
                return  # Allow one request through to test
            else:
                remaining = int(self._cb_recovery_timeout - elapsed)
                raise Neo4jCircuitBreakerError(f"Database circuit is OPEN. Fast-failing. Try again in {remaining}s.")

    def _record_success(self):
        """Resets the circuit breaker on a successful database operation."""
        if self._cb_state != "CLOSED":
            logger.info("Circuit Breaker: Connection restored. Circuit CLOSED.")
            self._cb_state = "CLOSED"
        self._cb_failures = 0

    def _record_failure(self):
        """Records a failure and potentially opens the circuit."""
        self._cb_failures += 1
        self._cb_last_failure_time = time.monotonic()

        if self._cb_state == "HALF-OPEN" or self._cb_failures >= self._cb_failure_threshold:
            if self._cb_state != "OPEN":
                logger.warning(f"Circuit Breaker: Threshold reached ({self._cb_failures} failures). Circuit OPENED.")
            self._cb_state = "OPEN"

    async def execute_query(self, query: str, params: dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Executes a query using the native async driver.

        Raises:
            Neo4jClientError: If the driver is disconnected or the query fails.
        """
        if not self.driver:
            raise Neo4jClientError("Cannot execute query: Neo4j Async Driver is not initialized.")

        # Evaluate Circuit Breaker Status
        self._check_circuit()

        logger.debug(f"Executing query: {query}")

        try:
            result = await self.driver.execute_query(
                cast(LiteralString, query),
                parameters_=(params or {}),
                database_=self.database
            )

            # Record Success
            self._record_success()

            records = result.records
            summary = result.summary
            response: Dict[str, Any] = {"data": [record.data() for record in records]}

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

        except ServiceUnavailable as e:
            # Expected network drop (e.g., container offline, DNS failure).
            # Record failure, log the string, but suppress the traceback.
            self._record_failure()
            logger.error(f"Database connection offline: {e}")
            raise Neo4jClientError(f"Connection lost while executing query: {str(e)}") from e

        except (DriverError, OSError) as e:
            # Unexpected driver or OS issue.
            # Record failure, and KEEP the traceback for debugging.
            self._record_failure()
            logger.error(f"Unexpected network/driver execution failed: {e}", exc_info=True)
            raise Neo4jClientError(f"Unexpected connectivity error: {str(e)}") from e

        except Exception as e:
            logger.error(f"Unexpected error executing query: {e}", exc_info=True)
            raise Neo4jClientError(f"Unexpected error executing Cypher query: {str(e)}") from e

    async def close(self):
        """Asynchronously closes the driver connection."""
        if self.driver:
            await self.driver.close()
            logger.info("Neo4j Async Driver connection closed.")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()