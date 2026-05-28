# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
This module defines a central registry for managing tool providers, enabling
tool discovery, initialization, and collection.

The `ToolRegistry` class orchestrates the overall lifecycle of tool providers,
including registration, dependency injection into providers' constructors, and
access to all tools provided by the registered providers.
"""
import json
import logging
from typing import List, Type

from pydantic import BaseModel, ValidationError
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama

from max_assistant.clients.neo4j_client import Neo4jClient, Neo4jClientError, Neo4jCircuitBreakerError

logger = logging.getLogger(__name__)


class BaseToolProvider:
    """
    Abstract base class for a class that provides tools.
    This helps with type hinting and structure.
    """
    def __init__(self, db_client: Neo4jClient = None, llm: ChatOllama = None):
        self.db_client = db_client
        self.llm = llm

    def get_tools(self) -> List[BaseTool]:
        raise NotImplementedError

    async def _query_and_validate_nodes(
            self,
            query: str,
            params: dict,
            model_class: Type[BaseModel],
            result_key: str
    ) -> str:
        """
        Executes a query, validates results against a Pydantic model,
        and returns a JSON string gracefully handling all DB/Validation errors.
        """
        logger.debug(f"Executing query for model: {model_class.__name__}")

        try:
            result = await self.db_client.execute_query(query, params)

            raw_nodes = [item[result_key] for item in result.get("data", [])]
            validated_nodes = [model_class.model_validate(node) for node in raw_nodes]

            return json.dumps(
                [node.model_dump(mode='json') for node in validated_nodes],
                indent=2,
                default=str
            )

        except Neo4jCircuitBreakerError as e:
            # 1. Catch the Fast-Fail FIRST
            logger.warning(f"Circuit Breaker blocked {model_class.__name__} query: {e}")
            return json.dumps({
                "error": "Database_Offline_Circuit_Open",
                "instruction": "The system database is currently offline. Do not attempt further queries. Inform the user you cannot access their data right now.",
                "details": str(e)
            })

        except Neo4jClientError as e:
            # 2. Catch standard query/driver errors
            logger.error(f"Database error in {model_class.__name__} query: {e}")
            return json.dumps({"error": "Database_Unavailable", "details": str(e)})

        except ValidationError as e:
            logger.error(f"Validation error for {model_class.__name__}: {e.errors()}")
            return json.dumps({"error": "Data validation failed", "details": e.errors()}, default=str)

        except KeyError:
            logger.error(f"Validation: Unexpected data structure. Missing key '{result_key}'.")
            return json.dumps({"error": "Data parsing failed", "details": f"Missing key: {result_key}"})

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return json.dumps({"error": "Data parsing failed", "details": str(e)})

    async def _safe_execute_query(self, query: str, params: dict = None) -> str:
        """
        Executes a raw query and safely returns the JSON stringified result.
        Ideal for write operations or queries that don't need Pydantic validation.
        """
        try:
            result = await self.db_client.execute_query(query, params or {})
            return json.dumps(result, indent=2, default=str)

        except Neo4jCircuitBreakerError as e:
            logger.warning(f"Circuit Breaker blocked raw query: {e}")
            return json.dumps({
                "error": "Database_Offline_Circuit_Open",
                "instruction": "The system database is currently offline. Do not attempt further queries. Inform the user you cannot access their data right now.",
                "details": str(e)
            })

        except Neo4jClientError as e:
            logger.error(f"Database error during raw query execution: {e}")
            return json.dumps({"error": "Database_Unavailable", "details": str(e)})

        except Exception as e:
            logger.error(f"Unexpected error during raw query execution: {e}")
            return json.dumps({"error": "Internal_Error", "details": str(e)})

class ToolRegistry:
    """
    A registry to manage the collection and initialization of tool providers.
    """

    def __init__(self, db_client: Neo4jClient, llm: ChatOllama):
        self.db_client = db_client
        self.llm = llm
        self._providers: List[BaseToolProvider] = []
        self._tools: List[BaseTool] = []

    def register_provider(self, provider_class: Type[BaseToolProvider]):
        """
        Initializes and registers a tool provider.
        The provider class is instantiated with the db_client and llm.
        """
        if not issubclass(provider_class, BaseToolProvider):
            logger.warning(
                f"Class {provider_class.__name__} does not inherit from BaseToolProvider. "
                "Registration might not work as expected."
            )

        # Instantiate the provider, passing the necessary clients.
        provider_instance = provider_class(db_client=self.db_client, llm=self.llm)
        self._providers.append(provider_instance)
        # Eagerly collect tools upon registration.
        new_tools = provider_instance.get_tools()
        self._tools.extend(new_tools)
        logger.info(f"Registered {len(new_tools)} tools from {provider_class.__name__}.")

    def get_all_tools(self) -> List[BaseTool]:
        """
        Returns a flat list of all tools from all registered providers.
        """
        return self._tools