# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
This module defines a central registry for managing tool providers, enabling
tool discovery, initialization, and collection.

The `ToolRegistry` class orchestrates the overall lifecycle of tool providers,
including registration, dependency injection into providers' constructors, and
access to all tools provided by the registered providers.
"""
import logging
from typing import List, Type

from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama

from max_assistant.clients.neo4j_client import Neo4jClient

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