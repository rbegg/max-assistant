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
import inspect
from typing import List, Type, Any, Dict

from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel

from max_assistant.clients.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    A central registry for discovering, initializing, and collecting all tools.
    """

    def __init__(self, db_client: Neo4jClient, llm: BaseChatModel):
        """
        Initializes the registry with shared services that tools may need.
        """
        self.db_client = db_client
        self.llm = llm
        self._tool_providers: List[Type[Any]] = []
        self._tool_instances: Dict[Type[Any], Any] = {}

    def register_provider(self, provider_class: Type[Any]):
        """Registers a class that provides tools (e.g., PersonTools)."""
        if provider_class not in self._tool_providers:
            self._tool_providers.append(provider_class)

    def _get_provider_instance(self, provider_class: Type[Any]) -> Any:
        """
        Lazily instantiates and caches a tool provider, injecting the
        necessary dependencies by inspecting its constructor.
        """
        if provider_class not in self._tool_instances:
            constructor_params = inspect.signature(provider_class.__init__).parameters
            dependencies = {}
            if 'client' in constructor_params:
                dependencies['client'] = self.db_client
            if 'llm' in constructor_params:
                dependencies['llm'] = self.llm

            logger.info(f"Instantiating {provider_class.__name__} with dependencies: {list(dependencies.keys())}")
            instance = provider_class(**dependencies)
            self._tool_instances[provider_class] = instance
        return self._tool_instances[provider_class]

    def get_all_tools(self) -> List[BaseTool]:
        """
        Instantiates all registered providers and collects their tools into a flat list.
        """
        all_tools: List[BaseTool] = []
        for provider_class in self._tool_providers:
            provider_instance = self._get_provider_instance(provider_class)
            if hasattr(provider_instance, 'get_tools'):
                tools = provider_instance.get_tools()
                all_tools.extend(tools)
                logger.info(f"Loaded {len(tools)} tools from {provider_class.__name__}")
        return all_tools

    def get_tool_provider(self, provider_class: Type[Any]) -> Any:
        """Gets a specific, initialized tool provider instance by its class."""
        return self._get_provider_instance(provider_class)