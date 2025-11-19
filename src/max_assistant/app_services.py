# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
Defines the AppServices container class to centralize service initialization.
Common to main.py and text_client.py.
"""

import logging
import asyncio
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableSerializable
from typing import Any, Dict, Tuple

from max_assistant.config import (
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,
    OLLAMA_MODEL_NAME, OLLAMA_BASE_URL
)
from max_assistant.clients.neo4j_client import Neo4jClient
from max_assistant.clients.ollama_preloader import create_llm_instance, preload_model_async
from max_assistant.tools import ALL_TOOL_PROVIDERS
from max_assistant.tools.registry import ToolRegistry
from max_assistant.tools.person_tools import PersonTools
from max_assistant.agent.graph import create_reasoning_engine

logger = logging.getLogger(__name__)

# Use a stable, protocol-based type hint for the compiled graph.
# This avoids import issues and accurately describes the object's capabilities.
ReasoningEngine = RunnableSerializable


class AppServices:
    """
    A container class to encapsulate all singleton services for the application.
    This solves the WET code issue by centralizing service initialization.
    """

    def __init__(
            self,
            db_client: Neo4jClient,
            llm: ChatOllama,
            tool_registry: ToolRegistry,
            user_info: Dict[str, Any],
            reasoning_engine: ReasoningEngine,
            llm_ready_event: asyncio.Event
    ):
        self.db_client = db_client
        self.llm = llm
        self.tool_registry = tool_registry
        self.user_info = user_info
        self.reasoning_engine = reasoning_engine
        self.llm_ready_event = llm_ready_event

    @classmethod
    async def create(cls) -> "AppServices":
        """
        Asynchronously creates and initializes all application services.
        This is the single source of truth for service setup.
        """
        logger.info("Initializing application services...")
        try:
            # Event to signal when the LLM is warmed up and ready.
            llm_ready_event = asyncio.Event()

            # --- 1. Initialize Core Clients (DB and LLM) ---
            db_client, llm = await cls._initialize_clients(llm_ready_event)

            # --- 2. Fetch User Info ---
            # This requires the db_client to be ready.
            user_info = await cls._fetch_user_info(db_client)

            # --- 3. Initialize and Populate Tool Registry ---
            tool_registry = cls._initialize_tool_registry(db_client, llm)

            # --- 4. Create Reasoning Engine ---
            reasoning_engine = await create_reasoning_engine(llm, tool_registry)
            logger.info("Reasoning engine initialized.")

            # --- 5. Create and return the container instance ---
            return cls(
                db_client=db_client,
                llm=llm,
                tool_registry=tool_registry,
                user_info=user_info,
                reasoning_engine=reasoning_engine,
                llm_ready_event=llm_ready_event,
            )

        except Exception as e:
            logger.critical(f"Failed to initialize application services: {e}", exc_info=True)
            raise

    @staticmethod
    async def _initialize_clients(llm_ready_event: asyncio.Event) -> Tuple[Neo4jClient, ChatOllama]:
        """Initializes the Neo4j client and the Ollama LLM concurrently."""
        logger.info("Initializing Neo4j client and LLM...")

        async def _init_llm_and_warmup() -> ChatOllama:
            """Creates LLM instance and starts warm-up in a background task."""
            llm = create_llm_instance(OLLAMA_MODEL_NAME, OLLAMA_BASE_URL, temperature=0)
            asyncio.create_task(preload_model_async(llm, ready_event=llm_ready_event))
            logger.info("LLM warm-up process started in the background.")
            return llm

        results = await asyncio.gather(
            Neo4jClient.create(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD),
            _init_llm_and_warmup()
        )
        db_client, llm = results[0], results[1]

        if not db_client:
            raise RuntimeError("Fatal: Failed to initialize Neo4j client.")
        if not llm:
            raise RuntimeError("Fatal: Failed to initialize the LLM.")

        logger.info("Successfully initialized Neo4j client and LLM instance.")
        return db_client, llm

    @staticmethod
    async def _fetch_user_info(db_client: Neo4jClient) -> Dict[str, Any]:
        """Fetches essential user information on startup."""
        logger.info("Fetching user info...")
        # We temporarily create PersonTools to fetch this on startup.
        # This is acceptable as it's a one-off operation.
        person_tools_instance = PersonTools(db_client)
        user_info = await person_tools_instance.get_user_info_internal()
        logger.info("User info fetched successfully.")
        return user_info

    @staticmethod
    def _initialize_tool_registry(db_client: Neo4jClient, llm: ChatOllama) -> ToolRegistry:
        """Creates the tool registry and dynamically registers all providers."""
        logger.info("Initializing and populating tool registry...")
        tool_registry = ToolRegistry(db_client=db_client, llm=llm)

        # Dynamically register all tool providers from the central list.
        for provider_class in ALL_TOOL_PROVIDERS:
            tool_registry.register_provider(provider_class)
            logger.info(f"-> Registered tool provider: {provider_class.__name__}")

        logger.info(f"Tool registry populated with {len(ALL_TOOL_PROVIDERS)} providers.")
        return tool_registry