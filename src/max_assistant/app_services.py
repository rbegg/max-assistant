# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
Defines the AppServices container class to centralize service initialization.
Common to main.py and text_client.py.
"""

import logging
import asyncio
from langchain_ollama import ChatOllama
from typing import Any, Dict  # Used for the compiled graph type

from max_assistant.config import (
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,
    OLLAMA_MODEL_NAME, OLLAMA_BASE_URL
)
from max_assistant.clients.neo4j_client import Neo4jClient
from max_assistant.clients.ollama_preloader import warm_up_ollama_async
from max_assistant.tools.registry import ToolRegistry
from max_assistant.tools.person_tools import PersonTools
from max_assistant.tools.family_tools import FamilyTools
from max_assistant.tools.schedule_tools import ScheduleTools
from max_assistant.tools.gmail_tools import GmailTools
from max_assistant.tools.general_query_tools import GeneralQueryTools
from max_assistant.agent.graph import create_reasoning_engine

logger = logging.getLogger(__name__)


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
            reasoning_engine: Any  # This is the compiled StateGraph
    ):
        self.db_client = db_client
        self.llm = llm
        self.tool_registry = tool_registry
        self.user_info = user_info
        self.reasoning_engine = reasoning_engine

    @classmethod
    async def create(cls) -> "AppServices":
        """
        Asynchronously creates and initializes all application services.
        This is the single source of truth for service setup.
        """
        logger.info("Initializing application services...")

        try:
            # 1 & 2. Initialize Database and LLM in parallel
            logger.info("Initializing the neo4j client and the llm ...")
            results = await asyncio.gather(
                Neo4jClient.create(
                    NEO4J_URI,
                    NEO4J_USERNAME,
                    NEO4J_PASSWORD
                ),
                warm_up_ollama_async(
                    OLLAMA_MODEL_NAME,
                    OLLAMA_BASE_URL,
                    temperature=0
                )
            )
            db_client = results[0]
            llm = results[1]

            if not db_client:
                raise RuntimeError("Failed to initialize Neo4j client.")
            if not llm:
                raise RuntimeError("Failed to initialize the LLM.")
            logger.info("Successfully initialized Neo4j client and LLM.")

            # 3. Fetch user info once
            # We temporarily create PersonTools to fetch this on startup
            person_tools_instance = PersonTools(db_client)
            user_info = await person_tools_instance.get_user_info_internal()

            # 4. Create Tool Registry and register all tool providers
            logger.info("Initializing and populating tool registry...")
            tool_registry = ToolRegistry(db_client=db_client, llm=llm)
            tool_registry.register_provider(PersonTools)
            tool_registry.register_provider(FamilyTools)
            tool_registry.register_provider(ScheduleTools)
            tool_registry.register_provider(GmailTools)
            tool_registry.register_provider(GeneralQueryTools)
            logger.info("Tool providers registered.")

            # 5. Create Reasoning Engine, passing the registry
            logger.info("Initializing the reasoning engine...")
            reasoning_engine = await create_reasoning_engine(llm, tool_registry)
            logger.info("Reasoning engine initialized.")

            # 6. Create and return the container instance
            return cls(
                db_client=db_client,
                llm=llm,
                tool_registry=tool_registry,
                user_info=user_info,
                reasoning_engine=reasoning_engine,
            )

        except Exception as e:
            logger.critical(f"Failed to initialize application services: {e}", exc_info=True)
            raise