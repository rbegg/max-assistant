# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
Defines the AppServices container class to centralize service initialization.
Common to main.py and text_client.py.
"""

import logging
import asyncio
from langchain_ollama import ChatOllama
from typing import Any  # Used for the compiled graph type

from max_assistant.config import (
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,
    OLLAMA_MODEL_NAME, OLLAMA_BASE_URL
)
from max_assistant.clients.neo4j_client import Neo4jClient
from max_assistant.clients.ollama_preloader import warm_up_ollama_async
from max_assistant.tools.person_tools import PersonTools
from max_assistant.tools.family_tools import FamilyTools
from max_assistant.tools.schedule_tools import ScheduleTools
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
            person_tools: PersonTools,
            family_tools: FamilyTools,
            schedule_tools: ScheduleTools,
            reasoning_engine: Any  # This is the compiled StateGraph
    ):
        self.db_client = db_client
        self.llm = llm
        self.person_tools = person_tools
        self.family_tools = family_tools
        self.schedule_tools = schedule_tools
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

            # 3. Create Tool Services
            logger.info("Initializing tool services...")
            person_tools = PersonTools(db_client)
            family_tools = FamilyTools(client=db_client)
            schedule_tools = ScheduleTools(client=db_client)
            logger.info("Tool services initialized.")

            # 4. Create Reasoning Engine
            logger.info("Initializing the reasoning engine...")
            # This now passes the correct arguments, fixing the P0 bug
            reasoning_engine = await create_reasoning_engine(llm, person_tools, family_tools, schedule_tools)
            logger.info("Reasoning engine initialized.")

            # 5. Create and return the container instance
            return cls(
                db_client=db_client,
                llm=llm,
                person_tools=person_tools,
                family_tools=family_tools,
                schedule_tools=schedule_tools,
                reasoning_engine=reasoning_engine
            )

        except Exception as e:
            logger.critical(f"Failed to initialize application services: {e}", exc_info=True)
            raise