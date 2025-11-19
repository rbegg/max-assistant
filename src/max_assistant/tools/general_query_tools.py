# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
Defines a dynamic, LLM-powered tool for answering general-purpose
questions against the Neo4j database.
"""
import json
import re
import logging
from typing import Type

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from max_assistant.clients.neo4j_client import Neo4jClient
from max_assistant.agent.prompts import CYPHER_GENERATION_PROMPT

logger = logging.getLogger(__name__)


class GeneralQuestionArgs(BaseModel):
    """Input arguments for the answer_general_question tool."""
    question: str = Field(
        ...,
        description="A natural language question to be answered by the graph."
    )
    user_info_json: str = Field(
        ...,
        description="The user's info (a JSON string) from the main graph state. This is required to resolve questions like 'my' or 'I'."
    )


class GeneralQueryTools:
    """
    A toolset that uses an LLM to dynamically generate and execute
    Cypher queries for ad-hoc questions.
    """

    def __init__(self, client: Neo4jClient, llm: BaseChatModel):
        """
        Initializes the toolset with a Neo4j client and an LLM.
        """
        self.client = client
        self.llm = llm
        self.cypher_generation_chain = CYPHER_GENERATION_PROMPT | self.llm
        logger.info("GeneralQueryTools initialized with Neo4j client and LLM.")

    def _parse_cypher_from_response(self, response_content: str) -> str:
        """
        Safely extracts a Cypher query from an LLM's markdown response.
        """
        # Look for a Cypher code block
        match = re.search(r"```(?:cypher|CYPHER)\n(.*?)```", response_content, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fallback: if no code block, assume the whole response is the query
        # but clean it of common LLM "chatter"
        query = response_content.strip()
        if query.startswith("MATCH") or query.startswith("RETURN"):
            return query

        logger.warning(f"Could not parse Cypher from LLM response: {response_content}")
        # Return a query that will gracefully fail
        return "RETURN 'Error: Could not parse Cypher query from LLM response'"

    async def answer_general_question(self, question: str, user_info_json: str) -> str:
        """
        Use this tool for complex questions about relationships or entities in the graph database that CANNOT be answered
        by other, more specific tools.
        IMPORTANT: Do NOT use this tool for simple questions about the user's own identity, such as "what is my name?"
        or "where do I live?". Answer those directly from the user_info context.
        This tool translates a natural language question into a Cypher query,
        executes it, and returns the raw JSON data.
        """
        logger.info(f"Tool: answer_general_question for: {question}")

        try:
            # 1. Get the graph schema
            schema_str = await self.client.get_schema()

            # Check for error in schema fetching
            try:
                schema_data = json.loads(schema_str)
                if isinstance(schema_data, dict) and "error" in schema_data:
                    logger.error(f"Error retrieving graph schema: {schema_data}")
                    return json.dumps(
                        {"error": "Could not retrieve graph schema.", "details": schema_data.get("message")})
            except json.JSONDecodeError:
                logger.error(f"Failed to decode schema JSON: {schema_str}")
                return json.dumps({"error": "Failed to decode graph schema."})

            # 2. Generate the Cypher query
            logger.debug("Generating Cypher query...")
            response = await self.cypher_generation_chain.ainvoke({
                "schema": schema_str,
                "question": question,
                "user_info": user_info_json
            })

            cypher_query = self._parse_cypher_from_response(response.content)
            logger.info(f"Generated Cypher: {cypher_query}")

            # 3. Execute the query
            # We use params={} as the LLM is instructed to embed values
            result = await self.client.execute_query(cypher_query, params={})

            # 4. Return the raw JSON string
            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"Error in answer_general_question: {e}", exc_info=True)
            return json.dumps({"error": e.__class__.__name__, "message": str(e)})

    def get_tools(self) -> list:
        """
        Returns a list of all tool methods bound to this instance.
        """
        return [
            StructuredTool.from_function(
                func=None,
                coroutine=self.answer_general_question,
                name="answer_general_question",
                description=self.answer_general_question.__doc__,
                args_schema=GeneralQuestionArgs
            ),
        ]