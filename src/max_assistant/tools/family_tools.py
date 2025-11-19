# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
Defines LangGraph tools for querying the User's family tree.
"""
import json
import logging
from typing import Type

from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from pydantic import ValidationError, BaseModel

from max_assistant.clients.neo4j_client import Neo4jClient
from max_assistant.models.person_models import PersonDetails
from max_assistant.tools.registry import BaseToolProvider

logger = logging.getLogger(__name__)


class NoArgs(BaseModel):
    """An empty schema for tools that take no arguments."""
    pass


class FamilyTools(BaseToolProvider):
    """
    A class that encapsulates family-tree-related tools,
    with all queries relative to the :User node.
    """

    def __init__(self, db_client: Neo4jClient, llm: ChatOllama = None):
        """
        Initializes the toolset with a specific Neo4j client.
        """
        super().__init__(db_client, llm)
        logger.info("FamilyTools initialized with a Neo4j client.")

    async def _query_and_validate_nodes(
            self,
            query: str,
            params: dict,
            model_class: Type[BaseModel],
            result_key: str
    ) -> str:
        """
        Private helper to execute a query, validate results against a
        Pydantic model, and return a JSON string.
        (This is a copy of the helper in PersonTools)
        """
        logger.debug(f"Executing query for model: {model_class.__name__}")
        result = await self.db_client.execute_query(query, params)

        if "error" in result:
            return json.dumps(result)

        try:
            raw_nodes = [item[result_key] for item in result.get("data", [])]
            validated_nodes = [model_class.model_validate(node) for node in raw_nodes]
            return json.dumps(
                [node.model_dump(mode='json') for node in validated_nodes],
                indent=2,
                default=str
            )
        except ValidationError as e:
            logger.error(f"Validation error for {model_class.__name__}: {e.errors()}")
            return json.dumps(
                {"error": "Data validation failed", "details": e.errors()},
                default=str
            )
        except KeyError:
            logger.error(f"Validation: Unexpected data structure. Expected key '{result_key}'.")
            return json.dumps({"error": "Data parsing failed",
                               "details": f"Missing key: {result_key}"})
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return json.dumps({"error": "Data parsing failed", "details": str(e)})

    async def get_my_parents(self) -> str:
        """
        Finds the user's parents, mother, father.
        This looks for nodes that have a :PARENT_OF relationship *to* the user.
        """
        logger.info("Tool: get_my_parents")
        query = """
            MATCH (parent)-[:PARENT_OF]->(u:User)
            RETURN properties(parent) AS person
            """
        return await self._query_and_validate_nodes(
            query, {}, PersonDetails, "person"
        )

    async def get_my_children(self) -> str:
        """
        Finds the user's children, kids, offspring.
        This looks for nodes that the user has a :PARENT_OF relationship *to*.
        """
        logger.info("Tool: get_my_children")
        query = """
            MATCH (u:User)-[:PARENT_OF]->(child)
            RETURN properties(child) AS person
            """
        return await self._query_and_validate_nodes(
            query, {}, PersonDetails, "person"
        )

    async def get_my_grandchildren(self) -> str:
        """
        Finds the user's grandchildren (children of the user's children).
        """
        logger.info("Tool: get_my_grandchildren")
        query = """
            MATCH (u:User)-[:PARENT_OF]->(child)-[:PARENT_OF]->(grandchild)
            RETURN DISTINCT properties(grandchild) AS person
            """
        return await self._query_and_validate_nodes(
            query, {}, PersonDetails, "person"
        )

    async def get_my_siblings(self) -> str:
        """
        Finds the user's siblings, brothers, sisters (other children of the user's parents).
        """
        logger.info("Tool: get_my_siblings")
        query = """
            MATCH (parent)-[:PARENT_OF]->(u:User)
            WITH parent, u
            MATCH (parent)-[:PARENT_OF]->(sibling)
            WHERE u <> sibling
            RETURN DISTINCT properties(sibling) AS person
            """
        return await self._query_and_validate_nodes(
            query, {}, PersonDetails, "person"
        )

    async def get_my_spouse(self) -> str:
        """
        Use this specific tool for questions like "Who is my husband?",
        "Who is my wife?", "Do I have a spouse?", or "Who is my partner?".
        It finds the person connected to the :User by a :MARRIED_TO or :PARTNER_OF relationship.
        """
        logger.info("Tool: get_my_spouse")
        query = """
            MATCH (u:User)-[:MARRIED_TO|PARTNER_OF]-(spouse)
            RETURN properties(spouse) AS person
            LIMIT 1
            """
        return await self._query_and_validate_nodes(
            query, {}, PersonDetails, "person"
        )

    # --- NEW: IN-LAW TOOLS ---

    async def get_my_parents_in_law(self) -> str:
        """Finds the user's parents-in-law (the parents of the user's spouse)."""
        logger.info("Tool: get_my_parents_in_law")
        query = """
            MATCH (u:User)-[:MARRIED_TO|PARTNER_OF]-(spouse)<-[:PARENT_OF]-(parent_in_law)
            RETURN DISTINCT properties(parent_in_law) AS person
            """
        return await self._query_and_validate_nodes(
            query, {}, PersonDetails, "person"
        )

    async def get_my_children_in_law(self) -> str:
        """Finds the user's children-in-law (the spouses of the user's children)."""
        logger.info("Tool: get_my_children_in_law")
        query = """
            MATCH (u:User)-[:PARENT_OF]->(child)-[:MARRIED_TO|PARTNER_OF]-(child_in_law)
            RETURN DISTINCT properties(child_in_law) AS person
            """
        return await self._query_and_validate_nodes(
            query, {}, PersonDetails, "person"
        )

    async def get_my_siblings_in_law(self) -> str:
        """
        Finds the user's siblings-in-law, which includes both:
        1. The user's spouse's siblings.
        2. The user's siblings' spouses.
        """
        logger.info("Tool: get_my_siblings_in_law")
        query = """
            // 1. Get spouse's siblings
            MATCH (u:User)-[:MARRIED_TO|PARTNER_OF]-(spouse)<-[:PARENT_OF]-(parent)
            WITH u, spouse, parent
            MATCH (parent)-[:PARENT_OF]->(sibling_in_law)
            WHERE sibling_in_law <> spouse
            RETURN DISTINCT properties(sibling_in_law) AS person

            UNION

            // 2. Get siblings' spouses
            MATCH (u:User)<-[:PARENT_OF]-(parent)-[:PARENT_OF]->(sibling)
            WHERE sibling <> u
            WITH sibling
            MATCH (sibling)-[:MARRIED_TO|PARTNER_OF]-(sibling_in_law)
            RETURN DISTINCT properties(sibling_in_law) AS person
            """
        return await self._query_and_validate_nodes(
            query, {}, PersonDetails, "person"
        )

    # --- END NEW TOOLS ---

    def get_tools(self) -> list:
        """
        Returns a list of all tool methods bound to this instance.
        """
        return [
            StructuredTool.from_function(
                func=None,
                coroutine=self.get_my_parents,
                name="get_my_parents",
                description=self.get_my_parents.__doc__,
                args_schema=NoArgs
            ),
            StructuredTool.from_function(
                func=None,
                coroutine=self.get_my_children,
                name="get_my_children",
                description=self.get_my_children.__doc__,
                args_schema=NoArgs
            ),
            StructuredTool.from_function(
                func=None,
                coroutine=self.get_my_grandchildren,
                name="get_my_grandchildren",
                description=self.get_my_grandchildren.__doc__,
                args_schema=NoArgs
            ),
            StructuredTool.from_function(
                func=None,
                coroutine=self.get_my_siblings,
                name="get_my_siblings",
                description=self.get_my_siblings.__doc__,
                args_schema=NoArgs
            ),
            StructuredTool.from_function(
                func=None,
                coroutine=self.get_my_spouse,
                name="get_my_spouse",
                description=self.get_my_spouse.__doc__,
                args_schema=NoArgs
            ),
            # --- ADD NEW TOOLS TO THE LIST ---
            StructuredTool.from_function(
                func=None,
                coroutine=self.get_my_parents_in_law,
                name="get_my_parents_in_law",
                description=self.get_my_parents_in_law.__doc__,
                args_schema=NoArgs
            ),
            StructuredTool.from_function(
                func=None,
                coroutine=self.get_my_children_in_law,
                name="get_my_children_in_law",
                description=self.get_my_children_in_law.__doc__,
                args_schema=NoArgs
            ),
            StructuredTool.from_function(
                func=None,
                coroutine=self.get_my_siblings_in_law,
                name="get_my_siblings_in_law",
                description=self.get_my_siblings_in_law.__doc__,
                args_schema=NoArgs
            ),
        ]