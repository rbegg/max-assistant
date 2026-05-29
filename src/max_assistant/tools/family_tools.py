# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
Defines LangGraph tools for querying the User's family tree.
"""
import logging
from typing import Annotated

from langchain_core.tools import StructuredTool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import InjectedState

from max_assistant.clients.neo4j_client import Neo4jClient
from max_assistant.models.person_models import PersonDetails
from max_assistant.tools.base_provider import BaseToolProvider


logger = logging.getLogger(__name__)

# noinspection GrazieInspection
class FamilyTools(BaseToolProvider):
    """
    A class that encapsulates family-tree-related tools,
    with all queries relative to the :User node.
    """

    def __init__(self, db_client: Neo4jClient, llm: ChatOllama | None = None):
        """
        Initializes the toolset with a specific Neo4j client.
        """
        super().__init__(db_client, llm)
        logger.debug("FamilyTools initialized with a Neo4j client.")


    async def get_my_parents(self, user_info: Annotated[dict, InjectedState("userinfo")]) -> str:
        """
        Finds the user's parents, mother, father.
        This looks for nodes that have a :PARENT_OF relationship *to* the user.
        """
        logger.info("Tool: get_my_parents")

        user_id = self._get_verified_user_id(user_info)

        query = """
            MATCH (u:User {id: $user_id})<-[:PARENT_OF]-(parent)
            RETURN properties(parent) AS person
            """
        params = {"user_id": user_id}
        logger.info(f"Tool: get_my_parents, Params = {params}")

        return await self._query_and_validate_nodes(
            query=query,
            model_class= PersonDetails,
            result_key="person",
            params=params,
        )

    async def get_my_children(self, user_info: Annotated[dict, InjectedState("userinfo")]) -> str:
        """
        Finds the user's children, kids, offspring.
        This looks for nodes that the user has a :PARENT_OF relationship *to*.
        """
        logger.info("Tool: get_my_children")

        user_id = self._get_verified_user_id(user_info)

        query = """
            MATCH (u:User {id: $user_id})-[:PARENT_OF]->(child)
            RETURN properties(child) AS person
            """
        params = {"user_id": user_id}
        logger.info(f"Tool: get_my_children, Params = {params}")

        return await self._query_and_validate_nodes(
            query=query,
            model_class=PersonDetails,
            result_key="person",
            params=params,
        )

    async def get_my_grandchildren(self, user_info: Annotated[dict, InjectedState("userinfo")] ) -> str:
        """
        Finds the user's grandchildren (children of the user's children).
        """
        logger.info("Tool: get_my_grandchildren")

        user_id = self._get_verified_user_id(user_info)

        query = """
            MATCH (u:User {id: $user_id})-[:PARENT_OF]->(child)-[:PARENT_OF]->(grandchild)
            RETURN DISTINCT properties(grandchild) AS person
            """
        params = {"user_id": user_id}

        return await self._query_and_validate_nodes(
            query=query,
            model_class=PersonDetails,
            result_key="person",
            params=params,
        )

    async def get_my_siblings(self, user_info: Annotated[dict, InjectedState("userinfo")]) -> str:
        """
        Finds the user's siblings, brothers, sisters (other children of the user's parents).
        """
        logger.info("Tool: get_my_siblings")

        user_id = self._get_verified_user_id(user_info)

        query = """
            MATCH (parent)-[:PARENT_OF]->(u:User {id: $user_id})
            WITH parent, u
            MATCH (parent)-[:PARENT_OF]->(sibling)
            WHERE u <> sibling
            RETURN DISTINCT properties(sibling) AS person
            """
        params = {"user_id": user_id}

        return await self._query_and_validate_nodes(
            query=query,
            model_class=PersonDetails,
            result_key="person",
            params=params,
        )

    async def get_my_spouse(self, user_info: Annotated[dict, InjectedState("userinfo")]) -> str:
        """
        Use this specific tool for questions like "Who is my husband?",
        "Who is my wife?", "Do I have a spouse?", or "Who is my partner?".
        It finds the person connected to the :User by a :MARRIED_TO or :PARTNER_OF relationship.
        """
        logger.info("Tool: get_my_spouse")

        user_id = self._get_verified_user_id(user_info)

        query = """
            MATCH (u:User {id: $user_id})-[:MARRIED_TO|PARTNER_OF]-(spouse)
            RETURN properties(spouse) AS person
            LIMIT 1
            """
        params = {"user_id": user_id}

        return await self._query_and_validate_nodes(
            query=query,
            model_class=PersonDetails,
            result_key="person",
            params=params,
        )

    async def get_my_parents_in_law(self, user_info: Annotated[dict, InjectedState("userinfo")]) -> str:
        """Finds the user's parents-in-law (the parents of the user's spouse)."""
        logger.info("Tool: get_my_parents_in_law")

        user_id = self._get_verified_user_id(user_info)

        query = """
            MATCH (u:User {id: $user_id})-[:MARRIED_TO|PARTNER_OF]-(spouse)<-[:PARENT_OF]-(parent_in_law)
            RETURN DISTINCT properties(parent_in_law) AS person
            """
        params = {"user_id": user_id}

        return await self._query_and_validate_nodes(
            query=query,
            model_class=PersonDetails,
            result_key="person",
            params=params,
        )

    async def get_my_children_in_law(self, user_info: Annotated[dict, InjectedState("userinfo")]) -> str:
        """Finds the user's children-in-law (the spouses of the user's children)."""
        logger.info("Tool: get_my_children_in_law")

        user_id = self._get_verified_user_id(user_info)

        query = """
            MATCH (u:User {id: $user_id})-[:PARENT_OF]->(child)-[:MARRIED_TO|PARTNER_OF]-(child_in_law)
            RETURN DISTINCT properties(child_in_law) AS person
            """
        params = {"user_id": user_id}

        return await self._query_and_validate_nodes(
            query=query,
            model_class=PersonDetails,
            result_key="person",
            params=params,
        )

    async def get_my_siblings_in_law(self, user_info: Annotated[dict, InjectedState("userinfo")]) -> str:
        """
        Finds the user's siblings-in-law, which includes both:
        1. The user's spouse's siblings.
        2. The user's siblings' spouses.
        """
        logger.info("Tool: get_my_siblings_in_law")

        user_id = self._get_verified_user_id(user_info)

        query = """
            // 1. Get spouse's siblings
            MATCH (u:User {id: $user_id})
                -[:MARRIED_TO|PARTNER_OF]-(spouse)<-[:PARENT_OF]-(parent)-[:PARENT_OF]->(sibling_in_law)
            WHERE sibling_in_law <> spouse
            RETURN DISTINCT properties(sibling_in_law) AS person
            LIMIT 25
        
            UNION
        
            MATCH (u:User {id: $user_id})<-[:PARENT_OF]-(parent)-[:PARENT_OF]->(sibling)
            WHERE sibling <> u
            MATCH (sibling)-[:MARRIED_TO|PARTNER_OF]-(sibling_in_law)
            RETURN DISTINCT properties(sibling_in_law) AS person
            LIMIT 25
            """
        params = {"user_id": user_id}

        return await self._query_and_validate_nodes(
            query=query,
            model_class=PersonDetails,
            result_key="person",
            params=params,
        )

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
                handle_tool_error=self.format_system_tool_error,
            ),
            StructuredTool.from_function(
                func=None,
                coroutine=self.get_my_children,
                name="get_my_children",
                description=self.get_my_children.__doc__,
                handle_tool_error=self.format_system_tool_error,
            ),
            StructuredTool.from_function(
                func=None,
                coroutine=self.get_my_grandchildren,
                name="get_my_grandchildren",
                description=self.get_my_grandchildren.__doc__,
                handle_tool_error=self.format_system_tool_error,
            ),
            StructuredTool.from_function(
                func=None,
                coroutine=self.get_my_siblings,
                name="get_my_siblings",
                description=self.get_my_siblings.__doc__,
                handle_tool_error=self.format_system_tool_error,
            ),
            StructuredTool.from_function(
                func=None,
                coroutine=self.get_my_spouse,
                name="get_my_spouse",
                description=self.get_my_spouse.__doc__,
                handle_tool_error=self.format_system_tool_error,
            ),
            StructuredTool.from_function(
                func=None,
                coroutine=self.get_my_parents_in_law,
                name="get_my_parents_in_law",
                description=self.get_my_parents_in_law.__doc__,
                handle_tool_error=self.format_system_tool_error,
            ),
            StructuredTool.from_function(
                func=None,
                coroutine=self.get_my_children_in_law,
                name="get_my_children_in_law",
                description=self.get_my_children_in_law.__doc__,
                handle_tool_error=self.format_system_tool_error,
            ),
            StructuredTool.from_function(
                func=None,
                coroutine=self.get_my_siblings_in_law,
                name="get_my_siblings_in_law",
                description=self.get_my_siblings_in_law.__doc__,
                handle_tool_error=self.format_system_tool_error,
            ),
        ]