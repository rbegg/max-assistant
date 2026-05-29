# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
Defines LangGraph tools for finding people and understanding relationships.
"""
import json
import logging
from typing import Dict, Any, Annotated

from langchain_ollama import ChatOllama
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import InjectedState
from pydantic import ValidationError

from max_assistant.clients.neo4j_client import Neo4jClient, Neo4jClientError, Neo4jCircuitBreakerError
from max_assistant.models.person_models import (
    PersonDetails,
)
from max_assistant.models.location_models import LocationDetails
from max_assistant.tools.registry import BaseToolProvider
from max_assistant.utils.decorators import requires_db


logger = logging.getLogger(__name__)


class PersonTools(BaseToolProvider):
    """
    A class that encapsulates person-related tools and holds a
    dedicated Neo4j client instance.
    """

    def __init__(self, db_client: Neo4jClient, llm: ChatOllama = None):
        """
        Initializes the toolset with a specific Neo4j client.
        """
        super().__init__(db_client, llm)
        logger.debug("PersonTools initialized with a Neo4j client.")


    @staticmethod
    def _get_relationship_description(path_data: Dict[str, Any]) -> str:
        """
        Synchronous helper to convert a Neo4j path into a human-readable description.
        Replaces the old _process_relationship_path.
        """
        rel_types = path_data.get('rel_types', [])
        gender = path_data.get('gender')
        num_rels = len(rel_types)
        description = "related"  # Default

        if num_rels == 1:
            rel = rel_types[0]
            if rel == 'MARRIED_TO':
                description = "wife" if gender == 'female' else "husband" if gender == 'male' else "spouse"
            elif rel == 'PARENT_OF':
                description = "mother" if gender == 'female' else "father" if gender == 'male' else "parent"
            elif rel == 'PARTNER_OF':
                description = "partner"
            elif rel == 'FRIEND_OF':
                description = "friend"
            elif rel == 'SUPPORTED_BY':
                description = "support contact"
            elif rel == 'LIVES_WITH':
                description = "lives with"

        elif num_rels == 2:
            if rel_types == ['PARENT_OF', 'PARENT_OF']:
                description = "sister" if gender == 'female' else "brother" if gender == 'male' else "sibling"
            elif rel_types == ['MARRIED_TO', 'PARENT_OF'] or rel_types == ['PARTNER_OF', 'PARENT_OF']:
                description = "mother-in-law" if gender == 'female' else "father-in-law" if gender == 'male' else "parent-in-law"

        elif num_rels > 1:
            # Fallback for more complex paths
            description = f"family ({rel_types[0]})"

        return description

    async def _find_relationship_path(self, person_id: str, user_id: str) -> Dict[str, Any] | None:
        """
        Internal helper to find the shortest relationship path from the: User
        to a person, given that person's unique `id` property.
        """
        params = {
            "person_id": person_id,
            "user_id": user_id,
        }

        # First, check for close family relationships
        family_query = """
            MATCH (u:User {id: $user_id}), (p {id: $person_id})
            MATCH path = shortestPath((u)-[r:MARRIED_TO|PARENT_OF|PARTNER_OF*1..2]-(p))
            RETURN [r IN relationships(path) | type(r)] AS rel_types, p.gender as gender
            ORDER BY length(path) ASC
            LIMIT 1
            """
        try:
            result = await self.db_client.execute_query(family_query, params)
            if result.get("data"):
                logger.debug(f"Found family path for id={person_id}")
                return result["data"][0]

            # If no family path, check for other relationships
            other_query = """
                MATCH (u:User {id: $user_id}), (p {id: $person_id})
                MATCH path = shortestPath((u)-[r:FRIEND_OF|SUPPORTED_BY|LIVES_WITH*1..3]-(p))
                RETURN [r IN relationships(path) | type(r)] AS rel_types, p.gender as gender
                ORDER BY length(path) ASC
                LIMIT 1
                """
            result = await self.db_client.execute_query(other_query, params)
            if result.get("data"):
                logger.debug(f"Found other path for id={person_id}")
                return result["data"][0]

            logger.debug(f"No path found for id={person_id}")
            return None

        # Allow Neo4jCircuitBreakerError exception to caller

        except Neo4jClientError as e:
            logger.warning(f"Database error while searching for relationship path: {e}")
            return None

    @requires_db
    async def find_person_by_name(
            self,
            user_info: Annotated[dict, InjectedState("userinfo")],  # Injected by base class
            first_name: str | None = None,
            last_name: str |  None = None,
    ) -> str:
        """
        Finds a person, family member, friend, or support contact by their first name,
        last name, or both. It returns a list of potential matches with all attributes,
        including phone number, email, notes, and address.
        AND a 'relationship' field describing how they are related to the user.
        At least one name must be provided. Case-insensitive.
        """

        logger.info(f"Tool: find_person_by_name: fn={first_name}, ln={last_name}")

        user_id = self._get_verified_user_id(user_info)

        query = """
            MATCH (u:User {id: $user_id})-[*1..3]-(p:Person|Family|Friend|Support)
            WHERE ($first_name IS NULL OR toLower(p.firstName) CONTAINS $first_name)
              AND ($last_name IS NULL OR toLower(p.lastName) CONTAINS $last_name)
            RETURN properties(p) AS person, labels(p) as labels
            LIMIT 10
            """
        params = {
            "first_name": first_name.lower() if first_name else None,
            "last_name": last_name.lower() if last_name else None,
            "user_id": user_id,
        }

        try:
            result = await self.db_client.execute_query(query, params)

            validated_results = []
            for item in result.get("data", []):
                person_props = item.get("person")
                person_labels = item.get("labels")

                if person_props:
                    validated_person = PersonDetails.model_validate(person_props)
                    person_id = validated_person.id

                    # _find_relationship_path now safely handles DB errors internally
                    path_data = await self._find_relationship_path(person_id, user_id)
                    relationship_desc = self._get_relationship_description(path_data) if path_data else "unknown"

                    validated_results.append({
                        "person": validated_person.model_dump(mode='json'),
                        "labels": person_labels,
                        "relationship": relationship_desc
                    })

            return json.dumps(validated_results, indent=2, default=str)

        except Neo4jCircuitBreakerError as e:
            logger.warning(f"Circuit Breaker blocked find_person_by_name query: {e}")
            return json.dumps({
                "error": "Database_Offline_Circuit_Open",
                "instruction": "The system database is currently offline. Do not attempt further queries. Inform the user you cannot access their data right now.",
                "details": str(e)
            })
        except Neo4jClientError as e:
            logger.error(f"Database error in find_person_by_name: {e}")
            return json.dumps({"error": "Database_Unavailable", "message": str(e)})

        except ValidationError as e:
            logger.error(f"Validation error for PersonDetails: {e.errors()}")
            return json.dumps({"error": "Data validation failed", "details": e.errors()}, default=str)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return json.dumps({"error": "Data parsing failed", "details": str(e)})

    async def find_person_by_title(
            self,
            title: str,
            user_info: Annotated[dict, InjectedState("userinfo")]
    ) -> str:
        """
        Use this tool to find a person by a title, like 'Doctor' or 'Nurse'.
        This tool searches the 'title' field of all Person and Support nodes for a
        partial, case-insensitive match.
        """
        logger.info(f"Tool: find_person_by_title: title={title}")

        user_id = self._get_verified_user_id(user_info)

        query = """
            MATCH (u:User {id: $user_id})-[*1..2]-(p:Person|Support)
            WHERE toLower(p.title) CONTAINS $title
            RETURN properties(p) AS person
            LIMIT 10
            """
        params = {"title": title.lower(), "user_id": user_id}

        return await self._query_and_validate_nodes(
            query=query,
            model_class=PersonDetails,
            result_key="person",
            params=params,
        )

    @requires_db
    async def get_relationship_to_user(
            self,
            first_name: str,
            last_name: str,
            user_info: Annotated[dict, InjectedState("userinfo")]
        ) -> str:
        """
        Use this tool to get the relationship between the user and another person referenced by First and Last Name/
        """

        user_id = self._get_verified_user_id(user_info)

        find_query = """
            MATCH (u:User {id: $user_id})-[*1..3]-(p:Person|Family|Friend|Support)
            WHERE toLower(p.firstName) = $first_name AND toLower(p.lastName) = $last_name
            RETURN p.id AS person_id
            LIMIT 1
            """
        params = {
            "first_name": first_name.lower(),
            "last_name": last_name.lower(),
            "user_id": user_id,
        }

        try:
            find_result = await self.db_client.execute_query(find_query, params)

            if not find_result.get("data"):
                return json.dumps({"error": "Person not found", "details": "No person found with that name."})

            person_id = find_result["data"][0].get("person_id")
            if not person_id:
                return json.dumps(
                    {"error": "Data parsing failed", "details": "Person found, but they have no 'id' property."})

            # 2. Now, find the path using the ID
            path_data = await self._find_relationship_path(person_id, user_id)

            if not path_data:
                return json.dumps(
                    {"error": "No relationship found", "details": "No relationship path was found in the graph."})

            # 3. Process the path
            description = self._get_relationship_description(path_data)

            return json.dumps({
                "relationship": description,
                "path_length": len(path_data.get('rel_types', []))
            }, indent=2)

        except Neo4jCircuitBreakerError as e:
            logger.warning(f"Circuit Breaker blocked get_relationship_to_user query: {e}")
            return json.dumps({
                "error": "Database_Offline_Circuit_Open",
                "instruction": "The system database is currently offline. Do not attempt further queries. Inform the user you cannot access their data right now.",
                "details": str(e)
            })
        except Neo4jClientError as e:
            logger.error(f"Database error in get_relationship_to_user: {e}")
            return json.dumps({"error": "Database_Unavailable", "message": str(e)})
        except Exception as e:
            logger.error(f"Unexpected error in get_relationship_to_user: {e}")
            return json.dumps({"error": "Internal_Error", "details": str(e)})

    @requires_db
    async def get_user_info_internal(self, username: str) -> Dict[str, Any]:
        """
        Internal method to fetch user and location info.
        Returns a dictionary, not a JSON string.
        """
        logger.info("Tool: get_user_info_internal")

        query = """
            MATCH (u:User {userName: $username})
            OPTIONAL MATCH (u)-[:LIVES_AT]->(l:Location)
            RETURN properties(u) AS user, properties(l) AS location
            LIMIT 1
            """
        params = {"username": username}

        try:
            result = await self.db_client.execute_query(query, params)

            if not result.get("data"):
                return {"error": f"User not found.", "details": f"No User node found with userName = {username}"}

            data = result["data"][0]
            user_props = data.get("user")
            location_props = data.get("location")

            if not user_props:
                return {"error": "Data parsing failed", "details": "Found a user relationship but no user properties."}

            validated_user = PersonDetails.model_validate(user_props)
            validated_location = None
            if location_props:
                validated_location = LocationDetails.model_validate(location_props)

            return {
                "user": validated_user.model_dump(mode='json'),
                "location": validated_location.model_dump(mode='json') if validated_location else None
            }

        except Neo4jCircuitBreakerError as e:
            logger.warning(f"Circuit Breaker blocked get_relationship_to_user query: {e}")
            return {"error": "Database_Unavailable", "details": str(e)}
        except Neo4jClientError as e:
            logger.error(f"Database error in get_user_info_internal: {e}")
            return {"error": "Database_Unavailable", "details": str(e)}
        except ValidationError as e:
            logger.error(f"Validation error for User/Location: {e.errors()}")
            return {"error": "User record failed validation.", "details": e.errors()}
        except Exception as e:
            logger.error(f"Unexpected error in get_user_info: {e}")
            return {"error": "Data parsing failed", "details": str(e)}

    def get_tools(self) -> list:
        """
        Returns a list of all tool methods bound to this instance.
        """
        return [
            StructuredTool.from_function(
                name="find_person_by_name",
                func=None,
                coroutine=self.find_person_by_name,  # No .__wrapped__ needed anymore!
                description=self.find_person_by_name.__doc__,
                handle_tool_error=self.format_system_tool_error,
            ),
            StructuredTool.from_function(
                name="find_person_by_title",
                func=None,
                coroutine=self.find_person_by_title,  # No .__wrapped__ needed anymore!
                description=self.find_person_by_title.__doc__,
                handle_tool_error=self.format_system_tool_error,
            ),
            StructuredTool.from_function(
                name="get_relationship_to_user",
                func=None,
                coroutine=self.get_relationship_to_user,
                description=self.get_relationship_to_user.__doc__,
                handle_tool_error=self.format_system_tool_error,
            ),
        ]