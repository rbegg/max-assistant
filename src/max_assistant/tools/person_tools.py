# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
Defines LangGraph tools for finding people and understanding relationships.
"""
import json
import logging
from typing import Optional, Type, Dict, Any

from langchain_core.tools import StructuredTool
from pydantic import ValidationError, BaseModel

from max_assistant.clients.neo4j_client import Neo4jClient
from max_assistant.models.person_models import (
    PersonDetails,
    FindPersonByNameArgs,
    FindPersonByTitleArgs,
    GetRelationshipArgs,
    GetUserInfoArgs
)
from max_assistant.models.location_models import LocationDetails

logger = logging.getLogger(__name__)


class PersonTools:
    """
    A class that encapsulates person-related tools and holds a
    dedicated Neo4j client instance.
    """

    def __init__(self, client: Neo4jClient):
        """
        Initializes the toolset with a specific Neo4j client.
        """
        self.client = client
        logger.info("PersonTools initialized with a Neo4j client.")

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
        """
        logger.debug(f"Executing query with params: {params} for model: {model_class.__name__}")
        result = await self.client.execute_query(query, params)

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
            logger.error(f"Validation: Unexpected data structure from DB. Expected key '{result_key}'.")
            return json.dumps({"error": "Data parsing failed",
                               "details": f"Unexpected data structure from DB. Missing key: {result_key}"})
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return json.dumps({"error": "Data parsing failed", "details": str(e)})

    # --- NEW: REUSABLE RELATIONSHIP HELPERS ---

    def _get_relationship_description(self, path_data: Dict[str, Any]) -> str:
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

    async def _find_relationship_path(self, person_id: str) -> Dict[str, Any] | None:
        """
        Internal helper to find the shortest relationship path from the :User
        to a person, given that person's unique `id` property.
        """
        params = {"person_id": person_id}

        # First, check for close family relationships
        family_query = """
            MATCH (u:User), (p {id: $person_id})
            MATCH path = shortestPath((u)-[r:MARRIED_TO|PARENT_OF|PARTNER_OF*1..2]-(p))
            RETURN [r IN relationships(path) | type(r)] AS rel_types, p.gender as gender
            ORDER BY length(path) ASC
            LIMIT 1
            """
        result = await self.client.execute_query(family_query, params)
        if "error" in result:
            logger.warning(f"Family path query failed: {result['error']}")
            return None
        if result.get("data"):
            logger.debug(f"Found family path for id={person_id}")
            return result["data"][0]

        # If no family path, check for other relationships
        other_query = """
            MATCH (u:User), (p {id: $person_id})
            MATCH path = shortestPath((u)-[r:FRIEND_OF|SUPPORTED_BY|LIVES_WITH*1..3]-(p))
            RETURN [r IN relationships(path) | type(r)] AS rel_types, p.gender as gender
            ORDER BY length(path) ASC
            LIMIT 1
            """
        result = await self.client.execute_query(other_query, params)
        if "error" in result:
            logger.warning(f"Other path query failed: {result['error']}")
            return None
        if result.get("data"):
            logger.debug(f"Found other path for id={person_id}")
            return result["data"][0]

        logger.debug(f"No path found for id={person_id}")
        return None


    async def find_person_by_name(
            self,
            first_name: Optional[str] = None,
            last_name: Optional[str] = None
    ) -> str:
        """
        Finds a person, family member, friend, or support contact by their first name,
        last name, or both. It returns a list of potential matches with all attributes,
        including phone number, email, notes, and address.
        AND a 'relationship' field describing how they are related to the user.
        At least one name must be provided. Case-insensitive.
        """
        logger.info(f"Tool: find_person_by_name: fn={first_name}, ln={last_name}")

        if not first_name and not last_name:
            return json.dumps({"error": "Search failed", "details": "You must provide at least a first or last name."})

        query = """
            MATCH (p:Person|Family|Friend|Support)
            WHERE ($first_name IS NULL OR toLower(p.firstName) CONTAINS $first_name)
              AND ($last_name IS NULL OR toLower(p.lastName) CONTAINS $last_name)
            RETURN properties(p) AS person, labels(p) as labels
            LIMIT 10
            """
        params = {
            "first_name": first_name.lower() if first_name else None,
            "last_name": last_name.lower() if last_name else None
        }

        result = await self.client.execute_query(query, params)

        if "error" in result:
            return json.dumps(result)

        try:
            validated_results = []
            for item in result.get("data", []):
                person_props = item.get("person")
                person_labels = item.get("labels")

                if person_props:
                    # 1. Validate the person
                    validated_person = PersonDetails.model_validate(person_props)

                    # 2. Find their relationship to the user using the new helpers
                    person_id = validated_person.id
                    path_data = await self._find_relationship_path(person_id)
                    relationship_desc = "unknown"  # Default
                    if path_data:
                        relationship_desc = self._get_relationship_description(path_data)

                    # 3. Add the new 'relationship' field to the output
                    validated_results.append({
                        "person": validated_person.model_dump(mode='json'),
                        "labels": person_labels,
                        "relationship": relationship_desc
                    })

            return json.dumps(validated_results, indent=2, default=str)

        except ValidationError as e:
            logger.error(f"Validation error for PersonDetails: {e.errors()}")
            return json.dumps(
                {"error": "Data validation failed", "details": e.errors()},
                default=str
            )
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return json.dumps({"error": "Data parsing failed", "details": str(e)})


    async def find_person_by_title(self, title: str) -> str:
        """
        Use this tool to find a person by a title, like 'Doctor' or 'Nurse'.
        This tool searches the 'title' field of all Person and Support nodes for a
        partial, case-insensitive match.
        """
        logger.info(f"Tool: find_person_by_title: title={title}")

        query = """
            MATCH (p:Person|Support)
            WHERE toLower(p.title) CONTAINS $title
            RETURN properties(p) AS person
            LIMIT 10
            """
        params = {"title": title.lower()}

        return await self._query_and_validate_nodes(
            query,
            params,
            model_class=PersonDetails,
            result_key="person"
        )


    async def get_relationship_to_user(self, first_name: str, last_name: str) -> str:
        """
        Use this tool ONLY when the user provides a full name and asks
        for their relationship (e.g., "Who is Jane Doe?").
        It returns a description of their relationship to the user (e.g., 'friend', 'doctor').

        DO NOT use this tool for general questions like 'who is my husband' or 'who are my parents'.
        Use the specific family_tools (like get_my_spouse) for those queries.
        """
        logger.info(f"Tool: get_relationship_to_user for {first_name} {last_name}")

        # 1. First, find the person's ID.
        find_query = """
            MATCH (p:Person|Family|Friend|Support)
            WHERE toLower(p.firstName) = $first_name AND toLower(p.lastName) = $last_name
            RETURN p.id AS person_id
            LIMIT 1
            """
        params = {
            "first_name": first_name.lower(),
            "last_name": last_name.lower()
        }
        find_result = await self.client.execute_query(find_query, params)

        if "error" in find_result:
            return json.dumps(find_result)
        if not find_result.get("data"):
            return json.dumps({"error": "Person not found", "details": "No person found with that name."})

        person_id = find_result["data"][0].get("person_id")
        if not person_id:
            return json.dumps(
                {"error": "Data parsing failed", "details": "Person found, but they have no 'id' property."})

        # 2. Now, find the path using the ID
        path_data = await self._find_relationship_path(person_id)

        if not path_data:
            return json.dumps(
                {"error": "No relationship found", "details": "No relationship path was found in the graph."})

        # 3. Process the path
        description = self._get_relationship_description(path_data)

        return json.dumps({
            "relationship": description,
            "path_length": len(path_data.get('rel_types', []))
        }, indent=2)


    async def get_user_info_internal(self) -> Dict[str, Any]:
        """
        Internal method to fetch user and location info.
        Returns a dictionary, not a JSON string.
        """
        logger.info("Tool: get_user_info_internal")

        query = """
            MATCH (u:User)
            OPTIONAL MATCH (u)-[:LIVES_AT]->(l:Location)
            RETURN properties(u) AS user, properties(l) AS location
            LIMIT 1
            """
        result = await self.client.execute_query(query, {})

        if "error" in result:
            return result
        if not result.get("data"):
            return {"error": "User not found", "details": "No :User node was found in the graph."}

        try:
            data = result["data"][0]
            user_props = data.get("user")
            location_props = data.get("location")

            if not user_props:
                return {"error": "Data parsing failed", "details": "Found a user relationship but no user properties."}

            validated_user = PersonDetails.model_validate(user_props)
            validated_location = None
            if location_props:
                validated_location = LocationDetails.model_validate(location_props)

            output = {
                "user": validated_user.model_dump(mode='json'),
                "location": validated_location.model_dump(mode='json') if validated_location else None
            }
            return output
        except ValidationError as e:
            logger.error(f"Validation error for User/Location: {e.errors()}")
            return {"error": "Data validation failed", "details": e.errors()}
        except Exception as e:
            logger.error(f"Unexpected error in get_user_info: {e}")
            return {"error": "Data parsing failed", "details": str(e)}

    async def get_user_info(self) -> str:
        """
        Fetches the primary User node and their LIVES_AT location.
        It takes no arguments and assumes a single User node in the graph.
        Returns the user's properties and their location's properties as a JSON string.
        This data is cached in the state context Userinfo, use that data instead of this tool.
        """
        output_dict = await self.get_user_info_internal()
        return json.dumps(output_dict, indent=2, default=str)


    def get_tools(self) -> list:
        """
        Returns a list of all tool methods bound to this instance.
        """
        return [
            StructuredTool.from_function(
                func=None,
                coroutine=self.find_person_by_name,
                name="find_person_by_name",
                description=self.find_person_by_name.__doc__,
                args_schema=FindPersonByNameArgs
            ),
            StructuredTool.from_function(
                func=None,
                coroutine=self.find_person_by_title,
                name="find_person_by_title",
                description=self.find_person_by_title.__doc__,
                args_schema=FindPersonByTitleArgs
            ),
            # StructuredTool.from_function(
            #     func=None,
            #     coroutine=self.get_relationship_to_user,
            #     name="get_relationship_to_user",
            #     description=self.get_relationship_to_user.__doc__,
            #     args_schema=GetRelationshipArgs
            # ),
            StructuredTool.from_function(
                func=None,
                coroutine=self.get_user_info,
                name="get_user_info",
                description=self.get_user_info.__doc__,
                args_schema=GetUserInfoArgs
            ),
        ]