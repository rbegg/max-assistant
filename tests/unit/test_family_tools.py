import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from max_assistant.tools.family_tools import FamilyTools
from max_assistant.clients.neo4j_client import Neo4jCircuitBreakerError


@pytest.mark.asyncio
async def test_get_my_parents_returns_json_list(sample_user_info):
    db_client = MagicMock()
    db_client.execute_query = AsyncMock(
        return_value={
            "data": [
                {"person": {"id": 1, "firstName": "Alice", "lastName": "Doe"}}
            ]
        }
    )

    tools = FamilyTools(db_client=db_client, llm=MagicMock())
    result = await tools.get_my_parents(sample_user_info)

    parsed = json.loads(result)
    assert parsed[0]["id"] == "1"
    assert parsed[0]["firstName"] == "Alice"


@pytest.mark.asyncio
async def test_get_my_children_returns_error_db_offline(sample_user_info):
    db_client = MagicMock()

    neo4j_fault = Neo4jCircuitBreakerError("Database_Offline_Circuit_Open")

    db_client.execute_query = AsyncMock(side_effect = neo4j_fault)

    tools = FamilyTools(db_client=db_client, llm=MagicMock())
    result = await tools.get_my_children(sample_user_info)

    parsed = json.loads(result)
    assert parsed["error"] == "Database_Offline_Circuit_Open"


def test_get_tools_exposes_expected_names():
    """
    Verifies the complete contractual surface area of FamilyTools.
    Ensures no tools are accidentally dropped, renamed, or leaked.
    """
    tools = FamilyTools(db_client=MagicMock(), llm=MagicMock())
    tool_names = {tool.name for tool in tools.get_tools()}

    # Define the exact expected schema footprint
    expected_tools = {
        "get_my_parents",
        "get_my_children",
        "get_my_grandchildren",
        "get_my_spouse",
        "get_my_siblings",
        "get_my_siblings_in_law",
        "get_my_parents_in_law",
        "get_my_children_in_law"
    }

    # A set comparison ensures an exact 1:1 match, regardless of order
    assert tool_names == expected_tools


@pytest.mark.asyncio
async def test_get_my_siblings_in_law_returns_list(sample_user_info):
    db_client = MagicMock()
    db_client.execute_query = AsyncMock(
        return_value={
            "data": [
                {"person": {"id": 3, "firstName": "Sam", "lastName": "Doe"}}
            ]
        }
    )

    tools = FamilyTools(db_client=db_client, llm=MagicMock())
    result = await tools.get_my_siblings_in_law(sample_user_info)

    parsed = json.loads(result)
    assert parsed[0]["id"] == "3"