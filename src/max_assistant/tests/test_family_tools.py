import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from max_assistant.tools.family_tools import FamilyTools


@pytest.mark.asyncio
async def test_get_my_parents_returns_json_list():
    db_client = MagicMock()
    db_client.execute_query = AsyncMock(
        return_value={
            "data": [
                {"person": {"id": 1, "firstName": "Alice", "lastName": "Doe"}}
            ]
        }
    )

    tools = FamilyTools(db_client=db_client, llm=MagicMock())
    result = await tools.get_my_parents()

    parsed = json.loads(result)
    assert parsed[0]["id"] == "1"
    assert parsed[0]["firstName"] == "Alice"


@pytest.mark.asyncio
async def test_get_my_children_returns_error_passthrough():
    db_client = MagicMock()
    db_client.execute_query = AsyncMock(return_value={"error": "Neo4jError", "message": "boom"})

    tools = FamilyTools(db_client=db_client, llm=MagicMock())
    result = await tools.get_my_children()

    parsed = json.loads(result)
    assert parsed["error"] == "Neo4jError"
    assert parsed["message"] == "boom"


def test_get_tools_exposes_expected_names():
    tools = FamilyTools(db_client=MagicMock(), llm=MagicMock())
    tool_names = [tool.name for tool in tools.get_tools()]

    assert "get_my_parents" in tool_names
    assert "get_my_children" in tool_names
    assert "get_my_spouse" in tool_names
    assert "get_my_siblings_in_law" in tool_names


@pytest.mark.asyncio
async def test_get_my_siblings_in_law_returns_list():
    db_client = MagicMock()
    db_client.execute_query = AsyncMock(
        return_value={
            "data": [
                {"person": {"id": 3, "firstName": "Sam", "lastName": "Doe"}}
            ]
        }
    )

    tools = FamilyTools(db_client=db_client, llm=MagicMock())
    result = await tools.get_my_siblings_in_law()

    parsed = json.loads(result)
    assert parsed[0]["id"] == "3"