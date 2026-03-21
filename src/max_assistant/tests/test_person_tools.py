import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from max_assistant.models.person_models import PersonDetails
from max_assistant.tools.person_tools import PersonTools


@pytest.mark.asyncio
async def test_find_person_by_title_success():
    db_client = MagicMock()
    db_client.execute_query = AsyncMock(
        return_value={
            "data": [
                {
                    "person": {
                        "id": 1,
                        "firstName": "Jane",
                        "lastName": "Doe",
                    }
                }
            ]
        }
    )

    tools = PersonTools(db_client=db_client, llm=MagicMock())
    result = await tools.find_person_by_title("doctor")

    parsed = json.loads(result)
    assert parsed[0]["id"] == "1"
    assert parsed[0]["firstName"] == "Jane"
    assert parsed[0]["lastName"] == "Doe"


@pytest.mark.asyncio
async def test_find_person_by_name_requires_input():
    tools = PersonTools(db_client=MagicMock(), llm=MagicMock())

    result = await tools.find_person_by_name()

    parsed = json.loads(result)
    assert parsed["error"] == "Search failed"


@pytest.mark.asyncio
async def test_get_user_info_internal_success():
    db_client = MagicMock()
    db_client.execute_query = AsyncMock(
        return_value={
            "data": [
                {
                    "user": {"id": 1, "firstName": "Max", "lastName": "Assistant"},
                    "location": {"id": 2, "name": "Home"},
                }
            ]
        }
    )

    tools = PersonTools(db_client=db_client, llm=MagicMock())
    result = await tools.get_user_info_internal()

    assert result["user"]["id"] == "1"
    assert result["user"]["firstName"] == "Max"
    assert result["location"]["name"] == "Home"


@pytest.mark.asyncio
async def test_get_relationship_description_mapping():
    tools = PersonTools(db_client=MagicMock(), llm=MagicMock())

    assert tools._get_relationship_description({"rel_types": ["MARRIED_TO"], "gender": "female"}) == "wife"
    assert tools._get_relationship_description({"rel_types": ["MARRIED_TO"], "gender": "male"}) == "husband"
    assert tools._get_relationship_description({"rel_types": ["PARENT_OF"], "gender": "female"}) == "mother"
    assert tools._get_relationship_description({"rel_types": ["PARENT_OF"], "gender": "male"}) == "father"
    assert tools._get_relationship_description({"rel_types": ["FRIEND_OF"], "gender": "female"}) == "friend"