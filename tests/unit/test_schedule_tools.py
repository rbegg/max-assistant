import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from max_assistant.tools.schedule_tools import ScheduleTools


@pytest.mark.asyncio
async def test_get_appointments_for_date_success(sample_user_info):
    db_client = MagicMock()
    db_client.execute_query = AsyncMock(
        return_value={
            "data": [
                {"appointment": {"id": 1, "title": "Dentist", "time": "09:00:00", "date": "2025-01-01"}}
            ]
        }
    )

    tools = ScheduleTools(db_client=db_client, llm=MagicMock())
    result = await tools.get_appointments_for_date("2025-01-01", sample_user_info)

    parsed = json.loads(result)
    assert parsed[0]["id"] == "1"
    assert parsed[0]["title"] == "Dentist"


@pytest.mark.asyncio
async def test_get_routines_for_date_success(sample_user_info):
    db_client = MagicMock()
    db_client.execute_query = AsyncMock(
        return_value={
            "data": [
                {
                    "routine": {
                        "id": 2,
                        "title": "Breakfast",
                        "type": "meal",
                        "dayOfWeek": ["Wednesday"],
                        "time": "08:00:00",
                        "startDate": "2025-01-01",
                    }
                }
            ]
        }
    )

    tools = ScheduleTools(db_client=db_client, llm=MagicMock())
    result = await tools.get_routines_for_date("2025-01-01", sample_user_info)

    parsed = json.loads(result)
    assert parsed[0]["id"] == "2"
    assert parsed[0]["title"] == "Breakfast"


@pytest.mark.asyncio
async def test_get_full_schedule_combines_and_sorts_items(sample_user_info):
    tools = ScheduleTools(db_client=MagicMock(), llm=MagicMock())
    tools.get_appointments_for_date = AsyncMock(
        return_value=json.dumps(
            [
                {"time": "10:00:00", "title": "Meeting", "duration": 30, "details": "Work"},
                {"time": "08:00:00", "title": "Call", "duration": 15, "details": "Family"},
            ]
        )
    )
    tools.get_routines_for_date = AsyncMock(
        return_value=json.dumps(
            [
                {"time": "09:00:00", "title": "Breakfast", "duration": 20, "details": "Home"}
            ]
        )
    )

    result = await tools.get_full_schedule("2025-01-01", sample_user_info)
    parsed = json.loads(result)

    assert [item["title"] for item in parsed] == ["Call", "Breakfast", "Meeting"]


@pytest.mark.asyncio
async def test_create_appointment_returns_db_result(sample_user_info):
    db_client = MagicMock()
    db_client.execute_query = AsyncMock(return_value={"data": [{"new_appointment_id": "abc"}]})

    tools = ScheduleTools(db_client=db_client, llm=MagicMock())
    result = await tools.create_appointment("Visit", "10:00", "2025-01-01", "details", 30, sample_user_info)

    parsed = json.loads(result)
    assert parsed["data"][0]["new_appointment_id"] == "abc"


def test_get_tools_exposes_schedule_tools():
    tools = ScheduleTools(db_client=MagicMock(), llm=MagicMock())
    tool_names = [tool.name for tool in tools.get_tools()]

    assert "get_appointments_for_date" in tool_names
    assert "get_routines_for_date" in tool_names
    assert "get_full_schedule" in tool_names
    assert "create_appointment" in tool_names
    assert "get_activities_info" in tool_names