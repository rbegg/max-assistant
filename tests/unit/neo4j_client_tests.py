import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from max_assistant.clients.neo4j_client import Neo4jClient


@pytest.mark.asyncio
async def test_execute_query_returns_data_and_summary():
    counters = SimpleNamespace(
        nodes_created=1,
        nodes_deleted=0,
        relationships_created=0,
        relationships_deleted=0,
        properties_set=2,
    )
    summary = SimpleNamespace(counters=counters)
    record = MagicMock()
    record.data.return_value = {"name": "Alice"}

    driver = MagicMock()
    driver.execute_query = AsyncMock(
        return_value=SimpleNamespace(records=[record], summary=summary)
    )

    client = Neo4jClient(driver=driver, database="neo4j")
    result = await client.execute_query("MATCH (n) RETURN n", {})

    assert result["data"] == [{"name": "Alice"}]
    assert result["summary"]["nodes_created"] == 1
    assert result["summary"]["properties_set"] == 2


@pytest.mark.asyncio
async def test_execute_query_returns_error_when_driver_missing():
    client = Neo4jClient(driver=None, database="neo4j")

    result = await client.execute_query("MATCH (n) RETURN n", {})

    assert result == {"error": "Neo4j is not connected"}


@pytest.mark.asyncio
async def test_get_schema_returns_cached_value_on_second_call():
    record = MagicMock()
    record.data.return_value = {
        "value": {
            "User": {
                "type": "node",
                "properties": {"id": {"type": "STRING"}},
                "relationships": {},
            }
        }
    }

    driver = MagicMock()
    driver.execute_query = AsyncMock(
        return_value=SimpleNamespace(records=[record])
    )

    client = Neo4jClient(driver=driver, database="neo4j")

    first = await client.get_schema()
    second = await client.get_schema()

    assert first == second
    assert driver.execute_query.await_count == 1


@pytest.mark.asyncio
async def test_get_schema_handles_missing_apoc():
    from neo4j.exceptions import Neo4jError

    error = Neo4jError("There is no procedure with the name `apoc.meta.schema`")
    driver = MagicMock()
    driver.execute_query = AsyncMock(side_effect=error)

    client = Neo4jClient(driver=driver, database="neo4j")
    result = await client.get_schema()

    parsed = json.loads(result)
    assert parsed["error"] == "APOC not installed"
