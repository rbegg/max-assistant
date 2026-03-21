import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from max_assistant.tools.general_query_tools import GeneralQueryTools


def test_parse_cypher_from_markdown_block():
    tools = GeneralQueryTools(db_client=MagicMock(), llm=MagicMock())
    response = "```cypher\nMATCH (n) RETURN n\n```"

    assert tools._parse_cypher_from_response(response) == "MATCH (n) RETURN n"


def test_parse_cypher_from_plain_query():
    tools = GeneralQueryTools(db_client=MagicMock(), llm=MagicMock())
    response = "MATCH (n) RETURN n LIMIT 1"

    assert tools._parse_cypher_from_response(response) == "MATCH (n) RETURN n LIMIT 1"


def test_parse_cypher_from_invalid_response_returns_safe_error_query():
    tools = GeneralQueryTools(db_client=MagicMock(), llm=MagicMock())
    response = "This is not a query"

    parsed = tools._parse_cypher_from_response(response)
    assert parsed == "RETURN 'Error: Could not parse Cypher query from LLM response'"


@pytest.mark.asyncio
async def test_answer_general_question_happy_path():
    db_client = MagicMock()
    db_client.get_schema = AsyncMock(return_value=json.dumps({"node_labels": ["User"]}))
    db_client.execute_query = AsyncMock(return_value={"data": [{"name": "A"}]})

    llm = MagicMock()
    chain = MagicMock()
    chain.ainvoke = AsyncMock(return_value=MagicMock(content="```cypher\nMATCH (n) RETURN n\n```"))
    llm.__ror__ = MagicMock(return_value=chain)

    tools = GeneralQueryTools(db_client=db_client, llm=llm)

    with patch.object(tools, "cypher_generation_chain", chain):
        result = await tools.answer_general_question("who is my father?", '{"user": {"id": 1}}')

    parsed = json.loads(result)
    assert parsed == {"data": [{"name": "A"}]}
    db_client.get_schema.assert_awaited_once()
    db_client.execute_query.assert_awaited_once()


@pytest.mark.asyncio
async def test_answer_general_question_returns_error_when_schema_is_error_json():
    db_client = MagicMock()
    db_client.get_schema = AsyncMock(return_value=json.dumps({"error": "APOC not installed", "message": "nope"}))
    db_client.execute_query = AsyncMock()

    tools = GeneralQueryTools(db_client=db_client, llm=MagicMock())

    result = await tools.answer_general_question("question", "{}")
    parsed = json.loads(result)

    assert parsed["error"] == "Could not retrieve graph schema."
    db_client.execute_query.assert_not_called()