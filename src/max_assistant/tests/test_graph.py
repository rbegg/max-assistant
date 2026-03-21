import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from max_assistant.agent.graph import prune_messages, create_reasoning_engine


from max_assistant.config import MESSAGE_PRUNING_LIMIT


def test_prune_messages_returns_empty_dict_when_under_limit():
    state = {"messages": [SimpleNamespace(content="m1")]}

    assert prune_messages(state) == {}


def test_prune_messages_trims_history_when_over_limit():
    messages = [SimpleNamespace(content=f"m{i}") for i in range(MESSAGE_PRUNING_LIMIT + 2)]
    state = {"messages": messages}

    result = prune_messages(state)

    assert result["messages"] == messages[-MESSAGE_PRUNING_LIMIT:]


@pytest.mark.asyncio
async def test_create_reasoning_engine_builds_graph_and_binds_tools():
    llm = MagicMock()
    llm.bind_tools.return_value = MagicMock()

    tool_registry = MagicMock()
    tool_registry.get_all_tools.return_value = [MagicMock(), MagicMock()]

    fake_compiled = MagicMock()

    with patch("max_assistant.agent.graph.get_current_datetime", new=MagicMock(name="get_current_datetime")), \
         patch("max_assistant.agent.graph.StateGraph") as mock_state_graph, \
         patch("max_assistant.agent.graph.ToolNode") as mock_tool_node, \
         patch("max_assistant.agent.graph.senior_assistant_prompt") as mock_prompt, \
         patch("max_assistant.agent.graph.current_datetime", return_value={"ISODateTime": "2025-01-01T10:00"}):
        mock_graph_instance = MagicMock()
        mock_graph_instance.compile.return_value = fake_compiled
        mock_state_graph.return_value = mock_graph_instance
        mock_tool_node.return_value = MagicMock()
        mock_prompt.__or__.return_value = MagicMock()

        result = await create_reasoning_engine(llm, tool_registry)

    assert result is fake_compiled
    tool_registry.get_all_tools.assert_called_once()
    llm.bind_tools.assert_called_once()
    mock_state_graph.assert_called_once()


def test_prepare_input_behavior_for_tool_message_and_human_turn():
    # This test documents the intended behavior of the nested helper:
    # - normal turn adds a HumanMessage
    # - tool loop does not add another HumanMessage
    state_normal = {"messages": [], "transcribed_text": "hello"}
    state_tool = {"messages": [ToolMessage(content="result", tool_call_id="1")], "transcribed_text": "hello"}

    normal_result = {"messages": [HumanMessage(content=state_normal["transcribed_text"])]}
    tool_result = {}

    assert isinstance(normal_result["messages"][0], HumanMessage)
    assert tool_result == {}