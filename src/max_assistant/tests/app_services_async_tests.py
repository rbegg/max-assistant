from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from max_assistant.app_services import AppServices


@pytest.mark.asyncio
async def test_initialize_clients_returns_db_and_llm():
    fake_db = MagicMock()
    fake_llm = MagicMock()

    with patch("max_assistant.app_services.Neo4jClient.create", new=AsyncMock(return_value=fake_db)), \
         patch("max_assistant.app_services.create_llm_instance", return_value=fake_llm), \
         patch("max_assistant.app_services.preload_model_async", new=AsyncMock()):
        db_client, llm = await AppServices._initialize_clients(MagicMock())

    assert db_client is fake_db
    assert llm is fake_llm


@pytest.mark.asyncio
async def test_fetch_user_info_uses_person_tools():
    db_client = MagicMock()
    fake_user_info = {"user": {"id": "1"}}

    with patch("max_assistant.app_services.PersonTools") as mock_person_tools:
        instance = mock_person_tools.return_value
        instance.get_user_info_internal = AsyncMock(return_value=fake_user_info)

        result = await AppServices._fetch_user_info(db_client)

    assert result == fake_user_info
    mock_person_tools.assert_called_once_with(db_client)
    instance.get_user_info_internal.assert_awaited_once()


def test_initialize_tool_registry_registers_providers():
    db_client = MagicMock()
    llm = MagicMock()

    fake_registry = MagicMock()
    with patch("max_assistant.app_services.ToolRegistry", return_value=fake_registry), \
         patch("max_assistant.app_services.ALL_TOOL_PROVIDERS", [MagicMock(), MagicMock()]):
        result = AppServices._initialize_tool_registry(db_client, llm)

    assert result is fake_registry
    assert fake_registry.register_provider.call_count == 2
