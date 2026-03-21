import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from max_assistant.tools.gmail_tools import GmailTools


@pytest.mark.asyncio
async def test_create_message_encodes_base64():
    tools = GmailTools(db_client=MagicMock())
    tools.sender_email = "sender@example.com"

    message = tools._create_message("to@example.com", "Subject", "Body")

    assert "raw" in message
    assert isinstance(message["raw"], str)


@pytest.mark.asyncio
async def test_send_message_returns_error_when_sender_missing():
    tools = GmailTools(db_client=MagicMock())
    tools.sender_email = ""

    result = await tools.send_message("to@example.com", "Subject", "Body")
    parsed = json.loads(result)

    assert "error" in parsed


@pytest.mark.asyncio
async def test_send_message_success_path():
    db_client = MagicMock()
    db_client.execute_query = AsyncMock(
        return_value={
            "data": [
                {
                    "refresh_token": "refresh",
                    "access_token": "access",
                    "expiry": datetime.utcnow().isoformat(),
                }
            ]
        }
    )

    creds = MagicMock()
    creds.valid = True
    creds.expired = False
    creds.refresh_token = "refresh"
    creds.token = "access"
    creds.expiry = datetime.utcnow()

    service = MagicMock()
    send_call = MagicMock()
    send_call.execute.return_value = {"id": "msg-1"}
    service.users.return_value.messages.return_value.send.return_value = send_call

    tools = GmailTools(db_client=db_client)
    tools.sender_email = "sender@example.com"
    tools.client_id = "client-id"
    tools.client_secret = "client-secret"

    with patch("max_assistant.tools.gmail_tools.Credentials", return_value=creds), \
         patch("max_assistant.tools.gmail_tools.build", return_value=service):
        result = await tools.send_message("to@example.com", "Hello", "Body")

    parsed = json.loads(result)
    assert parsed["success"] is True
    assert parsed["message_id"] == "msg-1"


@pytest.mark.asyncio
async def test_authenticate_skips_if_token_exists():
    db_client = MagicMock()
    db_client.execute_query = AsyncMock(return_value={"data": [{"token": "already-present"}]})

    tools = GmailTools(db_client=db_client)
    tools.client_id = "client-id"
    tools.client_secret = "client-secret"

    with patch("max_assistant.tools.gmail_tools.InstalledAppFlow") as mock_flow:
        await tools.authenticate()

    mock_flow.from_client_config.assert_not_called()
