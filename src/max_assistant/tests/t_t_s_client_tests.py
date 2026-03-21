import asyncio
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from max_assistant.clients.tts_client import TTSClient


@pytest.mark.asyncio
async def test_ensure_connected_sets_client():
    client = TTSClient(uri="tcp://tts:10200", retry_delay=0)

    fake_client = MagicMock()
    fake_client.connect = AsyncMock()

    with patch("max_assistant.clients.tts_client.AsyncClient.from_uri", return_value=fake_client):
        await client._ensure_connected()

    assert client._client is fake_client
    fake_client.connect.assert_awaited_once()


@pytest.mark.asyncio
async def test_synthesize_speech_returns_audio_bytes():
    client = TTSClient(uri="tcp://tts:10200", retry_delay=0)

    fake_client = MagicMock()
    fake_client.connect = AsyncMock()
    fake_client.write_event = AsyncMock()

    start_event = MagicMock()
    start_event.type = "audio-start"
    start_event.channels = 1
    start_event.width = 2
    start_event.rate = 16000

    chunk_event = MagicMock()
    chunk_event.type = "audio-chunk"
    chunk_event.audio = b"abc"

    stop_event = MagicMock()
    stop_event.type = "audio-stop"

    fake_client.read_event = AsyncMock(side_effect=[start_event, chunk_event, stop_event])

    with patch("max_assistant.clients.tts_client.AsyncClient.from_uri", return_value=fake_client), \
         patch("max_assistant.clients.tts_client.build", autospec=True, return_value=None), \
         patch("max_assistant.clients.tts_client.wave.open", autospec=True):
        result = await client.synthesize_speech("hello", "voice")

    assert result is not None


@pytest.mark.asyncio
async def test_close_disconnects_and_resets_client():
    client = TTSClient()
    fake_client = MagicMock()
    fake_client.disconnect = AsyncMock()
    client._client = fake_client

    await client.close()

    fake_client.disconnect.assert_awaited_once()
    assert client._client is None
