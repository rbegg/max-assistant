import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from max_assistant.clients.stt_client import STTClient


@pytest.mark.asyncio
async def test_forward_audio_sends_chunks_until_shutdown():
    audio_queue = asyncio.Queue()
    shutdown_event = asyncio.Event()
    ws = MagicMock()
    ws.send = AsyncMock()

    await audio_queue.put(b"chunk-1")
    shutdown_event.set()

    await STTClient._forward_audio(audio_queue, ws, shutdown_event)

    ws.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_transcript_generator_yields_messages():
    audio_queue = asyncio.Queue()
    shutdown_event = asyncio.Event()

    ws = MagicMock()
    ws.recv = AsyncMock(side_effect=["{\"data\": \"hello\"}", asyncio.CancelledError()])
    ws.__aenter__.return_value = ws
    ws.__aexit__.return_value = None

    connect_cm = MagicMock()
    connect_cm.__aenter__.return_value = ws
    connect_cm.__aexit__.return_value = None

    client = STTClient(uri="ws://test", retry_delay=0)

    with patch("max_assistant.clients.stt_client.websocket_connect", return_value=connect_cm), \
         patch.object(STTClient, "_forward_audio", new=AsyncMock()):
        gen = client.transcript_generator(audio_queue, shutdown_event)
        msg = await gen.__anext__()

    assert msg == "{\"data\": \"hello\"}"


@pytest.mark.asyncio
async def test_transcript_generator_retries_on_connection_refused():
    audio_queue = asyncio.Queue()
    shutdown_event = asyncio.Event()
    client = STTClient(uri="ws://test", retry_delay=0)

    with patch("max_assistant.clients.stt_client.websocket_connect", side_effect=ConnectionRefusedError), \
         patch("asyncio.sleep", new=AsyncMock()):
        task = asyncio.create_task(client.transcript_generator(audio_queue, shutdown_event).__anext__())
        await asyncio.sleep(0)
        task.cancel()
