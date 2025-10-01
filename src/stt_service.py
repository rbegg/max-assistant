import asyncio
import logging
import os
from fastapi import WebSocket
from websockets.client import connect as websocket_connect
from websockets.exceptions import ConnectionClosed

STT_WEBSOCKET_URL = os.environ.get("STT_WEBSOCKET_URL", "ws://stt/ws")


async def _forward_audio(client_ws: WebSocket, stt_ws):
    """A helper task to forward audio from the client to the STT service."""
    try:
        while True:
            await stt_ws.send(await client_ws.receive_bytes())
    except (ConnectionClosed):
        logging.info("STT connection closed during audio forwarding.")
    except Exception as e:
        logging.error(f"Error forwarding audio: {e}")


async def transcript_generator(client_ws: WebSocket):
    """
    Connects to the STT service and yields transcripts. This function
    hides the complexity of managing the bidirectional stream.
    """
    try:
        async with websocket_connect(STT_WEBSOCKET_URL) as stt_ws:
            logging.info(f"Connected to STT service at {STT_WEBSOCKET_URL}.")

            # Start the concurrent task to forward audio
            forwarder_task = asyncio.create_task(_forward_audio(client_ws, stt_ws))

            # Listen for responses from the STT service
            while True:
                message_str = await stt_ws.recv()
                logging.info(f"Received message from STT: {message_str}")
                yield message_str

            # This part will be reached if the STT connection closes
            forwarder_task.cancel()

    except Exception as e:
        logging.error(f"Error in STT transcript generator: {e}", exc_info=True)