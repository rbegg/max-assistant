import asyncio
import logging
import os
from websockets.client import connect as websocket_connect
from websockets.exceptions import ConnectionClosed

STT_WEBSOCKET_URL = os.environ.get("STT_WEBSOCKET_URL", "ws://stt/ws")


async def _forward_audio(audio_queue: asyncio.Queue, stt_ws):
    """A helper task to forward audio from an asyncio Queue to the STT service."""
    try:
        while True:
            audio_chunk = await audio_queue.get()
            await stt_ws.send(audio_chunk)
    except (ConnectionClosed):
        logging.info("STT connection closed during audio forwarding.")
    except Exception as e:
        logging.error(f"Error forwarding audio: {e}")


async def transcript_generator(audio_queue: asyncio.Queue):
    """
    Connects to the STT service and yields transcripts. This function
    hides the complexity of managing the bidirectional stream.
    """
    retry_delay = 5  # seconds
    while True:
        forwarder_task = None
        try:
            async with websocket_connect(STT_WEBSOCKET_URL) as stt_ws:
                logging.info(f"Connected to STT service at {STT_WEBSOCKET_URL}.")

                # Start the concurrent task to forward audio from the queue
                forwarder_task = asyncio.create_task(_forward_audio(audio_queue, stt_ws))

                # Listen for responses from the STT service
                while True:
                    message_str = await stt_ws.recv()
                    logging.info(f"Received message from STT: {message_str}")
                    yield message_str

        except (ConnectionRefusedError, ConnectionClosed):
            logging.warning(f"Connection to STT service failed or was lost. Retrying in {retry_delay} seconds...")
            await asyncio.sleep(retry_delay)
        except Exception as e:
            logging.error(f"An unexpected error occurred in the transcript generator: {e}", exc_info=True)
            break  # Stop retrying for unexpected errors
        finally:
            if forwarder_task:
                forwarder_task.cancel()