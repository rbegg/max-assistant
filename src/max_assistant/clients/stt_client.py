import asyncio
import logging
from websockets.asyncio.client import connect as websocket_connect
from websockets.exceptions import ConnectionClosed
from max_assistant.config import STT_WEBSOCKET_URL

logger = logging.getLogger(__name__)


class STTClient:
    """
    Manages a connection to the STT service and provides a transcript generator.
    """

    def __init__(self, uri: str = STT_WEBSOCKET_URL, retry_delay: int = 5):
        self.uri = uri
        self.retry_delay = retry_delay

    @staticmethod
    async def _forward_audio(audio_queue: asyncio.Queue, stt_ws):
        """A helper task to forward audio from an asyncio Queue to the STT service."""
        try:
            while True:
                audio_chunk = await audio_queue.get()
                await stt_ws.send(audio_chunk)
        except ConnectionClosed:
            logger.info("STT connection closed during audio forwarding.")
        except asyncio.CancelledError:
            logger.info("Audio forwarding task cancelled.")
        except Exception as e:
            logging.error(f"Error forwarding audio: {e}")

    async def transcript_generator(self, audio_queue: asyncio.Queue):
        """
        Connects to the STT service and yields transcripts.
        This generator handles connection and reconnection logic.
        """
        while True:
            forwarder_task = None
            try:
                async with websocket_connect(self.uri) as stt_ws:
                    logger.info(f"Connected to STT service at {self.uri}.")

                    # Start the concurrent task to forward audio from the queue
                    forwarder_task = asyncio.create_task(
                        self._forward_audio(audio_queue, stt_ws)
                    )

                    # Listen for responses from the STT service
                    while True:
                        message_str = await stt_ws.recv()
                        logger.info(f"Received message from STT: {message_str}")
                        yield message_str

            except (ConnectionRefusedError, ConnectionClosed):
                logging.warning(
                    f"Connection to STT service failed or was lost. Retrying in {self.retry_delay} seconds..."
                )
                await asyncio.sleep(self.retry_delay)
            except Exception as e:
                logging.error(
                    f"An unexpected error occurred in the transcript generator: {e}",
                    exc_info=True,
                )
                break  # Stop retrying for unexpected errors
            finally:
                if forwarder_task:
                    forwarder_task.cancel()