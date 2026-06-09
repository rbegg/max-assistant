import asyncio
import logging
from websockets.asyncio.client import connect as websocket_connect, ClientConnection
from websockets.exceptions import ConnectionClosed
from max_assistant.config import STT_WEBSOCKET_URL

logger = logging.getLogger(__name__)


class STTClient:
    """
    Manages a long-lived, persistent connection to the STT service
    reused cleanly across different chat flows with the user.
    """

    def __init__(self, uri: str = STT_WEBSOCKET_URL, retry_delay: int = 5):
        self.uri = uri
        self.retry_delay = retry_delay
        self._websocket: ClientConnection | None = None
        self._lock = asyncio.Lock()  # Guards connection creation concurrency

    async def _ensure_connected(self) -> ClientConnection:
        """
        Guarantees that a single warm connection is maintained
        across multiple user turns and different chat flows.
        """
        async with self._lock:
            if self._websocket is None or getattr(self._websocket, "closed", True):
                logger.info(f"Establishing persistent connection to STT service at {self.uri}...")
                try:
                    self._websocket = await websocket_connect(self.uri)
                    logger.info("STT persistent connection opened and hot.")
                except Exception as e:
                    logger.error(f"Failed to open connection to STT server: {e}")
                    raise
            return self._websocket

    async def _forward_audio_loop(self, audio_queue: asyncio.Queue, stt_ws: ClientConnection):
        """
        Forwards audio chunks from the queue immediately to the open socket.
        Uses task cancellation rather than high-overhead polling timeouts.
        """
        try:
            while True:
                audio_chunk = await audio_queue.get()
                await stt_ws.send(audio_chunk)
                audio_queue.task_done()
        except ConnectionClosed:
            logger.info("STT connection closed during audio forwarding.")
        except asyncio.CancelledError:
            pass  # Normal termination path
        except Exception as e:
            logger.error(f"Error forwarding audio stream: {e}")

    async def transcript_generator(self, audio_queue: asyncio.Queue, shutdown_event: asyncio.Event):
        """
        Yields live transcripts over a reused connection pipeline.
        Gracefully handles server dropouts and recovers instantly.
        """
        forwarder_task = None

        while not shutdown_event.is_set():
            try:
                # 1. Acquire or reuse the active socket across turns
                stt_ws = await self._ensure_connected()

                # 2. Spawn audio forwarder without inner poll loops
                forwarder_task = asyncio.create_task(self._forward_audio_loop(audio_queue, stt_ws))

                # 3. Create a listener monitor task to allow zero-overhead shutdown interception
                while not shutdown_event.is_set():
                    # We create a background read task so we can wait for either data or shutdown
                    recv_task = asyncio.create_task(stt_ws.recv())
                    shutdown_task = asyncio.create_task(shutdown_event.wait())

                    done, pending = await asyncio.wait(
                        {recv_task, shutdown_task},
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    # Clean up pending helper tasks immediately to avoid leakage
                    for task in pending:
                        task.cancel()

                    if shutdown_task in done:
                        break

                    if recv_task in done:
                        try:
                            message_str = recv_task.result()
                            logger.info(f"Received message from STT: {message_str}")
                            yield message_str
                        except Exception as recv_error:
                            # Re-raise to trigger reconnection logic if the stream broke
                            raise recv_error

            except (ConnectionRefusedError, ConnectionClosed, OSError):
                await self.close_connection()
                if shutdown_event.is_set():
                    break
                logger.warning(
                    f"STT pipeline disconnected. Retrying connection in {self.retry_delay}s..."
                )
                await asyncio.sleep(self.retry_delay)
            except Exception as e:
                logger.error(f"Unexpected crash in STT transcript generator loop: {e}", exc_info=True)
                break
            finally:
                if forwarder_task and not forwarder_task.done():
                    forwarder_task.cancel()
                    await asyncio.gather(forwarder_task, return_exceptions=True)

    async def close_connection(self):
        """Clean resource breakdown for application lifecycle turns."""
        async with self._lock:
            if self._websocket:
                try:
                    await self._websocket.close()
                except Exception:
                    pass
                self._websocket = None
                logger.info("STT connection pool cleared down.")