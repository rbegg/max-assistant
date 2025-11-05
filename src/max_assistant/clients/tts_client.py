import asyncio
import wave
from io import BytesIO
import logging

from wyoming.client import AsyncClient
from wyoming.tts import Synthesize, SynthesizeVoice
from wyoming.audio import AudioStart, AudioChunk, AudioStop
from wyoming.event import Event

logger = logging.getLogger(__name__)


class TTSClient:
    """Manages a persistent connection to a Wyoming TTS service."""

    def __init__(self, uri: str = "tcp://tts:10200", retry_delay: int = 5):
        self.uri = uri
        self.retry_delay = retry_delay
        self._client: AsyncClient | None = None
        self._lock = asyncio.Lock()

    async def connect(self):
        """Initiates connection to the TTS service."""
        async with self._lock:
            await self._ensure_connected()

    async def _ensure_connected(self):
        """Ensures there is an active connection to the TTS service. Must be called within a lock."""
        if self._client:
            return

        logger.info(f"Connecting to TTS service at {self.uri}")
        # This loop is for the initial connection.
        while True:
            try:
                client = AsyncClient.from_uri(self.uri)
                await client.connect()
                self._client = client
                logger.info("Connection to TTS established.")
                return
            except ConnectionRefusedError:
                logging.warning(f"Connection to {self.uri} refused. Retrying in {self.retry_delay} seconds...")
                await asyncio.sleep(self.retry_delay)
            except Exception as e:
                logging.error(f"An unexpected error occurred during TTS connection: {e}. Retrying in {self.retry_delay} seconds...")
                await asyncio.sleep(self.retry_delay)

    async def synthesize_speech(self, text: str, voice: str) -> bytes | None:
        """
        Sends text for synthesis and returns the received audio as bytes.
        Handles connection logic internally.
        """
        async with self._lock:
            try:
                await self._ensure_connected()
                if not self._client:
                    logging.error("TTS synthesis failed, client not connected.")
                    return None

                synthesize_event = Synthesize(
                    text=text,
                    voice=SynthesizeVoice(name=voice)
                )
                await self._client.write_event(synthesize_event.event())
                logger.info(f"Sent synthesize request with voice: {voice} text: '{text}'")

                wav_buffer = BytesIO()
                wav_writer = None
                audio_received = False

                while True:
                    event = await self._client.read_event()
                    if event is None:
                        logger.warning("TTS Connection closed by server. Will reconnect on next call.")
                        await self.close()
                        return None  # Current synthesis fails

                    if AudioStart.is_type(event.type):
                        start_event = AudioStart.from_event(event)
                        wav_writer = wave.open(wav_buffer, "wb")
                        wav_writer.setnchannels(start_event.channels)
                        wav_writer.setsampwidth(start_event.width)
                        wav_writer.setframerate(start_event.rate)
                        logger.info(
                            f"Audio stream started with params: rate={start_event.rate}, "
                            f"width={start_event.width}, channels={start_event.channels}"
                        )
                    elif AudioChunk.is_type(event.type):
                        chunk_event = AudioChunk.from_event(event)
                        if wav_writer:
                            wav_writer.writeframes(chunk_event.audio)
                            audio_received = True
                    elif AudioStop.is_type(event.type):
                        logger.info("Audio stream finished.")
                        break
                    elif Event.is_type(event.type, "error"):
                        logging.error(f"Received error from server: {event.data.get('text')}")
                        break

                if wav_writer:
                    wav_writer.close()

                return wav_buffer.getvalue() if audio_received else None

            except Exception as e:
                logging.error(f"An unexpected error occurred in TTSClient: {e}", exc_info=True)
                await self.close()
                return None

    async def close(self):
        """Closes the connection to the TTS service."""
        if self._client:
            try:
                await self._client.disconnect()
            except Exception as e:
                logging.warning(f"Error while disconnecting from TTS: {e}")
            finally:
                self._client = None
                logger.info("TTS connection closed.")
