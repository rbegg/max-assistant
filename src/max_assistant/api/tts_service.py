import wave
from io import BytesIO
import logging

from wyoming.client import AsyncClient
from wyoming.tts import Synthesize, SynthesizeVoice
from wyoming.audio import AudioStart, AudioChunk, AudioStop
from wyoming.event import Event

logger = logging.getLogger(__name__)

async def synthesize_speech(text: str, voice: str) -> bytes:
    """
    Connects to a Wyoming TTS service using the wyoming library,
    sends text for synthesis, and saves the received audio to a WAV file.
    """
    uri = "tcp://tts:10200"

    try:
        async with AsyncClient.from_uri(uri) as client:
            logger.info("Connection to tts established.")

            # 1. Send synthesize request
            synthesize_event = Synthesize(
                text=text,
                voice=SynthesizeVoice(name=voice)
            )
            await client.write_event(synthesize_event.event())
            logger.info(f"Sent synthesize request with voice: {voice} text: '{text}'")

            # 2. Prepare in-memory buffer for WAV data
            wav_buffer = BytesIO()
            wav_writer = None
            audio_received = False

            while True:
                event = await client.read_event()
                if event is None:
                    logger.info("TTS Connection closed by server.")
                    break

                if AudioStart.is_type(event.type):
                    start_event = AudioStart.from_event(event)
                    # Open a WAV writer that writes to our in-memory buffer
                    wav_writer = wave.open(wav_buffer, "wb")
                    wav_writer.setnchannels(start_event.channels)
                    wav_writer.setsampwidth(start_event.width)
                    wav_writer.setframerate(start_event.rate)
                    logger.info(
                        f"Audio stream started with params: rate={start_event.rate}, "
                        "width={start_event.width}, channels={start_event.channels}"
                    )

                elif AudioChunk.is_type(event.type):
                    chunk_event = AudioChunk.from_event(event)
                    if wav_writer is not None:
                        wav_writer.writeframes(chunk_event.audio)
                        audio_received = True

                elif AudioStop.is_type(event.type):
                    logger.info("Audio stream finished.")
                    break

                elif Event.is_type(event.type, "error"):
                    logging.error(f"Received error from server: {event.data.get('text')}")
                    break

            if wav_writer is not None:
                wav_writer.close()  # Finalizes the WAV header in the buffer

            if audio_received:
                # Return the complete WAV data from the buffer
                return wav_buffer.getvalue()
            else:
                logger.info("No audio data was received.")
                return None
    except ConnectionRefusedError:
        logging.error(f"Connection to {uri} refused. Is the service running?")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
