# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
This module defines the ConnectionManager class, which manages the state, communication,
and task coordination for a single client WebSocket connection. It handles input/output
queues, the processing of client messages, and interactions with speech-to-text, text-to-speech,
and reasoning services.

The module integrates with external components like a reasoning engine, STT generator,
TTS synthesizer, and manages conversation states for active WebSocket connections.

Classes:
    - ConnectionManager: Manages a client WebSocket connection.

"""
import asyncio
import json
import logging
from asyncio import Queue

from fastapi import WebSocket, WebSocketDisconnect

from src.agent.agent import Agent
from src.config import QUEUE_GET_TIMEOUT
from src.api.stt_service import transcript_generator
from src.api.tts_service import synthesize_speech


class ConnectionManager:
    """Manages the state and logic for a single client WebSocket connection."""

    def __init__(self, reasoning_engine, websocket: WebSocket):
        self.ws = websocket
        self.agent = Agent(reasoning_engine)
        # Each connection gets its own set of queues
        self.binary_input_queue = Queue()
        self.text_input_queue = Queue()
        self.client_output_queue = Queue()
        self._shutdown_event = asyncio.Event()

    async def handle_connection(self):
        """Manages all tasks for a single client connection."""
        logging.info("Handling new client connection.")
        tasks = [
            asyncio.create_task(self._agent_loop()),
            asyncio.create_task(self._text_input_handler_loop()),
            asyncio.create_task(self._client_reader()),
            asyncio.create_task(self._client_writer())
        ]

        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

            for task in done:
                if task.exception():
                    logging.error(f"A connection task failed: {task.exception()}", exc_info=True)
        finally:
            self._shutdown_event.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            logging.info("Connection handler for a client finished.")

    async def _client_reader(self):
        """Reads messages from the WebSocket and puts them into the appropriate queues."""
        try:
            while not self._shutdown_event.is_set():
                message = await self.ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if 'text' in message:
                    await self.text_input_queue.put(message['text'])
                elif 'bytes' in message:
                    await self.binary_input_queue.put(message['bytes'])
        except WebSocketDisconnect:
            logging.info("Client disconnected (reader).")
        except Exception as e:
            logging.error(f"WS reader error: {e}", exc_info=True)
        finally:
            self._shutdown_event.set()

    async def _client_writer(self):
        """Gets messages from the output queue and sends them to the WebSocket."""
        try:
            while not self._shutdown_event.is_set():
                message = await self.client_output_queue.get()
                if isinstance(message, bytes):
                    await self.ws.send_bytes(message)
                else:
                    await self.ws.send_text(str(message))
        except WebSocketDisconnect:
            logging.info("Client disconnected (writer).")
        except asyncio.CancelledError:
            logging.info("Client writer cancelled.")
        except Exception as e:
            logging.error(f"WS writer error: {e}", exc_info=True)
        finally:
            self._shutdown_event.set()

    async def _text_input_handler_loop(self):
        """Processes text input from the client and updates the conversation state."""
        while not self._shutdown_event.is_set():
            try:
                text_data = await asyncio.wait_for(self.text_input_queue.get(), timeout=QUEUE_GET_TIMEOUT)
                logging.info(f"TEXT_HANDLER: Received text from client: {text_data}")
                client_dict = json.loads(text_data)
                if "username" in client_dict:
                    self.agent.set_username(client_dict["username"])
                if "voice" in client_dict:
                    self.agent.set_voice(client_dict["voice"])
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in text input handler: {e}", exc_info=True)
        logging.info("Text input handler loop has stopped.")

    async def _agent_loop(self):
        """Handles STT, reasoning, and TTS for the connection."""
        try:
            async for stt_message_str in transcript_generator(self.binary_input_queue):
                if self._shutdown_event.is_set():
                    break
                try:
                    stt_response = json.loads(stt_message_str)
                    transcript = stt_response.get("data", "").strip()
                    if not transcript:
                        continue

                    await self.client_output_queue.put(stt_message_str)

                    llm_response = await self.agent.ainvoke(transcript)

                    response_payload = {"data": llm_response, "source": "assistant"}
                    await self.client_output_queue.put(json.dumps(response_payload))

                    output_audio = await synthesize_speech(llm_response,
                                                           self.agent.get_voice())
                    logging.info("Sending audio Response.")
                    await self.client_output_queue.put(output_audio)

                except (json.JSONDecodeError, AttributeError) as e:
                    logging.warning(f"Could not parse STT message: {stt_message_str} ({e})")
        except asyncio.CancelledError:
            logging.info("Agent loop cancelled.")
        except Exception as e:
            logging.error(f"Error in agent loop: {e}", exc_info=True)
        logging.info("Agent loop has stopped.")