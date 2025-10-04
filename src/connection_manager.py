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

from .config import DEFAULT_USERNAME, QUEUE_GET_TIMEOUT, TTS_VOICE
from .state import GraphState
from .stt_service import transcript_generator
from .tts_service import synthesize_speech


class ConnectionManager:
    """Manages the state and logic for a single client WebSocket connection."""

    def __init__(self, reasoning_engine, websocket: WebSocket):
        self.ws = websocket
        self.reasoning_engine = reasoning_engine
        # Each connection gets its own set of queues
        self.binary_input_queue = Queue()
        self.text_input_queue = Queue()
        self.client_output_queue = Queue()
        # Each connection gets its own state
        self.conversation_state: GraphState = {
            "messages": [],
            "username": DEFAULT_USERNAME,
            "transcribed_text": "",
            "voice": TTS_VOICE
        }
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
                    self.conversation_state["username"] = client_dict.get("username",
                                                                           self.conversation_state["username"])
                if "voice" in client_dict:
                    self.conversation_state["voice"] = client_dict.get("voice", self.conversation_state["voice"])
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

                    inputs: GraphState = {
                        "transcribed_text": transcript,
                        "messages": self.conversation_state.get("messages", []),
                        "username": self.conversation_state.get("username", DEFAULT_USERNAME),
                        "voice": self.conversation_state.get("voice", TTS_VOICE)
                    }
                    logging.info(f"Calling Reasoning engine with: {transcript}")
                    final_state = await self.reasoning_engine.ainvoke(inputs)
                    self.conversation_state = final_state

                    llm_response = ""
                    if final_state.get("messages") and len(final_state["messages"]) > 0:
                        last_message = final_state["messages"][-1]
                        llm_response = last_message.content

                    response_payload = {"data": llm_response, "source": "assistant"}
                    await self.client_output_queue.put(json.dumps(response_payload))

                    output_audio = await synthesize_speech(llm_response,
                                                           self.conversation_state.get("voice", TTS_VOICE))
                    await self.client_output_queue.put(output_audio)

                except (json.JSONDecodeError, AttributeError) as e:
                    logging.warning(f"Could not parse STT message: {stt_message_str} ({e})")
        except asyncio.CancelledError:
            logging.info("Agent loop cancelled.")
        except Exception as e:
            logging.error(f"Error in agent loop: {e}", exc_info=True)
        logging.info("Agent loop has stopped.")