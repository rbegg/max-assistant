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
from typing import List

from fastapi import WebSocket, WebSocketDisconnect

from max_assistant.agent.agent import Agent
from max_assistant.clients.stt_client import STTClient
from max_assistant.clients.tts_client import TTSClient
from .app_services import AppServices

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages the state and logic for a single client WebSocket connection."""

    def __init__(self, app_services: AppServices, websocket: WebSocket):
        self.ws = websocket
        self.agent = Agent(app_services.reasoning_engine, app_services.user_info)
        self.stt_client = STTClient()
        self.tts_client = TTSClient()
        self.app_services = app_services

        # Queues for decoupling producer/consumer tasks
        self.binary_input_queue = Queue()
        self.text_input_queue = Queue()
        self.client_output_queue = Queue()
        self.external_event_queue = Queue()

        self._shutdown_event = asyncio.Event()
        self._tasks: List[asyncio.Task] = []

    async def handle_connection(self):
        """
        Manages all tasks for a single client connection.
        This method acts as a supervisor, starting all necessary tasks and
        ensuring they are properly cleaned up upon exit.
        """
        logger.info("Handling new client connection.")

        self._tasks = [
            asyncio.create_task(self._client_reader()),
            asyncio.create_task(self._client_writer()),
            asyncio.create_task(self._run_main_logic()),
        ]

        try:
            # Wait for any of the main tasks to complete. This indicates the
            # connection should be closed, either normally or due to an error.
            done, pending = await asyncio.wait(self._tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                # Log any exceptions from the completed task
                if task.exception():
                    logger.error(f"A connection task failed: {task.exception()}", exc_info=True)
        finally:
            self._shutdown_event.set()
            await self._cancel_tasks(self._tasks)
            await self.tts_client.close()
            logger.info("Connection handler for a client finished.")

    async def submit_external_event(self, payload: dict):
        """
        Public API for external systems (like the reminder poller)
        to push events into the active connection's event loop.
        """
        await self.external_event_queue.put(payload)

    async def _run_main_logic(self):
        """
        Coordinates the primary logic, including LLM warmup, and runs the
        agent and text handler loops.
        """
        # Connect to dependent services.
        await asyncio.gather(
            self.tts_client.connect(),
            self._handle_llm_warmup()
        )

        # Once services are ready, start the core processing loops.
        processing_tasks = [
            asyncio.create_task(self._agent_loop()),
            asyncio.create_task(self._text_input_handler_loop()),
            asyncio.create_task(self._external_event_loop()),
        ]
        self._tasks.extend(processing_tasks)

        try:
            await asyncio.gather(*processing_tasks)
        except asyncio.CancelledError:
            logger.info("Main logic task cancelled.")
        finally:
            # Propagate cancellation to sub-tasks.
            await self._cancel_tasks(processing_tasks)
            logger.info("Main logic processing has stopped.")

    async def _handle_llm_warmup(self):
        """
        Checks if the LLM is ready and sends a waiting message to the client if not.
        """
        if not self.app_services.llm_ready_event.is_set():
            logger.info("LLM not ready. Sending a waiting message to the client.")
            response_text = "I am just getting set up, I'll be with you in a moment."

            response_payload = {"data": response_text, "source": "assistant"}
            await self.client_output_queue.put(json.dumps(response_payload))

            output_audio = await self.tts_client.synthesize_speech(
                response_text, self.agent.get_voice()
            )
            if output_audio:
                logger.info("Sending audio for waiting message.")
                await self.client_output_queue.put(output_audio)

            logger.info("Now waiting for LLM to be ready...")
            await self.app_services.llm_ready_event.wait()
            logger.info("LLM is now ready.")

    async def _client_reader(self):
        """Reads messages from the WebSocket and puts them into the appropriate queues."""
        try:
            while not self._shutdown_event.is_set():
                message = await self.ws.receive()
                if message.get("type") == "websocket.disconnect":
                    logger.info("Client initiated disconnect.")
                    break
                if 'text' in message:
                    await self.text_input_queue.put(message['text'])
                elif 'bytes' in message:
                    await self.binary_input_queue.put(message['bytes'])
        except WebSocketDisconnect:
            logger.info("Client disconnected (reader).")
        except asyncio.CancelledError:
            logger.info("Client reader task cancelled.")
        except Exception as e:
            # Catching unexpected errors for logging purposes.
            logging.error(f"Unexpected WS reader error: {e}", exc_info=True)
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
            logger.info("Client disconnected (writer).")
        except asyncio.CancelledError:
            logger.info("Client writer task cancelled.")
        except Exception as e:
            logging.error(f"Unexpected WS writer error: {e}", exc_info=True)
        finally:
            self._shutdown_event.set()

    async def _text_input_handler_loop(self):
        """Processes text input from the client and updates the conversation state."""
        try:
            while not self._shutdown_event.is_set():
                # This blocks naturally until an item is available, yielding control to the event loop
                text_data = await self.text_input_queue.get()

                logger.info(f"TEXT_HANDLER: Received text from client: {text_data}")

                try:
                    client_dict = json.loads(text_data)
                    if "username" in client_dict:
                        logger.info(f"username sent: {client_dict['username']}")
                    if "voice" in client_dict:
                        self.agent.set_voice(client_dict["voice"])
                except (json.JSONDecodeError, TypeError) as e:
                    logging.warning(f"Could not parse text message from client: {e}")
                finally:
                    # Explicitly mark the task as done to maintain queue integrity
                    self.text_input_queue.task_done()

        except asyncio.CancelledError:
            logger.info("Text input handler loop task cancelled.")
        except Exception as e:
            logging.error(f"Error in text input handler: {e}", exc_info=True)
            self._shutdown_event.set()
        finally:
            logger.info("Text input handler loop has stopped.")

    async def _agent_loop(self):
        """Handles STT, reasoning, and TTS for the connection."""
        try:
            async for stt_message_str in self.stt_client.transcript_generator(
                self.binary_input_queue, self._shutdown_event
            ):
                try:
                    stt_response = json.loads(stt_message_str)
                    transcript = stt_response.get("data", "").strip()
                    if not transcript:
                        continue

                    await self.client_output_queue.put(stt_message_str)

                    llm_response = await self.agent.ainvoke(transcript)

                    response_payload = {"data": llm_response, "source": "assistant"}
                    await self.client_output_queue.put(json.dumps(response_payload))

                    output_audio = await self.tts_client.synthesize_speech(
                        llm_response, self.agent.get_voice()
                    )
                    if output_audio:
                        logger.info("Sending audio Response.")
                        await self.client_output_queue.put(output_audio)

                except (json.JSONDecodeError, AttributeError) as e:
                    logging.warning(f"Could not parse STT message: {stt_message_str} ({e})")
        except asyncio.CancelledError:
            pass  # Task was canceled.
        except Exception as e:
            logging.error(f"Error in agent loop: {e}", exc_info=True)
            self._shutdown_event.set()
        finally:
            logger.info("Agent loop has stopped.")


    async def _cancel_tasks(self, tasks: List[asyncio.Task]):
        for task in tasks:
            if not task.done():
                task.cancel()

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception) and not isinstance(res, asyncio.CancelledError):
                logger.error(f"Background task raised an unhandled exception during shutdown: {res}")


    async def _external_event_loop(self):
        """Listens for system-generated events, processes them via the Agent, and sends to client."""
        try:
            while not self._shutdown_event.is_set():
                # Wait for an external event payload from the poller
                event_payload = await self.external_event_queue.get()
                logger.info(f"EXTERNAL_EVENT_HANDLER: Processing event: {event_payload}")

                # 1. Process the event through the LangGraph reasoning engine
                llm_response = await self.agent.handle_push_event(event_payload)

                if not llm_response:
                    continue

                # 2. Dispatch text payload to the active WebSocket
                response_payload = {"data": llm_response, "source": "assistant", "type": "notification"}
                await self.client_output_queue.put(json.dumps(response_payload))

                # 3. Synthesize audio stream via the session's active TTS worker
                output_audio = await self.tts_client.synthesize_speech(
                    llm_response, self.agent.get_voice()
                )

                if output_audio:
                    logger.info("Sending synthesized audio for background event.")
                    await self.client_output_queue.put(output_audio)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logging.error(f"Error in external event loop: {e}", exc_info=True)
            self._shutdown_event.set()
        finally:
            logger.info("External event loop has stopped.")