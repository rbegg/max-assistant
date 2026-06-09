# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
Manages connection boundaries and task coordination for WebSocket lifecycles.
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
from max_assistant.agent.session_mgr import session_manager
from max_assistant.app_services import AppServices
from max_assistant.tools import PersonTools
from max_assistant.config import TTS_VOICE

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages the state and logic for a single client WebSocket connection."""

    def __init__(self, app_services: AppServices, websocket: WebSocket):
        self.ws = websocket
        self.agent = None
        self.stt_client = STTClient()
        self.tts_client = TTSClient()
        self.app_services = app_services

        # Decoupled task interaction channels
        self._binary_input_queue = Queue()
        self._text_input_queue = Queue()
        self._client_output_queue = Queue()
        self._external_event_queue = Queue()

        self._shutdown_event = asyncio.Event()
        self._authenticated = asyncio.Event()
        self._tasks: List[asyncio.Task] = []
        self._registered_user_id = None  # Track state for deterministic cleanup

    async def handle_connection(self):
        """Acts as connection supervisor, coordinating loops and deterministic teardown."""
        logger.info("Handling new client connection.")

        self._tasks = [
            asyncio.create_task(self._client_reader()),
            asyncio.create_task(self._client_writer()),
            asyncio.create_task(self._run_main_logic()),
        ]

        try:
            done, pending = await asyncio.wait(self._tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task.exception():
                    logger.error(f"A connection task failed: {task.exception()}", exc_info=True)
        finally:
            self._shutdown_event.set()

            # Empty remaining elements to prevent queue join deadlocks
            while not self._text_input_queue.empty():
                try:
                    self._text_input_queue.get_nowait()
                    self._text_input_queue.task_done()
                except asyncio.QueueEmpty:
                    break

            # Deterministic centralized unregistration safeguard
            if self._registered_user_id:
                logger.info(f"Centralized Cleanup: Unregistering session for user {self._registered_user_id}")
                session_manager.unregister(self._registered_user_id, self.agent)

            await self._cancel_tasks(self._tasks)
            await self.tts_client.close()
            logger.info("Connection handler for a client finished.")

    async def _enforce_auth_timeout(self):
        """Closes the connection if the user fails to authenticate within 15 seconds."""
        try:
            await asyncio.wait_for(self._authenticated.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            if self.agent is None:
                logger.warning("Connection timed out waiting for authentication credentials. Closing.")
                error_payload = {"data": "Authentication timeout.", "source": "system"}
                await self.ws.send_text(json.dumps(error_payload))
                await self.ws.close(code=1008)
                self._shutdown_event.set()

    async def submit_external_event(self, payload: dict):
        """Public API for pushing external polling notices into this connection session."""
        await self._external_event_queue.put(payload)

    async def _run_main_logic(self):
        """Warms up core assets and launches processing loops concurrent executors."""
        await asyncio.gather(
            self.tts_client.connect(),
            self._handle_llm_warmup()
        )

        processing_tasks = [
            asyncio.create_task(self._agent_loop()),
            asyncio.create_task(self._text_input_handler_loop()),
            asyncio.create_task(self._external_event_loop()),
            asyncio.create_task(self._enforce_auth_timeout()),
        ]
        self._tasks.extend(processing_tasks)

        try:
            await asyncio.gather(*processing_tasks)
        except asyncio.CancelledError:
            logger.info("Main logic task cancelled.")
        finally:
            await self._cancel_tasks(processing_tasks)
            logger.info("Main logic processing has stopped.")

    async def _handle_llm_warmup(self):
        """Pushes waiting assets down the pipe if core engines are booting."""
        if not self.app_services.llm_ready_event.is_set():
            logger.info("LLM not ready. Sending a waiting message to the client.")
            response_text = "I am just getting set up, Please Try again in a moment."

            response_payload = {"data": response_text, "source": "assistant"}
            await self._client_output_queue.put(json.dumps(response_payload))

            output_audio = await self.tts_client.synthesize_speech(response_text, TTS_VOICE)
            if output_audio:
                await self._client_output_queue.put(output_audio)

            await self.app_services.llm_ready_event.wait()
            logger.info("LLM is now ready.")

    async def _client_reader(self):
        """Pushes socket data directly onto processing queues."""
        try:
            while not self._shutdown_event.is_set():
                message = await self.ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if 'text' in message:
                    await self._text_input_queue.put(message['text'])
                elif 'bytes' in message:
                    await self._binary_input_queue.put(message['bytes'])
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logging.error(f"Unexpected WS reader error: {e}", exc_info=True)
        finally:
            self._shutdown_event.set()

    async def _client_writer(self):
        """Pops items from the outbound container queue and delivers across WebSocket channel."""
        try:
            while not self._shutdown_event.is_set():
                message = await self._client_output_queue.get()
                if isinstance(message, bytes):
                    await self.ws.send_bytes(message)
                else:
                    await self.ws.send_text(str(message))
                self._client_output_queue.task_done()
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logging.error(f"Unexpected WS writer error: {e}", exc_info=True)
        finally:
            self._shutdown_event.set()

    async def _text_input_handler_loop(self):
        """Processes control frames, typed responses, and user credentials."""
        while not self._shutdown_event.is_set():
            text_data = await self._text_input_queue.get()
            logger.info(f"TEXT_HANDLER: Received raw data string: {text_data}")

            try:
                client_dict = json.loads(text_data)

                # --- GATEKEEPER (AUTHENTICATION) ---
                if self.agent is None:
                    if "username" not in client_dict:
                        logger.warning("Unauthenticated message received. Awaiting username.")
                        continue

                    target_username = client_dict['username']
                    person_tools = PersonTools(self.app_services.db_client)
                    user_data = await person_tools.get_user_info_internal(target_username)

                    if "error" in user_data:
                        logger.error(f"Auth failure for {target_username}: {user_data['error']}")
                        error_payload = {"data": "Authentication failed. User not found.", "source": "system"}
                        await self.ws.send_text(json.dumps(error_payload))
                        await self.ws.close(code=1008, reason="User not found")
                        self._shutdown_event.set()
                        break

                    # Core Session Initialization
                    self.agent = Agent(self.app_services.reasoning_engine, user_data)
                    self._authenticated.set()
                    self.agent.connection_manager = self
                    self.agent.set_user_info(user_data)

                    user_id = user_data.get("user", {}).get("id")
                    if user_id:
                        self._registered_user_id = user_id
                        session_manager.register(user_id, self.agent)
                    continue

                # --- NORMAL METADATA & TEXT ROUTING PROCESSING ---
                if "voice" in client_dict:
                    self.agent.set_voice(client_dict["voice"])

                if "text" in client_dict:
                    # FIX: Route text directly into an explicit execution runner task
                    # instead of recursively pushing back onto the raw transport queue
                    asyncio.create_task(self._execute_agent_text_turn(client_dict["text"]))

            except json.JSONDecodeError:
                logging.warning(f"Discarding malformed JSON text framing payload: {text_data}")
            except Exception as e:
                logging.error(f"Exception encountered inside text loop frame router: {e}", exc_info=True)
                self._shutdown_event.set()
            finally:
                # Guaranteed completion tracking invocation
                self._text_input_queue.task_done()

    async def _send_assistant_response(self, text: str):
        """
        Unified pipeline to frame an assistant response string, dispatch it
        to the client output socket, and synthesize/queue matching TTS audio.
        """
        if not text:
            return

        # 1. Dispatch the text payload
        response_payload = {"data": text, "source": "assistant"}
        await self._client_output_queue.put(json.dumps(response_payload))

        # 2. Synthesize and dispatch the audio stream
        output_audio = await self.tts_client.synthesize_speech(text, self.agent.get_voice())
        if output_audio:
            logger.info("Sending synthesized audio Response.")
            await self._client_output_queue.put(output_audio)

    async def _execute_agent_text_turn(self, user_text: str):
        """Isolated wrapper task to run text turns through the reasoning engine without blocking loop I/O."""
        try:
            cleaned_text = user_text.strip()
            if not cleaned_text:
                return

            # Mirror typed message directly to client output logs for interface symmetry
            echo_payload = {"data": cleaned_text, "source": "user", "type": "text_echo"}
            await self._client_output_queue.put(json.dumps(echo_payload))

            # Run through reasoning graph
            llm_response = await self.agent.ainvoke(cleaned_text)
            await self._send_assistant_response(llm_response)

        except Exception as e:
            logger.error(f"Error processing text-to-agent inference pass: {e}", exc_info=True)

    async def _agent_loop(self):
        """Processes continuous audio chunks received from streaming speech input."""
        try:
            async_gen = self.stt_client.transcript_generator(self._binary_input_queue, self._shutdown_event)
            async for stt_message_str in async_gen:
                try:
                    stt_response = json.loads(stt_message_str)
                    transcript = stt_response.get("data", "").strip()
                    if not transcript:
                        continue

                    await self._client_output_queue.put(stt_message_str)

                    llm_response = await self.agent.ainvoke(transcript)
                    await self._send_assistant_response(llm_response)

                except (json.JSONDecodeError, AttributeError) as e:
                    logging.warning(f"Could not parse STT stream framework: {e}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logging.error(f"Error executing streaming audio client loop: {e}", exc_info=True)
            self._shutdown_event.set()

    async def _external_event_loop(self):
        """Handles background notifications dispatched via automated polling engines."""
        try:
            while not self._shutdown_event.is_set():
                event_payload = await self._external_event_queue.get()
                logger.info(f"EXTERNAL_EVENT_HANDLER: Processing target push: {event_payload}")

                llm_response = await self.agent.handle_push_event(event_payload)
                if llm_response:
                    # Custom overrides can still be handled before or inside the helper if needed.
                    # Here we leverage the unified sender channel:
                    await self._send_assistant_response(llm_response)

                self._external_event_queue.task_done()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logging.error(f"Error inside proactive push system supervisor loop: {e}", exc_info=True)
            self._shutdown_event.set()

    @staticmethod
    async def _cancel_tasks(tasks: List[asyncio.Task]):
        """Safely collapses the array of async execution routines."""
        for task in tasks:
            if not task.done():
                task.cancel()

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception) and not isinstance(res, asyncio.CancelledError):
                logger.error(f"Background connection task threw error on cancellation unwind: {res}")