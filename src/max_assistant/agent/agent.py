# max_assistant/agent/agent.py

import logging
from uuid import uuid4
from typing import Dict, Any, Optional
from max_assistant.config import DEFAULT_USERNAME, TTS_VOICE

logger = logging.getLogger(__name__)


class Agent:
    """Encapsulates the reasoning engine and handles stateless text turn delivery."""

    def __init__(self, reasoning_engine: Any, initial_user_info: Optional[Dict[str, Any]] = None) -> None:
        self.reasoning_engine = reasoning_engine
        runtime_user_info = initial_user_info if initial_user_info is not None else {}

        self.conversation_state: Dict[str, Any] = {
            "userinfo": runtime_user_info,
            "transcribed_text": "",
            "voice": TTS_VOICE,
            "external_event": {},
            "is_background": False,
        }

        # Long-running context tracker block. Standard checkpointers resolve full history
        # using only the thread_id property via aget_tuple loop calls!
        self.config: Dict[str, Any] = {
            "configurable": {
                "thread_id": str(uuid4()),
                "checkpoint_ns": ""
            }
        }
        self.connection_manager = None
        user_name = runtime_user_info.get("user", {}).get("firstName", DEFAULT_USERNAME)
        logger.info(
            f"Agent initialized for user: {user_name} with Thread ID: {self.config['configurable']['thread_id']}")

    def set_thread_id(self, thread_id: str) -> None:
        self.config["configurable"]["thread_id"] = thread_id

    def get_thread_id(self) -> str:
        return self.config["configurable"]["thread_id"]

    def set_voice(self, voice: str) -> None:
        self.conversation_state["voice"] = voice

    def get_voice(self) -> str:
        return self.conversation_state.get("voice", TTS_VOICE)

    def set_user_info(self, user_info: Dict[str, Any]) -> None:
        self.conversation_state["userinfo"] = user_info

    async def ainvoke(self, text_input: str) -> str:
        inputs = {
            "transcribed_text": text_input,
            "userinfo": self.conversation_state["userinfo"],
            "voice": self.conversation_state["voice"],
            "external_event": {},
            "is_background": False
        }
        return await self._execute_graph_turn(inputs, self.config)

    async def handle_push_event(self, reminder_payload: dict) -> str:
        inputs = {
            "transcribed_text": "",
            "userinfo": self.conversation_state["userinfo"],
            "voice": self.conversation_state["voice"],
            "external_event": reminder_payload,
            "is_background": True,
        }
        return await self._execute_graph_turn(inputs, self.config)

    async def _execute_graph_turn(self, inputs: dict, config: dict[str, Any]) -> str:
        """Internal helper to execute LangGraph and extract final text responses."""
        try:
            # LangGraph handles parent relationship drawings natively across turns now
            final_state = await self.reasoning_engine.ainvoke(inputs, config=config)

            if isinstance(final_state, dict):
                # Update visual properties back to application session stores
                self.conversation_state.update({
                    k: v for k, v in final_state.items() if k != "messages"
                })

                # Grab the latest unified text message array update response
                new_messages = final_state.get("messages", [])
                if new_messages:
                    last_message = new_messages[-1]
                    if isinstance(last_message.content, str):
                        return last_message.content
                return ""
            else:
                logger.error(f"Engine returned invalid state type: {type(final_state)}")
                return "" if inputs.get("is_background") else "I encountered an internal logic error."

        except Exception as e:
            logger.error(f"Reasoning engine crashed during execution: {e}", exc_info=True)
            return "" if inputs.get("is_background") else "I'm sorry, I encountered an error."