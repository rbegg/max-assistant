# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
A command-line client for interacting with the text-based agent.
"""
import asyncio
import logging
import datetime
import argparse
import os
import sys
from pathlib import Path

# Load environment variables robustly using file-relative paths
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).parent.resolve()
local_env_file = SCRIPT_DIR / '../.env.local'
if not load_dotenv(local_env_file):
    print(f"Failed to load environment variables from {local_env_file}.")
    exit(1)

from max_assistant.config import LOG_LEVEL, DEFAULT_USERNAME
from max_assistant.app_services import AppServices
from max_assistant.agent.agent import Agent
from max_assistant.tools import PersonTools


class AsyncConsoleReader:
    """Manages a single, persistent async stream for reading from standard input."""

    def __init__(self):
        self.reader = None
        self.transport = None

    async def initialize(self):
        """Sets up the read pipe exactly once."""
        loop = asyncio.get_running_loop()
        self.reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(self.reader)

        # Connect to stdin ONCE. This keeps the file descriptor open and fast.
        self.transport, _ = await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    async def readline(self, prompt: str) -> str:
        """Reads a single line of input cleanly without blocking the event loop."""
        print(prompt, end="", flush=True)
        if not self.reader:
            await self.initialize()

        line = await self.reader.readline()
        if not line:  # Handles EOF (Ctrl+D) Safely
            raise EOFError()
        return line.decode().rstrip('\r\n')

    def close(self):
        """Detaches the transport cleanly without violently closing sys.stdin."""
        if self.transport:
            self.transport.close()


async def main(log_path=None, username=None):
    """
    A simple text-based client to interact with the Agent.
    """

    print("log path = ", log_path)
    if log_path:
        os.makedirs(log_path, exist_ok=True)
        log_filename = os.path.join(log_path, f"agent_client_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        print("log filename = ", log_filename)

        # Reconfigure logging to use the specified file
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=LOG_LEVEL,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            filename=log_filename,
                            filemode='w')

    print("Application startup...")

    try:
        app_services = await AppServices.create()
        print("Application services successfully initialized.")

    except Exception as e:
        print(f"Failed to initialize application: {e}")
        raise e

    # Sync with LLM background preloader [cite: 4, 8]
    if not app_services.llm_ready_event.is_set():
        print("Waiting for LLM model preloading to complete...")
        await app_services.llm_ready_event.wait()
        print("LLM core engine is warm and ready.")

    # Fetch User Profile Data from the database dynamically [cite: 2, 7]
    target_username = username or DEFAULT_USERNAME
    print(f"Loading user profile for username: '{target_username}'...")

    user_data = {}
    try:
        person_tools = PersonTools(app_services.db_client)
        user_data = await person_tools.get_user_info_internal(target_username)
        if "error" in user_data:
            print(f"Warning: {user_data['error']}. Falling back to default empty profile.")
            user_data = {}
    except Exception as e:
        print(f"Warning: Failed to fetch user info from Neo4j: {e}. Falling back to default empty profile.")

    # Initialize Agent with correct context signature
    agent = Agent(app_services.reasoning_engine, user_data)

    # Instantiate our unified async console reader
    console_reader = AsyncConsoleReader()
    await console_reader.initialize()

    print("\nAgent is ready. Type 'exit' to quit.")

    try:
        while True:
            user_input = await console_reader.readline("You: ")

            if user_input.lower() == 'exit':
                break
            if not user_input.strip():
                continue

            response = await agent.ainvoke(user_input)
            print(f"Agent: {response}")

    except (KeyboardInterrupt, EOFError, asyncio.CancelledError):
        print("\nSession interrupted by user.")
    finally:
        # Clean up console transport layers
        console_reader.close()

        # Defensive database client connection termination [cite: 5, 9]
        if 'app_services' in locals() and app_services.db_client:
            print("Closing active Neo4j client connection pooling...")
            try:
                await app_services.db_client.close()
            except Exception as e:
                print(f"Error closing Neo4j connectivity pool safely: {e}")
        print("Shutdown complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A command-line client for interacting with the text-based agent.")
    parser.add_argument("--log-path", type=str, help="Directory to store log files.")
    parser.add_argument("--username", type=str, help="Username to authenticate and load profile data.")
    args = parser.parse_args()

    try:
        asyncio.run(main(log_path=args.log_path, username=args.username))
    except KeyboardInterrupt:
        pass