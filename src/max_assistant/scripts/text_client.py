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


# Load environment variables for text client env
from dotenv import load_dotenv

if not load_dotenv('../.env.local'):
    print("Failed to load environment variables.")
    exit(1)

from max_assistant.config import LOG_LEVEL
from max_assistant.app_services import AppServices
from max_assistant.agent.agent import Agent


async def main(log_path=None):
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

    person_tools = None
    schedule_tools = None
    print("Application startup...")

    try:
        app_services = await AppServices.create()
        print("Application services successfully initialized.")

    except Exception as e:
        print(f"Failed to initialize application: {e}")
        raise e

    agent = Agent(app_services.reasoning_engine, app_services.person_tools)
    await agent.initialize_session()

    print("Agent is ready. Type 'exit' to quit.")

    while True:
        try:
            user_input = await asyncio.to_thread(input, "You: ")
            if user_input.lower() == 'exit':
                break

            response = await agent.ainvoke(user_input)
            print(f"Agent: {response}")

        except (KeyboardInterrupt, EOFError):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A command-line client for interacting with the text-based agent.")
    parser.add_argument("--log-path", type=str, help="Directory to store log files.")
    args = parser.parse_args()
    asyncio.run(main(log_path=args.log_path))
