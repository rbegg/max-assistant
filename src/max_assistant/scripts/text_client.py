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

from max_assistant.config import (
    LOG_LEVEL,
    NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,
    OLLAMA_MODEL_NAME, OLLAMA_BASE_URL
)
from max_assistant.agent.agent import Agent
from max_assistant.agent.graph import create_reasoning_engine
from max_assistant.clients.ollama_preloader import warm_up_ollama_async
from max_assistant.clients.neo4j_client import Neo4jClient


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

    print("Initializing the reasoning engine...")
    global reasoning_engine, db_client
    print("Application startup...")

    try:
        # 1. Initialize the Neo4j client
        print("Initializing Neo4j client...")
        db_client = await Neo4jClient.create(
            NEO4J_URI,
            NEO4J_USERNAME,
            NEO4J_PASSWORD
        )

        # 2. Initialize and warm up the LLM
        print("Initializing and warming up the LLM...")
        llm = await warm_up_ollama_async(
            OLLAMA_MODEL_NAME,
            OLLAMA_BASE_URL,
            temperature=0
        )
        if not llm:
            raise RuntimeError("Failed to initialize the LLM.")

        # 3. Create the reasoning engine with the dependencies
        print("Initializing the reasoning engine...")
        reasoning_engine = await create_reasoning_engine(db_client, llm)

    except Exception as e:
        print(f"Failed to initialize application: {e}")
        # We should probably exit if startup fails
        # For a production app, you might handle this differently
        raise e

    agent = Agent(reasoning_engine)

    username = ""
    while not username:
        username = await asyncio.to_thread(input, "Please enter your username: ")
        if not username:
            print("Username cannot be empty. Please try again.")
    agent.set_username(username)

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
