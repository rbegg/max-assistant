# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
A command-line client for interacting with the text-based agent.
"""
import asyncio
import logging

from .agent import Agent
from .graph import create_reasoning_engine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def main():
    """
    A simple text-based client to interact with the Agent.
    """
    print("Initializing the reasoning engine...")
    reasoning_engine = await create_reasoning_engine()
    if not reasoning_engine:
        print("Failed to initialize reasoning engine. Exiting.")
        return

    agent = Agent(reasoning_engine)
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
    asyncio.run(main())
