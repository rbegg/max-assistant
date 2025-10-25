# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
This script loads initial data into the Neo4j database.
"""
import asyncio
import logging
from neo4j import AsyncGraphDatabase
from src.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def load_data():
    """Connects to Neo4j and loads sample data."""
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    logging.info("Connecting to Neo4j to load data...")

    try:
        async with driver.session() as session:
            # Using MERGE ensures that we don't create duplicate data if the script is run multiple times.
            await session.run("""
                MERGE (u:User {name: 'Robert'})
                ON CREATE SET u.created = timestamp()
                MERGE (s:Schedule {summary: 'Breakfast: 8:30am, Lunch: 12:30pm, Dinner: 5:30pm'})
                ON CREATE SET s.created = timestamp()
                MERGE (u)-[:HAS_SCHEDULE]->(s)
            """)
            logging.info("Successfully loaded schedule for user 'Robert'.")

    except Exception as e:
        logging.error(f"An error occurred while loading data: {e}")
    finally:
        await driver.close()
        logging.info("Neo4j connection closed.")


if __name__ == "__main__":
    asyncio.run(load_data())
