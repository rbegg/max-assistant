import sys
import os
import logging
import asyncio

from dotenv import load_dotenv
if not load_dotenv('../.env.local'):
    print("Failed to load environment variables.")
    exit(1)

# Import both the tool and the client
from max_assistant.tools.gmail_tools import GmailTools
from max_assistant.clients.neo4j_client import Neo4jClient

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)




async def main():
    """
    Runs the one-time authentication process and sends a test email.
    """
    # 1. Check for command-line argument
    if len(sys.argv) < 2:
        logger.error("Usage: python gmail_authenticate.py <your-email@example.com>")
        logger.error("Please provide your email as an argument to send a test message.")
        sys.exit(1)

    test_recipient_email = sys.argv[1]

    # 2. Check for environment variables
    sender_email = os.environ.get("GOOGLE_SENDER_EMAIL")
    neo4j_uri = os.environ.get("NEO4J_URI")
    neo4j_user = os.environ.get("NEO4J_USER")
    neo4j_pass = os.environ.get("NEO4J_PASSWORD")

    if not all([sender_email, neo4j_uri, neo4j_user, neo4j_pass]):
        logger.error("FATAL: One or more environment variables are not set.")
        logger.error("Please ensure SENDER_EMAIL, NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD are set.")
        sys.exit(1)

    if sender_email != test_recipient_email:
        logger.warning(
            f"The SENDER_EMAIL ({sender_email}) is different from "
            f"the test recipient ({test_recipient_email})."
        )
        logger.warning("Make sure the SENDER_EMAIL is the account you authenticate with.")

    logger.info(f"Starting authentication for: {sender_email}")

    # 3. Create Neo4jClient instance
    client = None
    client = await Neo4jClient.create(uri=neo4j_uri, user=neo4j_user, password=neo4j_pass)

    try:
        # 4. Create tool instance (passing the client) and authenticate
        tools = GmailTools(client=client)
        await tools.authenticate()  # This will open the browser if credentials are not in DB

        logger.info("Authentication complete. Attempting to send test message...")

        # 5. Send a test message
        response = await tools.send_message(
            to=test_recipient_email,
            subject="Gmail API Authentication Test",
            message_text=(
                "Hello!\n\n"
                "This is a test message to confirm your Gmail API "
                "credentials were successfully saved to Neo4j."
            )
        )

        logger.info(f"Test message API response: {response}")

    except Exception as e:
        logger.error(f"An unexpected error occurred during the process: {e}")
        sys.exit(1)
    finally:
        # 6. Always close the client connection
        if client:
            await client.close()
        logger.info("Neo4j client connection closed.")


if __name__ == "__main__":
    # Use asyncio.run() to execute the async main function
    asyncio.run(main())