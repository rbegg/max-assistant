# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
This module provides tools to interact with the Gmail API for sending emails
and managing authentication tokens. It integrates with a Neo4j database to
store and retrieve credentials securely, ensuring smooth API interactions.

Classes:
- GmailTools: Encapsulates Gmail API operations, providing methods for
  user authentication, token management, and email sending.
"""
import os.path
import base64
import logging
import os
import json
import asyncio
from email.mime.text import MIMEText
from datetime import datetime

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from langchain_core.tools import StructuredTool
from max_assistant.models.google_models import SendGmailArgs
from max_assistant.clients.neo4j_client import Neo4jClient
from max_assistant.config import (
    GOOGLE_SENDER_EMAIL, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET
)


logger = logging.getLogger(__name__)

# --- Configuration ---
SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
# This is a constant for Google's token endpoint
TOKEN_URI = "https://oauth2.googleapis.com/token"


# --- Main Class ---

class GmailTools:
    """
    An async class that encapsulates Gmail API operations.
    It requires a Neo4jClient to store and retrieve user credentials.
    """

    def __init__(self, client: Neo4jClient):
        """
        Initializes the toolset with a Neo4j client.
        """
        self.client = client
        self.sender_email = GOOGLE_SENDER_EMAIL
        self.client_id = GOOGLE_CLIENT_ID
        self.client_secret = GOOGLE_CLIENT_SECRET

        if not self.client_id or not self.client_secret:
            logger.error("FATAL: 'GOOGLE_CLIENT_ID' or 'GOOGLE_CLIENT_SECRET' "
                         "environment variables not set. GmailTools will not function.")

    async def authenticate(self):
        """
        Runs the *initial* one-time authentication flow.
        Saves the refresh_token, access_token, and expiry to the :User node.
        """

        check_query = "MATCH (u:User) RETURN u.gmailRefreshToken AS token"
        result = await self.client.execute_query(check_query, {})
        if "data" in result and result["data"] and result["data"][0].get("token"):
            logger.warning(
                "Gmail refresh token already exists for the User in Neo4j. "
                "Skipping authentication. To re-authenticate, clear the "
                "gmail... properties on the :User node."
            )
            return

        if not self.client_id or not self.client_secret:
            logger.error("Cannot authenticate. App credentials not set in env.")
            return

        client_config = {
            "installed": {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": TOKEN_URI,
                "redirect_uris": ["http://localhost"]
            }
        }

        logger.info("Starting one-time authentication flow...")
        flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
        creds = await asyncio.to_thread(flow.run_local_server, port=0)

        if not creds.refresh_token:
            logger.error("Authentication failed: No refresh token was returned.")
            return

        # --- CHANGED: Save all user tokens and expiry to Neo4j ---
        set_query = """
        MATCH (u:User) 
        SET u.gmailRefreshToken = $refresh_token,
            u.gmailAccessToken = $access_token,
            u.gmailTokenExpiry = $expiry
        """
        params = {
            "refresh_token": creds.refresh_token,
            "access_token": creds.token,
            "expiry": creds.expiry.isoformat()  # Store expiry as an ISO string
        }

        await self.client.execute_query(set_query, params)
        logger.info("Authentication successful. All user tokens saved to :User node.")

    async def _get_credentials(self) -> Credentials | None:
        """
        Private helper to get valid credentials.
        It loads all user tokens from Neo4j, reconstructs the Credentials object,
        and refreshes/re-saves the access token *only if* it's expired.
        """
        if not self.client_id or not self.client_secret:
            logger.error("Failed to get credentials. App secrets not set in env.")
            return None

        # 1. Fetch all user tokens and cache from Neo4j
        get_query = """
        MATCH (u:User) 
        RETURN u.gmailRefreshToken AS refresh_token,
               u.gmailAccessToken AS access_token,
               u.gmailTokenExpiry AS expiry
        """
        result = await self.client.execute_query(get_query, {})

        if "error" in result:
            logger.error(f"Neo4j error: {result['error']}")
            return None

        data = result.get("data", [{}])[0]
        refresh_token = data.get("refresh_token")
        access_token = data.get("access_token")
        expiry_str = data.get("expiry")

        if not refresh_token:
            logger.error("No Gmail refresh token found on :User node. "
                         "Please run the 'gmail_authenticate.py' script first.")
            return None

        # 2. Reconstruct the Credentials object from its components
        try:
            # Parse the expiry string back into a datetime object
            expiry_dt = datetime.fromisoformat(expiry_str) if expiry_str else None

            creds = Credentials(
                token=access_token,
                refresh_token=refresh_token,
                token_uri=TOKEN_URI,
                client_id=self.client_id,
                client_secret=self.client_secret,
                scopes=SCOPES,
                expiry=expiry_dt
            )

            # 3. Check expiry and refresh *only if needed*
            if creds and creds.expired and creds.refresh_token:
                logger.info("Access token is expired. Refreshing...")
                await asyncio.to_thread(creds.refresh, Request())

                # 4. Save the new, refreshed token and expiry back to Neo4j
                set_query = """
                MATCH (u:User) 
                SET u.gmailAccessToken = $access_token,
                    u.gmailTokenExpiry = $expiry
                """
                params = {
                    "access_token": creds.token,
                    "expiry": creds.expiry.isoformat()
                }
                await self.client.execute_query(set_query, params)
                logger.info("Access token refreshed and saved back to Neo4j.")

            elif creds.valid:
                logger.info("Using cached, valid access token.")

        except Exception as e:
            logger.error(f"Failed to refresh access token: {e}")
            logger.error("The user's refresh token may be expired or revoked.")
            return None

        if not creds or not creds.valid:
            logger.error("Failed to load or refresh credentials.")
            return None

        return creds

    def _create_message(self, to: str, subject: str, message_text: str) -> dict:
        """
        Creates a MIMEText message object and encodes it for the Gmail API.
        This is a synchronous helper method as it involves no I/O.
        """
        message = MIMEText(message_text)
        message["to"] = to
        message["from"] = self.sender_email
        message["subject"] = subject

        # Encode the message in base64url format
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        return {"raw": raw_message}

    async def send_message(self, to: str, subject: str, message_text: str) -> str:
        """
        Sends an email message to a recipient's email address on the user's behalf.
        The to parameter must be a valid email address.
        Prompt the user for a message if not already provided, and set an appropriate subject.
        Example: User: "Send a message to Ryan"
                 Max: "What would you like to say?"
                 User: "What would you like for your birthday?"
                 Message: "Hi Ryan, this is Max sending on behalf of <user>.
                 What would you like for your birthday?"
                 Subject: "Birthday question"
        Use this tool to send emails if the user wants to ask someone a question, send a message, email etc.
        """
        if not self.sender_email:
            error_msg = "Error: GOOGLE_SENDER_EMAIL environment variable is not set."
            logger.error(error_msg)
            return json.dumps({"error": error_msg})

        creds = await self._get_credentials()
        if not creds:
            error_msg = "Failed to get valid credentials. Run authentication."
            logger.error(error_msg)
            return json.dumps({"error": error_msg})

        try:
            # The 'build' function is blocking, run in a thread
            service = await asyncio.to_thread(
                build, "gmail", "v1", credentials=creds
            )
            message = self._create_message(to, subject, message_text)
            logger.debug(f"Sending email with to: '{to }' subject: '{subject}' body '{message_text}' encoded-message: {message}")

            # The '.execute()' call is blocking, run in a thread
            sent_message = await asyncio.to_thread(
                service.users().messages().send(userId="me", body=message).execute
            )

            success_msg = f"Message sent! Message ID: {sent_message['id']}"
            logger.info(success_msg)
            return json.dumps({"success": True, "message_id": sent_message['id']})

        except HttpError as error:
            error_msg = f"An error occurred while sending the email: {error}"
            logger.error(error_msg)
            return json.dumps({"error": str(error)})
        except Exception as e:
            error_msg = f"An unexpected error occurred: {e}"
            logger.error(error_msg)
            return json.dumps({"error": str(e)})

    def get_tools(self) -> list:
        """
        Returns a list of all tool methods bound to this instance.
        """
        return [
            StructuredTool.from_function(
                func=None,
                coroutine=self.send_message,
                name="send_gmail_message",
                description="Sends an email message to a recipient on the user's behalf.",
                args_schema=SendGmailArgs
            ),
        ]