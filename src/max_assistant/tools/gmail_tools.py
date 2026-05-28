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
import base64
import logging
import json
import asyncio
from email.mime.text import MIMEText
from datetime import datetime
from typing import Annotated

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from langchain_ollama import ChatOllama
from langgraph.prebuilt import InjectedState
from langchain_core.tools import StructuredTool

from max_assistant.clients.neo4j_client import Neo4jClient, Neo4jClientError, Neo4jCircuitBreakerError
from max_assistant.config import (
    GOOGLE_SENDER_EMAIL, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET
)
from max_assistant.tools.base_provider import BaseToolProvider

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
# This is a constant for Google's token endpoint
TOKEN_URI = "https://oauth2.googleapis.com/token"

class GmailTools(BaseToolProvider):
    """
    An async class that encapsulates Gmail API operations.
    It requires a Neo4jClient to store and retrieve user credentials.
    """

    def __init__(self, db_client: Neo4jClient, llm: ChatOllama = None):
        """
        Initializes the toolset with a Neo4j client.
        """
        super().__init__(db_client, llm)
        self.sender_email = GOOGLE_SENDER_EMAIL
        self.client_id = GOOGLE_CLIENT_ID
        self.client_secret = GOOGLE_CLIENT_SECRET

        if not self.client_id or not self.client_secret:
            logger.error("FATAL: 'GOOGLE_CLIENT_ID' or 'GOOGLE_CLIENT_SECRET' "
                         "environment variables not set. GmailTools will not function.")

    async def authenticate(self, user_info: Annotated[dict, InjectedState("userinfo")]) -> str:
        """
        Runs the *initial* one-time authentication flow.
        Saves the refresh_token, access_token, and expiry to the :User node.

        Returns:
            A JSON-serialized string indicating success or failure status.
        """
        user_id = self._get_verified_user_id(user_info)

        params = {"user_id": user_id}

        check_query = "MATCH (u:User {id: $user_id}) RETURN u.gmailRefreshToken AS token"
        raw_result = await self._safe_execute_query(check_query, params)
        result = json.loads(raw_result)

        if "data" in result and result["data"] and result["data"][0].get("token"):
            logger.info("Gmail refresh token already exists. Skipping authentication.")
            return json.dumps({
                "success": True,
                "message": "Authentication already complete. Token is present on your user profile."
            })
        elif "error" in result:
            # Handles Database_Offline or Database_Unavailable gracefully
            return raw_result

        client_config = {
            "installed": {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": TOKEN_URI,
                "redirect_uris": ["http://localhost"]
            }
        }

        try:
            logger.info("Starting one-time authentication flow...")
            flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
            creds = await asyncio.to_thread(flow.run_local_server, port=0)

            if not creds.refresh_token:
                logger.error("Authentication failed: No refresh token was returned.")
                return json.dumps({
                    "success": False,
                    "error": "No_Refresh_Token",
                    "message": "Google did not return a refresh token. Ensure you have authorized the app."
                })

            # Save all user tokens and expiry to Neo4j
            set_query = """
                    MATCH (u:User {id: $user_id}) 
                    SET u.gmailRefreshToken = $refresh_token,
                        u.gmailAccessToken = $access_token,
                        u.gmailTokenExpiry = $expiry
                    """
            params = {
                "refresh_token": creds.refresh_token,
                "access_token": creds.token,
                "expiry": creds.expiry.isoformat(),
                "user_id": user_id,
            }
            await self.db_client.execute_query(set_query, params)
            logger.info("Authentication successful. Tokens saved to :User node.")
            return json.dumps({
                "success": True,
                "message": "Gmail setup complete! Your account is now linked."
            })

        except Neo4jClientError as e:
            logger.error(f"Authentication succeeded with Google, but failed to save to Neo4j: {e}")
            return json.dumps({
                "success": False,
                "error": "Database_Save_Failed",
                "message": "Authenticated with Google, but failed to store credentials in your database profile."
            })
        except Exception as e:
            logger.error(f"Unexpected error during authentication flow: {e}", exc_info=True)
            return json.dumps({
                "success": False,
                "error": "Unexpected_Error",
                "message": str(e)
            })

    async def _get_credentials(self, user_info: dict) -> Credentials | None:
        if not self.client_id or not self.client_secret:
            logger.error("Failed to get credentials. App secrets not set in env.")
            return None

        user_id = self._get_verified_user_id(user_info)

        # 1. Fetch tokens safely
        try:
            get_query = """
            MATCH (u:User {id: $user_id}) 
            RETURN u.gmailRefreshToken AS refresh_token,
                   u.gmailAccessToken AS access_token,
                   u.gmailTokenExpiry AS expiry
            """
            params = {"user_id": user_id}
            result = await self.db_client.execute_query(get_query, params)
            data = result.get("data", [{}])[0]

        except Neo4jCircuitBreakerError as e:
            logger.warning("Cannot fetch Gmail credentials. Database circuit is OPEN.")
            raise e
        except Neo4jClientError as e:
            logger.error(f"Database error fetching credentials: {e}")
            return None

        refresh_token = data.get("refresh_token")
        access_token = data.get("access_token")
        expiry_str = data.get("expiry")

        if not refresh_token:
            logger.error("No Gmail refresh token found on :User node.")
            return None

        try:
            expiry_dt = datetime.fromisoformat(expiry_str) if isinstance(expiry_str, str) else None
            creds = Credentials(
                token=access_token,
                refresh_token=refresh_token,
                token_uri=TOKEN_URI,
                client_id=self.client_id,
                client_secret=self.client_secret,
                scopes=SCOPES,
                expiry=expiry_dt
            )

            # Refresh token logic
            if creds and creds.expired and creds.refresh_token:
                logger.info("Access token is expired. Refreshing...")
                await asyncio.to_thread(creds.refresh, Request())

                # Save new token safely
                try:
                    set_query = """
                    MATCH (u:User {id: $user_id}) 
                    SET u.gmailAccessToken = $access_token,
                        u.gmailTokenExpiry = $expiry
                    """
                    params = {
                        "access_token": creds.token,
                        "expiry": creds.expiry.isoformat(),
                        "user_id": user_id,
                    }
                    await self.db_client.execute_query(set_query, params)
                    logger.info("Access token refreshed and saved back to Neo4j.")
                except Neo4jClientError as e:
                    # Non-fatal error. We have valid creds in memory for this run,
                    # we just failed to cache them for next time.
                    logger.warning(f"Failed to save refreshed token to database: {e}")

            elif creds.valid:
                logger.info("Using cached, valid access token.")

        except Exception as e:
            logger.error(f"Failed to refresh access token: {e}")
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

    async def send_message(
            self,
            to: str,
            subject: str,
            message_text: str,
            user_info: Annotated[dict, InjectedState("userinfo")]
    ) -> str:
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
        Use this tool to send emails if the user wants to ask someone a question, send a message, email, etc.
        """
        if not self.sender_email:
            error_msg = "Error: GOOGLE_SENDER_EMAIL environment variable is not set."
            logger.error(error_msg)
            return json.dumps({"error": error_msg})

        try:
            # This will raise an error if the DB is offline,
            # or return None if the user just isn't authenticated.
            creds = await self._get_credentials(user_info)

            if not creds:
                # SCENARIO A: The DB is fine, but the user isn't logged in.
                logger.warning("Email tool aborted: User is not authenticated.")
                return json.dumps({
                    "error": "Authentication_Required",
                    "instruction": "You cannot send the email because you do not have Gmail credentials on file. Tell the user they need to run the authentication setup first."
                })

        except Neo4jCircuitBreakerError:
            # SCENARIO B: The DB is offline.
            logger.warning("Email tool aborted: Database circuit is OPEN.")
            return json.dumps({
                "error": "Database_Offline_Circuit_Open",
                "instruction": "You cannot send the email because the system database is offline and you cannot retrieve your credentials. Apologize and inform the user."
            })

        service = await asyncio.to_thread(
            build, "gmail", "v1", credentials=creds
        )
        try:
            message = self._create_message(to, subject, message_text)
            logger.debug(f"Sending email with to: '{to}' subject: '{subject}' body '{message_text}' encoded-message: {message}")

            # The '.execute()' call is blocking, run in a thread
            sent_message = await asyncio.to_thread(
                service.users().messages().send(userId="me", body=message).execute
            )

            success_msg = f"Message sent! Message ID: {sent_message['id']}"
            logger.info(success_msg)
            return json.dumps({"success": True, "message_id": sent_message['id']})

        except HttpError as error:
            logger.error(f"Google API HttpError: {error}")
            return json.dumps({
                "error": "External_API_Failure",
                "message": "The Gmail service refused or failed the request.",
                "details": str(error)
            })

        except Exception as e:
            error_msg = f"An unexpected error occurred: {e}"
            logger.error(error_msg)
            return json.dumps({"error": str(e)})
        finally:
            service.close()

    def get_tools(self) -> list:
        """
        Returns a list of all tool methods bound to this instance.
        """
        return [
            StructuredTool.from_function(
                func=None,
                coroutine=self.send_message,
                name="send_gmail_message",
                description=self.send_message.__doc__,
                handle_tool_error=self.format_system_tool_error,
            ),
        ]