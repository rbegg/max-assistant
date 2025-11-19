from pydantic import BaseModel, Field

class SendGmailArgs(BaseModel):
    """Arguments for the send_message tool."""
    to: str = Field(..., description="The recipient's email address.")
    subject: str = Field(..., description="The subject line of the email.")
    message_text: str = Field(..., description="The plain text body of the email.")