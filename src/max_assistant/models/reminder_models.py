# max_assistant/models/reminder_models import BaseModel
from pydantic import BaseModel, Field

class ScheduleReminderArgs(BaseModel):
    message: str = Field(
        ...,
        description="The specific message or task detail the user wants to be reminded of (e.g., 'go to lunch')."
    )
    delay_minutes: float = Field(
        ...,
        description="The number of minutes from now to wait before triggering the reminder."
    )

