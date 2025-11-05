# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
Defines Pydantic models for the schedule-related tools and neo4j nodes
"""
from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import List, Optional, Any, Dict
from datetime import datetime, date, time
import logging

logger = logging.getLogger(__name__)


class BaseNeo4jModel(BaseModel):
    """
    A base model that handles common Neo4j data conversions.
    - Coerces integer IDs to strings.
    - Converts neo4j.time.Time/Date objects to native Python types.
    """
    model_config = ConfigDict(
        from_attributes=True,
        coerce_numbers_to_str=True,  # Fixes id: int -> str globally
    )

    @model_validator(mode='before')
    @classmethod
    def _convert_neo4j_types(cls, data: Any) -> Any:
        """
        Runs before any other validation.
        Iterates over the input dict and converts Neo4j types.
        """
        if isinstance(data, dict):
            for key, value in data.items():
                if hasattr(value, 'to_native'):
                    # This converts neo4j.time.Time/Date to datetime.time/date
                    data[key] = value.to_native()
        return data


class Appointment(BaseNeo4jModel):  # <-- Inherits from our base model
    """Represents an Appointment node from Neo4j."""
    id: str
    title: str
    time: time
    date: date
    duration: Optional[int] = None
    details: Optional[str] = None



class DailyRoutine(BaseNeo4jModel):  # <-- Also inherits
    """Represents a DailyRoutine node from Neo4j."""
    # id, title, type, dayOfWeek, time, duration, startDate, endDate, room, details, rating
    id: str
    title: str
    type: str
    dayOfWeek: List[str] = Field(..., alias="dayOfWeek")
    time: time
    duration: Optional[int] = None
    startDate: date                 # date activity was first offered
    endDate: Optional[date] = None  # date activity was last offered
    room: Optional[str] = None
    details: Optional[str] = None
    rating: Optional[str] = None


class CreateAppointmentArgs(BaseModel):
    """Defines the arguments for the create_appointment tool."""
    title: str = Field(..., description="The main title of the appointment.")
    date: str = Field(..., description="The date in YYYY-MM-DD format.")
    details: Optional[str] = Field(None, description="Optional notes or details.")
    duration: Optional[int] = Field(None, description="Duration in minutes.")