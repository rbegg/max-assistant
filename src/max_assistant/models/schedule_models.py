# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
Defines Pydantic models for the schedule-related tools and neo4j nodes
"""
from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import List, Optional, Any, Dict
from datetime import datetime, date, time
import logging

from max_assistant.models.base import BaseNeo4jModel

logger = logging.getLogger(__name__)


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