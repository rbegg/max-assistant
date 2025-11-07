# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
Defines Pydantic models for the location-related tools
"""
from pydantic import BaseModel, Field
from typing import Optional
from max_assistant.models.base import BaseNeo4jModel

class LocationDetails(BaseNeo4jModel):
    """Pydantic model for Location node properties."""
    address: Optional[str] = None
    id: Optional[str] = Field(None, description="Unique identifier for the location")
    name: Optional[str] = None
    room: Optional[str] = None
    type: Optional[str] = None