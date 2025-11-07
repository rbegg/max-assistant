# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
Defines Pydantic models for the person-related tools and neo4j nodes
"""
from typing import Optional
from pydantic import BaseModel, Field
from datetime import date  # <-- Import Python's 'date' type
import logging

from max_assistant.models.base import BaseNeo4jModel

logger = logging.getLogger(__name__)


class PersonDetails(BaseNeo4jModel):
    """
    Validates the properties of a Person, Family, Friend, or Support node.
    Inherits from BaseNeo4jModel to handle type conversions.
    """
    # model_config is now inherited from BaseNeo4jModel

    # 'id' will be coerced from int to str by the base model's config
    id: str
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    title: Optional[str] = None  # From Person and Support nodes

    # These types are now 'date' and will be converted by the validator
    dob: Optional[date] = None
    dod: Optional[date] = None
    gender: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    notes: Optional[str] = None
    startDate: Optional[date] = None  # From Person and Support nodes
    endDate: Optional[date] = None  # From Person and Support nodes


class FindPersonByNameArgs(BaseModel):
    """Input arguments for the find_person_by_name tool."""
    first_name: Optional[str] = Field(
        default=None,
        description="The person's first name. Case-insensitive, partial match."
    )
    last_name: Optional[str] = Field(
        default=None,
        description="The person's last name. Case-insensitive, partial match."
    )


class FindPersonByTitleArgs(BaseModel):
    """Input arguments for the find_person_by_title tool."""
    title: str = Field(
        ...,
        description="The person's title, e.g., 'Doctor' or 'Nurse'. Case-insensitive, partial match."
    )


class GetRelationshipArgs(BaseModel):
    """Input arguments for the get_relationship_to_user tool."""
    first_name: str = Field(..., description="The person's first name.")
    last_name: str = Field(..., description="The person's last name.")

class GetUserInfoArgs(BaseModel):
    """Arguments for get_user_info tool. Takes no input."""
    pass