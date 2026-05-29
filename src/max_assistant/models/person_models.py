# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
Defines Pydantic models for the person-related tools and neo4j nodes
"""
from datetime import date
import logging

from max_assistant.models.base import BaseNeo4jModel

logger = logging.getLogger(__name__)


class PersonDetails(BaseNeo4jModel):
    """
    Validates the properties of a Person, Family, Friend, or Support node.
    Inherits from BaseNeo4jModel to handle type conversions.
    """
    # model_config is inherited from BaseNeo4jModel
    id: str
    firstName: str | None = None
    lastName: str | None = None
    title: str | None = None # From Person and Support nodes
    dob: date | None = None
    dod: date | None = None
    gender: str | None = None
    email: str | None = None
    phone: str | None = None
    notes: str | None = None
    startDate: date | None = None # From Person and Support nodes
    endDate: date | None = None # From Person and Support nodes
