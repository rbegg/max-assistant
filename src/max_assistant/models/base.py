# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
from pydantic import BaseModel, ConfigDict, model_validator
from typing import Any

class BaseNeo4jModel(BaseModel):
    """
    A base model that handles common Neo4j data conversions.
    - Coerces integer IDs to strings.
    - Converts neo4j.time.Time/Date objects to native Python types.
    """
    model_config = ConfigDict(
        from_attributes=True,
        coerce_numbers_to_str=True,  # Fixes id: int -> str globally
        extra='ignore',
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