import json
import logging
from functools import wraps

logger = logging.getLogger(__name__)


def requires_db(func):
    """
    Decorator to ensure self.db_client is initialized before executing a method.
    Must be applied to class methods where 'self' has a 'db_client' attribute.
    Returns a JSON formatted error payload suitable for the LLM if missing.
    """

    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        if not getattr(self, "db_client", None):
            logger.error(f"Cannot execute '{func.__name__}': Database client is not initialized.")
            return json.dumps({
                "error": "Configuration_Error",
                "instruction": "The database client is missing. Inform the user that the system is misconfigured.",
                "details": "self.db_client is None"
            })

        # Proceed with the original method if the client exists
        return await func(self, *args, **kwargs)

    return wrapper