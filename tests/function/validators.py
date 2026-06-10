import re
from max_assistant.clients.neo4j_client import Neo4jClient


def assert_substring_present(expected_substring: str):
    """Verifies clear token visibility inside the raw response string."""

    def validator(response_text: str, db_client: Neo4jClient):
        assert expected_substring.lower() in response_text.lower(), \
            f"Expected substring '{expected_substring}' missing from response: '{response_text}'"

    return validator


from typing import List
from max_assistant.clients.neo4j_client import Neo4jClient


def assert_substrings_present(expected_substrings: List[str], match_all: bool = True):
    """
    Evaluates an array of substrings against the response text in a single pass.

    :param expected_substrings: List of string tokens to search for.
    :param match_all: If True, all strings must match. If False, at least one must match.
    """

    def validator(response_text: str, db_client: Neo4jClient):
        normalized_response = response_text.lower()

        # Track which substrings were missing
        missing = [
            sub for sub in expected_substrings
            if sub.lower() not in normalized_response
        ]

        if match_all and missing:
            assert False, (
                f"Substring validation failed! Response was missing the following required "
                f"tokens: {missing}. \nFull Response: '{response_text}'"
            )

        elif not match_all and len(missing) == len(expected_substrings):
            assert False, (
                f"Any-match validation failed! Response did not match any of the provided "
                f"tokens: {expected_substrings}. \nFull Response: '{response_text}'"
            )

    return validator


def assert_neo4j_node_exists(label: str, property_key: str, property_value: str):
    """
    Queries Neo4j directly to confirm a data-mutating transaction
    successfully executed under the hood.
    """

    def validator(response_text: str, db_client: Neo4jClient):
        # We run an explicit async query check inside a synchronous wrapper loop
        # managed safely by our outer test execution step.
        loop = asyncio.get_running_loop()
        query = f"MATCH (n:{label} {{{property_key}: $value}}) RETURN count(n) AS node_count"

        # Safe async runner wrapper execution
        async def run_query():
            records, _, _ = await db_client.execute_query(query, {"value": property_value})
            return records[0]["node_count"] if records else 0

        count = loop.run_until_complete(run_query())
        assert count > 0, f"Graph verification failed: Node (:{label} {{{property_key}: '{property_value}'}}) not found!"

    return validator