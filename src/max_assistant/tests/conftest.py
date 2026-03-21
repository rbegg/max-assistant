import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_llm():
    return MagicMock(name="mock_llm")


@pytest.fixture
def mock_db_client():
    client = MagicMock(name="mock_db_client")
    client.execute_query = MagicMock()
    client.get_schema = MagicMock()
    return client


@pytest.fixture
def sample_user_info():
    return {
        "user": {
            "id": "1",
            "firstName": "Max",
            "lastName": "Assistant",
        },
        "location": {
            "name": "Home",
        },
    }


@pytest.fixture
def sample_graph_state(sample_user_info):
    return {
        "messages": [],
        "userinfo": sample_user_info,
        "thread_id": "thread-123",
        "transcribed_text": "",
        "voice": "en_US-hfc_female-medium",
    }


@pytest.fixture
def mock_message():
    class Message:
        def __init__(self, content):
            self.content = content
            self.tool_calls = []

    return Message


@pytest.fixture
def mock_tool_call_message():
    class Message:
        def __init__(self, content="", tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls or []

    return Message