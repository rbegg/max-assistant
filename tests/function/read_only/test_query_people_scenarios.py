import pytest
from sqlalchemy import false
from tests.function.test_runner import execute_scenario_workflow
from tests.function.validators import assert_substring_present, assert_substrings_present, assert_semantic_criteria

SCENARIOS = {
    "margaret's family": {
        "username": "Margaret.Miller",
        "steps": [
            {
                "user_input": "Who is my spouse?",
                "validators": [
                    assert_substring_present("Thomas")
                ]
            },
            {
                "user_input": "Who are my children?",
                "validators": [
                    assert_substring_present("Jennifer"),
                    assert_substring_present("Michael"),
                ]
            },
            {
                "user_input": "What are the names of my grandchildren?",
                "validators": [
                    assert_substrings_present(["Emily", "Ryan", "Olivia"], True)
                ]
            },
            {
                "user_input": "Do I have great grandchildren?",
                "validators": [
                    assert_substring_present("Anne"),
                ]
            },
            {
                "user_input": "I have forgotten the names of my parents",
                "validators": [
                    assert_substrings_present(["Eleanor", "Smith", "Robert"], True)
                ]
            },
            {
                "user_input": "What was my mother's maiden name",
                "validators": [
                    assert_substrings_present(["Johnson"], True)
                ]
            },
            {
                "user_input": "What was my mother's maiden name",
                "validators": [
                    assert_substrings_present(["Johnson"], True)
                ]
            },
            {
                "user_input": "What is my brother's name",
                "validators": [
                    assert_substrings_present(["David"], True)
                ]
            },
            {
                "user_input": "Do I have a sister",
                "validators": [
                    assert_semantic_criteria(["Verify that Margaret does not have a sister"])
                ]
            },
        ]
    },
    "Margaret's connection to Arthur": {
        "username": "Margaret.Miller",
        "steps": [
            {
                "user_input": "Who is Arthur",
                "validators": [
                    assert_substrings_present(["Smith", "Grandfather","Black","friend"], True)
                ]
            },
            {
                "user_input": "Tell me more about my grandfather",
                "validators": [
                    assert_substrings_present(["born", "1895", "Smith", "Paternal"], True)
                ]
            },
            {
                "user_input": "How many kids did he have",
                "validators": [
                    assert_substrings_present(["three", "3", ], False)
                ]
            },
            {
                "user_input": "What were their names",
                "validators": [
                    assert_substrings_present(["Clara", "Robert", "George"], True)
                ]
            },
        ]
    }
}

@pytest.mark.asyncio
@pytest.mark.parametrize("scenario_name", SCENARIOS.keys())
async def test_agent_declarative_scenarios(scenario_name: str, request):
    """
    Pytest execution loop wrapper. Generates clear, isolated turns
    for every scenario tracked inside the declarative map registry.
    """
    scenario_data = SCENARIOS[scenario_name]
    await execute_scenario_workflow(
        username=scenario_data["username"],
        steps=scenario_data["steps"],
        request=request
    )