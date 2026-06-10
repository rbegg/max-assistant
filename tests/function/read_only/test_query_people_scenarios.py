import pytest
from sqlalchemy import false
from tests.function.test_runner import execute_scenario_workflow
from tests.function.validators import assert_substring_present, assert_substrings_present,assert_neo4j_node_exists

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
            }
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
async def test_agent_declarative_scenarios(scenario_name: str):
    """
    Pytest execution loop wrapper. Generates clear, isolated turns
    for every scenario tracked inside the declarative map registry.
    """
    scenario_data = SCENARIOS[scenario_name]
    await execute_scenario_workflow(
        username=scenario_data["username"],
        steps=scenario_data["steps"]
    )