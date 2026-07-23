# tests/function/test_yaml_gateway.py
import pytest
from tests.function.test_runner import execute_scenario_workflow

@pytest.mark.asyncio
async def test_execute_yaml_workflow(dynamic_scenario_data, request):
    """This standard test absorbs all the dynamic parameters generated from your YAMLs."""
    await execute_scenario_workflow(
        username=dynamic_scenario_data["username"],
        steps=dynamic_scenario_data["steps"],
        request=request
    )