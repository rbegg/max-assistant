import inspect
from typing import List, Dict, Any
from max_assistant.app_services import AppServices
from max_assistant.agent.agent import Agent
from max_assistant.tools import PersonTools
from max_assistant.config import OLLAMA_MODEL_NAME


async def execute_scenario_workflow(username: str, steps: List[Dict[str, Any]], request=None):
    """
    A test execution engine for chat scenarios defined in an injection array.
    Supports both synchronous token validators and asynchronous semantic/graph validators.
    """

    app_services = await AppServices.create()

    if not app_services.llm_ready_event.is_set():
        await app_services.llm_ready_event.wait()

    person_tools = PersonTools(app_services.db_client)
    user_data = await person_tools.get_user_info_internal(username)
    if "error" in user_data:
        user_data = {}

    agent = Agent(app_services.reasoning_engine, user_data)
    thread_id = agent.get_thread_id()

    # This creates a highly scannable visual header inside PyCharm's console window
    print("\n" + "=" * 80)
    print(f" WORKING THREAD ID : {thread_id}")
    print(f" USERNAME          : {username}")
    print(f" USERNAME          : {OLLAMA_MODEL_NAME}")
    print("=" * 80 + "\n")

    # Execute conversational array step-by-step
    try:
        for step in steps:
            user_input = step["user_input"]
            validators = step.get("validators", [])

            # Programmatic code execution bypassing AsyncConsoleReader/sys.stdin
            actual_response = await agent.ainvoke(user_input)

            # Fire off pluggable validators conditionally
            for validator_fn in validators:
                if inspect.iscoroutinefunction(validator_fn):
                    # Natively await async validators (like semantic evaluations or graph checks)
                    await validator_fn(actual_response, app_services.db_client)
                else:
                    # Execute standard synchronous substring/regex assertions directly
                    validator_fn(actual_response, app_services.db_client)

    finally:
        # Guarantee safe database connection pooling teardown
        if app_services.db_client:
            await app_services.db_client.close()