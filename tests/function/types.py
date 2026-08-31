from typing import TypedDict, List

# 1. Type definition for individual step metrics
class StepResult(TypedDict):
    step: int
    user_input: str
    actual_result: str
    elapsed_time: float

# 2. Type definition for the overarching scenario payload
class ScenarioResult(TypedDict, total=False):
    testcase_name: str
    thread_id: str | None
    model: str
    success: bool
    total_execution_time: float
    step_results: List[StepResult]