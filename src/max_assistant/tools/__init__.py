from .person_tools import PersonTools
from .family_tools import FamilyTools
from .schedule_tools import ScheduleTools
from .gmail_tools import GmailTools
from .general_query_tools import GeneralQueryTools

# A central list of all tool provider classes to be registered.
# To add a new tool, simply import it and add its class to this list.
ALL_TOOL_PROVIDERS = [
    PersonTools,
    FamilyTools,
    ScheduleTools,
    GmailTools,
    GeneralQueryTools,
]

__all__ = [
    "ALL_TOOL_PROVIDERS",
    "PersonTools",
    "FamilyTools",
    "ScheduleTools",
    "GmailTools",
    "GeneralQueryTools",
]