'''
    Load environment variables for running scripts in local dev/test mode.
    Import this file before src.config or any other src files.
'''
from pathlib import Path
from dotenv import load_dotenv

# Define paths relative to this configuration file cleanly
SCRIPT_DIR = Path(__file__).parent.resolve()
local_env_file = SCRIPT_DIR / '../../../.env.local'

def init_environment(required=True):
    """
    Centralized hook to load local environment layers dynamically.
    If 'required' is True, it will exit the process on failure.
    """
    if not load_dotenv(local_env_file):
        print(f"Failed to load environment variables from {local_env_file}.")
        if required:
            exit(1)
        return False
    return True


if __name__ == "__main__":
    init_environment()