'''
    Load environment variables for running scripts in local dev/test mode.
    Import this file before src.config or any other src files.
'''
from pathlib import Path

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).parent.resolve()
local_env_file = SCRIPT_DIR / '../../.env.local'
if not load_dotenv(local_env_file):
    print(f"Failed to load environment variables from {local_env_file}.")
    exit(1)