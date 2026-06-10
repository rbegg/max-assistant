# tests/conftest.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Find the workspace root based on this file's position
TESTS_DIR = Path(__file__).parent.resolve()
WORKSPACE_ROOT = TESTS_DIR.parent

local_env = WORKSPACE_ROOT / '..' / '.env.local'

if local_env.exists():
    load_dotenv(local_env)
else:
    print(f"[TEST SETUP] Warning: Root env file not found at {local_env}")