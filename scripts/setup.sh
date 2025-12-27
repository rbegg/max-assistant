#!/bin/bash
# Get the directory where the script is located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SRC_DIR=$SCRIPT_DIR/../src

cd "$SCRIPT_DIR"

# Load the variables from the file
ENV_FILE="../.env.local"

# Load and export variables, skipping comments and empty lines
if [ -f "$ENV_FILE" ]; then
    # Filter out lines starting with # and empty lines, then export
    export $(grep -v '^#' "$ENV_FILE" | grep -v '^\s*$' | xargs)
    echo "Successfully loaded variables from $ENV_FILE"
    echo $NEO4J_USER
else
    echo "Error: $ENV_FILE not found."
    exit 1
fi

echo "***"
echo "*** loading data"
echo "***"

echo change to $SRC_DIR
cd $SRC_DIR
pwd
python3 -m max_assistant.scripts.load_data

echo "***"
echo "*** authenticating gmail credentials from ENV"
echo "***"

export BROWSER=wslview

python3 -m max_assistant.scripts.gmail_authenticate $GOOGLE_SENDER_EMAIL
