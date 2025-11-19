import asyncio
import csv
import yaml
import json
from collections import defaultdict
from argparse import ArgumentParser

# --- MODIFIED ---
# Remove the synchronous GraphDatabase import
from max_assistant.scripts.local_config import SCRIPT_DIR
from max_assistant.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from max_assistant.clients.neo4j_client import Neo4jClient

# --- Configuration ---
DATA_DIR = SCRIPT_DIR / "../../../csv_data"
NODE_CONFIG_FILE = DATA_DIR / "nodes.yaml"
RELATIONSHIP_CONFIG_FILE = DATA_DIR / "relationships.yaml"


def print_banner(text):
    BANNER = "*" * (len(text) + 8)
    print(f"{BANNER}\n*** {text:} ***\n{BANNER}")


# --- MODIFIED ---
async def run_query(client: Neo4jClient, query, params=None):
    """Helper to run a query using the async Neo4jClient."""
    try:
        # Use the async client's execute_query method
        result_dict = await client.execute_query(query, params)

        # Handle the dictionary-based error/data format
        if "error" in result_dict:
            print(f"Error running query:\n{query}\nParams: {params}\n{result_dict.get('message')}")
            return None  # Return None on error, like the old function

        # Return the 'data' list, which mimics the old result.data()
        return result_dict.get("data", [])

    except Exception as e:
        print(f"Error running query:\n{query}\nParams: {params}\n{e}")
        return None


# --- MODIFIED ---
async def clear_database(client: Neo4jClient):
    """Wipes all nodes and relationships from the database. Be careful!"""
    print("    Clearing database ...", end='')
    await run_query(client, "MATCH (n) DETACH DELETE n")
    await run_query(client, "CALL apoc.schema.assert({}, {})")
    print(" Database cleared.✔")


def process_csv_file(file_path):
    """
    Reads a CSV file and loads it into a list of dictionaries.
    (This function is I/O bound, no async needed)
    """
    data = []
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned_row = {k: (v if v != "" else None) for k, v in row.items()}
            data.append(cleaned_row)
    return data


# --- MODIFIED ---
async def process_nodes(client: Neo4jClient, nodes):
    print_banner("Processing Nodes")
    # Loop through each node defined in the YAML
    for node in nodes:
        print(f"    Loading {node['name']} ...", end='')

        if node['constraints']:
            print(" Creating constraints...", end='')
            for c in node['constraints']:
                await run_query(client, c)  # Use await

        if node['filename']:
            try:
                # Process the CSV file in Python
                file_path = DATA_DIR / node['filename']
                data_batch = process_csv_file(file_path)

                if data_batch:
                    # Use await
                    r = await run_query(client, node['query'], {'data': data_batch})

                    # The new run_query returns None on error
                    if r is None:
                        print(f"    Query failed for node {node['name']}.❌")
                    elif len(r) == 0:
                        print(f"    No summary returned for node {node['name']}.❌")
                    else:
                        node_count = r[0]['count']
                        flag_char = '✔' if node_count == len(data_batch) else '❌'
                        print(f" Processed {len(data_batch)} rows - created {node_count} nodes .{flag_char}")
                else:
                    print(f" No data found in {node['name']}.❌")

            except Exception as e:
                print(f"      !!! FAILED loading node: {node['name']} !!!❌")
                print(f"      Error: {e}\n")


# --- MODIFIED ---
async def process_relationships(client: Neo4jClient, relationships):
    print_banner("Processing Relationships")

    for rel in relationships:
        print(f"    Loading {rel['name']} ... ", end='')
        file_path = DATA_DIR / rel['filename']
        rel_data = process_csv_file(file_path)

        if rel_data:
            # Use await
            r = await run_query(client, rel['query'], {'data': rel_data})

            # The new run_query returns None on error
            if r is None:
                print(f"    Query failed for relationship {rel['name']}.❌")
            elif len(r) == 0:
                print(f"    No summary returned for {rel['name']}.❌")
            else:
                rel_count = r[0]['count']
                flag_char = '✔' if rel_count == len(rel_data) else '❌'
                print(f"Processed {len(rel_data)} rows, wrote {rel_count} relationships.{flag_char}")
        else:
            print(f"    No data found in {rel['name']}.❌")


# --- DELETED ---
# The entire old `get_schema(driver)` function has been removed.
# We will now use the superior `client.get_schema()` method.


# --- MODIFIED ---
async def main():
    parser = ArgumentParser(description="Load data into Neo4j and optionally export schema.")
    parser.add_argument(
        '--schema-file',
        type=str,
        help='If provided, export the DB schema to this file path *after* data loading.'
    )
    args = parser.parse_args()

    # Use the async client
    client = await Neo4jClient.create(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

    # Remove the old sync driver

    print_banner("Data Loading Starting!")
    await clear_database(client)  # Use await

    with open(NODE_CONFIG_FILE, 'r') as f:
        nodes = yaml.safe_load(f)
        await process_nodes(client, nodes)  # Use await

    with open(RELATIONSHIP_CONFIG_FILE, 'r') as f:
        relationships = yaml.safe_load(f)
        await process_relationships(client, relationships)  # Use await

    print_banner("Data Loading Complete!")

    if args.schema_file:
        print_banner("Exporting Schema!")
        print(f"    Exporting schema to: {args.schema_file}")

        # --- MODIFIED SCHEMA EXPORT ---
        # Call the client's built-in, async, APOC-based get_schema method
        schema_json_string = await client.get_schema()  #

        schema_data = None
        if schema_json_string:
            try:
                # Check if the string is an error message
                schema_data = json.loads(schema_json_string)
                if isinstance(schema_data, dict) and "error" in schema_data:
                    print(f"Could not export schema, error from client: {schema_data.get('message')} ❌")
                    schema_data = None
            except json.JSONDecodeError:
                print(f"Could not parse schema from client: {schema_json_string} ❌")
                schema_data = None

        if schema_data:
            try:
                # Write the schema data (which is a dict) as JSON
                with open(args.schema_file, 'w', encoding='utf-8') as f:
                    json.dump(schema_data, f, ensure_ascii=False, indent=4)
                print(f"   Successfully wrote schema ✔")
            except Exception as e:
                print(f"Error writing schema to file:❌\n    Error: {e}")
        else:
            print("Could not export schema, data was empty or invalid.❌")

    await client.close()  # Use async close


if __name__ == "__main__":
    # --- MODIFIED ---
    # Run the async main function
    asyncio.run(main())