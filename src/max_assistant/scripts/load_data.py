import csv
import yaml
import json
from neo4j import GraphDatabase
from argparse import ArgumentParser


from max_assistant.scripts.local_config import SCRIPT_DIR
from max_assistant.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

# --- Configuration ---
DATA_DIR = SCRIPT_DIR / "../../csv_data"
NODE_CONFIG_FILE = DATA_DIR / "nodes.yaml"
RELATIONSHIP_CONFIG_FILE = DATA_DIR / "relationships.yaml"

def print_banner(text):
    BANNER = "*" * (len(text) + 8)
    print(f"{BANNER}\n*** {text:} ***\n{BANNER}")


def run_query(driver, query, params=None):
    """Helper to run a query with automatic session management."""
    with driver.session() as session:
        try:
            result = session.run(query, params)
            return result.data()
        except Exception as e:
            print(f"Error running query:\n{query}\nParams: {params}\n{e}")
            return None


def clear_database(driver):
    """Wipes all nodes and relationships from the database. Be careful!"""
    print("    Clearing database ...", end='')
    run_query(driver, "MATCH (n) DETACH DELETE n")
    run_query(driver, "CALL apoc.schema.assert({}, {})")
    print(" Database cleared.✔")


def process_csv_file(file_path):
    """
    Reads a CSV file and loads it into a list of dictionaries.
    This is where you would put your Python data cleaning logic.
    """
    data = []
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned_row = {k: (v if v != "" else None) for k, v in row.items()}
            data.append(cleaned_row)
    return data


def process_nodes(driver, nodes):
    print_banner("Processing Nodes")
    # Loop through each node defined in the YAML
    for node in nodes:
        print(f"    Loading {node['name']} ...", end='')

        if node['constraints']:
            print(" Creating constraints...", end='')
            for c in node['constraints']:
                run_query(driver, c)

        if node['filename']:
            try:
                # Process the CSV file in Python
                file_path = DATA_DIR / node['filename']
                data_batch = process_csv_file(file_path)

                # Run the Cypher query from the YAML, passing the
                # Python data as the $data parameter.
                if data_batch:
                    r = run_query(driver, node['query'], {'data': data_batch})
                    if len(r) == 0:
                        print(f"    No nodes writen for node {node['name']}")
                    else:
                        node_count = r[0]['count']
                        flag_char = '✔' if node_count == len(data_batch) else '❌'
                        print(f" Processed {len(data_batch)} rows - created {node_count} nodes .{flag_char}")
                else:
                    print(f" No data found in {node['name']}.❌")

            except Exception as e:
                print(f"      !!! FAILED loading node: {node['name']} !!!❌")
                print(f"      Error: {e}\n")


def process_relationships(driver, relationships):
    print_banner("Processing Relationships")

    for rel in relationships:
        print(f"    Loading {rel['name']} ... ", end='')
        file_path = DATA_DIR / rel['filename']
        rel_data = process_csv_file(file_path)

        if rel_data:
            r = run_query(driver, rel['query'], {'data': rel_data} )
            if len(r) == 0:
                print(f"    No relationships created for {rel['name']}.❌")
            else:
                rel_count = r[0]['count']
                flag_char = '✔' if rel_count == len(rel_data) else '❌'
                print(f"Processed {len(rel_data)} rows, wrote {rel_count} relationships.{flag_char}")
        else:
            print(f"    No data found in {rel['name']}.❌")


from collections import defaultdict


def get_schema(driver):
    """
    Fetches the graph schema using three separate built-in procedures
    to correctly combine structure AND data properties for an LLM.
    """

    # --- Step 1: Get all Node properties ---
    # This is safe for the helper. It returns a list of dicts.
    node_props_query = "CALL db.schema.nodeTypeProperties()"
    node_props_data = run_query(driver, node_props_query)

    # --- Step 2: Get all Relationship properties ---
    # This is also safe for the helper.
    rel_props_query = "CALL db.schema.relTypeProperties()"
    rel_props_data = run_query(driver, rel_props_query)

    # --- Step 3: Get the graph structure ---
    # This query MUST be run with a local session to get live objects.
    viz_record = None
    try:
        with driver.session() as session:
            result = session.run("CALL db.schema.visualization()")
            viz_record = result.single()

    except Exception as e:
        print(f"    Error running schema visualization query: {e}")
        return None

    if not node_props_data or not rel_props_data or not viz_record:
        print("    Error: Failed to get complete schema info.")
        return None

    # --- Process Step 1: Build Node Property Lookup Map ---
    # Use defaultdict for cleaner code
    node_properties_map = defaultdict(list)
    for row in node_props_data:
        # The row['nodeLabels'] is a list, get the primary one
        if row['nodeLabels']:
            label = row['nodeLabels'][0]
            node_properties_map[label].append(row['propertyName'])

    # --- Process Step 2: Build Relationship Property Lookup Map ---
    rel_properties_map = defaultdict(list)
    for row in rel_props_data:
        # relType is a string like '`KNOWS`', so we strip the backticks
        rel_type = row['relType'].strip('`')
        rel_properties_map[rel_type].append(row['propertyName'])

    # --- Process Step 3: Build Schema Strings using Lookups ---
    node_labels_set = set()
    rel_types_set = set()
    node_schema_strings = []
    rel_schema_strings = []

    # Use the visualization record for node labels
    for node in viz_record['nodes']:
        if not node.labels:
            continue
        label = list(node.labels)[0]
        node_labels_set.add(label)

        # Use our new lookup map to get the *real* properties
        properties = sorted(node_properties_map.get(label, []))
        node_schema_strings.append(f"{label} {{properties: {properties}}}")

    # Use the visualization record for relationship structure
    for rel in viz_record['relationships']:
        rel_type = rel.type
        rel_types_set.add(rel_type)

        start_label = list(rel.start_node.labels)[0]
        end_label = list(rel.end_node.labels)[0]

        # Use our new lookup map to get the *real* properties
        properties = sorted(rel_properties_map.get(rel_type, []))
        rel_schema_strings.append(
            f"({start_label})-[:{rel_type} {{properties: {properties}}}]->({end_label})"
        )

    # --- Combine into the final dictionary ---
    final_schema = {
        "node_labels": sorted(list(node_labels_set)),
        "relationship_types": sorted(list(rel_types_set)),
        "schema": sorted(list(set(node_schema_strings))) + sorted(list(set(rel_schema_strings)))
        # Use set to remove duplicates
    }

    print("    Successfully formatted schema with properties.✔")
    return final_schema


def main():
    parser = ArgumentParser(description="Load data into Neo4j and optionally export schema.")
    parser.add_argument(
        '--schema-file',
        type=str,
        help='If provided, export the DB schema to this file path *after* data loading.'
    )
    args = parser.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    print_banner("Data Loading Starting!")
    clear_database(driver)

    with open(NODE_CONFIG_FILE, 'r') as f:
        nodes = yaml.safe_load(f)
        process_nodes(driver, nodes)

    with open(RELATIONSHIP_CONFIG_FILE, 'r') as f:
        relationships = yaml.safe_load(f)
        process_relationships(driver, relationships)

    print_banner("Data Loading Complete!")

    if args.schema_file:
        print_banner("Exporting Schema!")
        print(f"    Exporting schema to: {args.schema_file}")
        schema_data = get_schema(driver)

        if schema_data:
            try:
                # Write the schema data as JSON to the specified file
                with open(args.schema_file, 'w', encoding='utf-8') as f:
                    json.dump(schema_data, f, ensure_ascii=False, indent=4)
                print(f"   Successfully wrote schema")
            except Exception as e:
                print(f"Error writing schema to file:❌\n    Error: {e}")
        else:
            print("Could not export schema, data was empty.❌")

    driver.close()
if __name__ == "__main__":
    main()