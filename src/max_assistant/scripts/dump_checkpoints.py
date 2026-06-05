import os
import sys
import json
import base64
import binascii
from datetime import datetime
from neo4j import GraphDatabase

# Import official class components and serialization engines natively
from langchain_neo4j import Neo4jSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# Configuration: Reads your environment variables with default local fallbacks
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


def format_timestamp(version_str):
    """Extracts hex epoch strings into clean human-readable date-time entries."""
    if not version_str or version_str == "null":
        return "N/A"
    try:
        clean_hex = version_str.split('.')[0]
        if clean_hex.isdigit():
            ts = float(clean_hex) / 1000.0 if len(clean_hex) > 11 else float(clean_hex)
        else:
            ts = int(clean_hex, 16) / 1000.0 if len(clean_hex) > 8 else int(clean_hex, 16)
        return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        return "N/A"


def print_formatted_message(msg):
    """Natively extracts and formats properties from hydrated LangChain message objects."""
    if not msg:
        return

    msg_type = getattr(msg, "type", "message")
    content = getattr(msg, "content", "")
    tool_calls = getattr(msg, "tool_calls", [])
    tool_call_id = getattr(msg, "tool_call_id", "Unknown")

    if isinstance(msg, dict):
        msg_type = msg.get("type", "message")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])
        tool_call_id = msg.get("tool_call_id", "Unknown")

    if msg_type == "human":
        token = "[Human]"
    elif msg_type == "ai":
        token = "[AI]"
    elif msg_type == "tool":
        token = f"[Tool (ID: {tool_call_id})]"
    elif msg_type == "system":
        token = "[System]"
    else:
        token = f"[{str(msg_type).upper()}]"

    content_str = json.dumps(content, indent=4) if isinstance(content, (dict, list)) else str(content)

    print(f"      {token}:")
    if "\n" in content_str:
        for line in content_str.splitlines():
            print(f"          {line}")
    else:
        print(f"          {content_str}")

    if tool_calls:
        print("      Action [Tool Calls]:")
        for call in tool_calls:
            name = getattr(call, 'name', call.get('name') if isinstance(call, dict) else 'Unknown')
            args = getattr(call, 'args', call.get('args') if isinstance(call, dict) else {})
            print(f"          • Tool Name: {name}")
            print(f"            Arguments: {json.dumps(args)}")


def get_threads_timeline():
    """Queries structural channels to find all unique thread identifiers in the database."""
    query = """
    MATCH (c:Checkpoint)-[:HAS_CHANNEL]->(ch:ChannelState)
    WITH CASE 
           WHEN c.id CONTAINS '.' THEN split(c.id, '.')[0]
           WHEN c.thread_id IS NOT NULL THEN c.thread_id
           ELSE "1837a6a1-26e6-454c-b938-73d05cfbc92e"
         END AS thread_id, ch
    RETURN thread_id, min(ch.version) AS initial_version, max(ch.version) AS latest_version
    ORDER BY latest_version DESC
    """
    threads_list = []
    seen = set()
    try:
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
            with driver.session() as session:
                result = session.run(query)
                for record in result:
                    t_id = record["thread_id"]
                    if t_id and t_id != "null" and t_id not in seen:
                        seen.add(t_id)
                        threads_list.append({
                            "thread_id": t_id,
                            "initial_time": format_timestamp(record["initial_version"])
                        })
    except Exception as e:
        print(f"[!] Error scanning threads timeline indices: {e}")
    return threads_list


def dump_thread_with_official_api(target_thread_id):
    """
    Uses the official Neo4jSaver connection wrapper and serialization API
    to pull, decode, and render conversation milestones for a thread.
    """
    print(f"\n[+] Extracting official state history for Thread: '{target_thread_id}'")
    print("=" * 100)

    # Reconstruct snapshot states directly via channel aggregations grouped by step versions
    query = """
    MATCH (c:Checkpoint)-[:HAS_CHANNEL]->(ch:ChannelState)
    WHERE c.id STARTS WITH $thread_id 
       OR c.id CONTAINS $thread_id 
       OR c.thread_id = $thread_id
       OR $thread_id = "1837a6a1-26e6-454c-b938-73d05cfbc92e"
    WITH ch
    ORDER BY ch.version ASC
    WITH ch.version AS checkpoint_id, collect({channel: ch.channel, type: ch.type, blob: ch.blob}) AS shards
    RETURN checkpoint_id, shards
    ORDER BY checkpoint_id ASC
    """

    try:
        serializer = JsonPlusSerializer()
        counter = 0

        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
            with driver.session() as session:
                result = session.run(query, thread_id=target_thread_id)

                for record in result:
                    cp_id = record["checkpoint_id"]
                    shards = record["shards"]

                    source_node = "Unknown"
                    step_turn = "N/A"
                    messages = None

                    for shard in shards:
                        channel_name = shard.get("channel")
                        blob_payload = shard.get("blob")

                        if not blob_payload or blob_payload == "null":
                            continue

                        try:
                            # Safely unbox JSON strings versus raw byte payloads
                            if isinstance(blob_payload, str):
                                try:
                                    raw_bytes = base64.b64decode(blob_payload)
                                except Exception:
                                    raw_bytes = blob_payload.encode('utf-8')
                            else:
                                raw_bytes = bytes(blob_payload)

                            # Handle system metrics via the JSON layer
                            if channel_name == "__metadata__":
                                try:
                                    decoded_meta = json.loads(raw_bytes.decode('utf-8'))
                                except Exception:
                                    decoded_meta = serializer.loads_typed(("json", raw_bytes.decode('utf-8')))

                                if isinstance(decoded_meta, dict):
                                    source_node = decoded_meta.get("source", source_node)
                                    step_turn = decoded_meta.get("step", step_turn)

                            # Handle conversation streams via the msgpack layer
                            elif any(x in str(channel_name).lower() for x in ["messages", "history", "chat"]):
                                try:
                                    parsed_dict = json.loads(raw_bytes.decode('utf-8'))
                                    if isinstance(parsed_dict, dict) and "__serde_data__" in parsed_dict:
                                        msgpack_bytes = binascii.unhexlify(parsed_dict["__serde_data__"])
                                        messages = serializer.loads_typed(("msgpack", msgpack_bytes))
                                except Exception:
                                    messages = serializer.loads_typed(("json", raw_bytes.decode('utf-8')))

                        except Exception:
                            continue

                    if not messages:
                        continue

                    counter += 1
                    print(f"THREAD ID           : {target_thread_id}")
                    print(f"  [{counter}] Checkpoint ID : {cp_id}")
                    print(f"      Source Node   : {source_node}")
                    print(f"      Step Turn     : {step_turn}")
                    print("      Messages Transcript:")

                    if isinstance(messages, dict) and ("messages" in messages or "v" in messages):
                        messages = messages.get("messages", messages.get("v", []))

                    if not isinstance(messages, list):
                        messages = [messages]

                    for msg in messages:
                        print_formatted_message(msg)

                    print("-" * 50)

                if counter == 0:
                    print(f"\n[!] Complete. No conversational records were captured on thread '{target_thread_id}'.")
                print("=" * 100)

    except Exception as e:
        print(f"\n[X] Native API processing failure: {e}")


def main():
    if len(sys.argv) > 1:
        target_thread = sys.argv[1]
        dump_thread_with_official_api(target_thread)
        return

    print("Connecting to database to discover available thread indices...")
    threads = get_threads_timeline()

    if not threads:
        print("\n[!] No active conversation logs or thread shards were discovered in this database instance.")
        return

    print("\nAvailable Thread Sessions in Database:")
    print("-" * 75)
    for t in threads:
        print(f" • ID: {t['thread_id']:<36} | Started: {t['initial_time']}")
    print("-" * 75)

    latest_thread = threads[0]["thread_id"]
    print(f"\n[+] No thread ID specified. Automatically targeting newest thread session: '{latest_thread}'")
    dump_thread_with_official_api(latest_thread)


if __name__ == "__main__":
    main()