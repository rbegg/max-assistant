import os
import json
import base64
import re
from datetime import datetime
from neo4j import GraphDatabase

# Safe conditional check for LangGraph framework dependencies
try:
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False

# Configuration: Environment variables with safe fallback routes
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


def format_timestamp(ts):
    """Converts Neo4j ms-based system epoch timestamps to human-readable strings."""
    if not ts:
        return "N/A"
    try:
        return datetime.fromtimestamp(ts / 1000.0).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    except Exception:
        return str(ts)


def parse_corrupted_binary_string(text):
    """
    Intelligently analyzes compressed or separator-stripped raw message blobs,
    isolates key fields (content, tool calls, tool responses, metadata),
    restores structured spacing, and formats them cleanly.
    """
    parsed_fields = {}

    # Isolate implied message role mapping explicitly for fallback handling
    role = "Unknown"
    if "messages.human" in text or "[Human]" in text:
        role = "Human"
    elif "messages.ai" in text or "[AI]" in text:
        role = "AI"
    elif "messages.system" in text or "[System]" in text:
        role = "System"
    elif "messages.tool" in text or "[Tool]" in text:
        role = "Tool"

    # 1. Isolate the main textual conversation content / tool response payload
    # Handles normal fields as well as compressed tool-execution structures
    content_match = re.search(r'content(.*?)additional_kwargs', text)
    if content_match:
        content_raw = content_match.group(1).strip()
        # Check if the text inside the content is a stringified JSON block (common for tools)
        if "{" in content_raw and "}" in content_raw:
            try:
                start_idx = content_raw.find("{")
                end_idx = content_raw.rfind("}") + 1
                json_data = json.loads(content_raw[start_idx:end_idx])
                parsed_fields["Payload [Data]"] = json_data
            except Exception:
                parsed_fields["Content"] = content_raw
        else:
            parsed_fields["Content"] = content_raw
    else:
        # Fallback if additional_kwargs boundary is squashed
        content_fallback = re.search(r'content(.*)', text)
        if content_fallback and not any(
                k in content_fallback.group(1) for k in ["tool_calls", "response_metadata", "tool_input"]):
            parsed_fields["Content"] = content_fallback.group(1).strip()

    # 2. Extract explicit Tool Invocations (AI requesting a tool call)
    if "tool_calls" in text:
        tool_info = {}
        name_match = re.search(r'name([a-zA-Z0-9_\-]+)args', text)
        if name_match:
            tool_info["Target Tool"] = name_match.group(1).strip()

        args_match = re.search(r'args(.*?)id\$', text)
        if args_match:
            args_raw = args_match.group(1).strip()
            for common_key in ["first_name", "last_name", "query", "id", "user_id"]:
                args_raw = re.sub(rf'({common_key})', r'\n• \1: ', args_raw)
            tool_info["Arguments"] = args_raw.strip()

        if tool_info:
            parsed_fields["Action [Tool Call]"] = tool_info

    # 3. Extract tool execution properties (Tool execution response details)
    tool_id_match = re.search(r'tool_call_id([a-zA-Z0-9_\-]+)', text)
    if tool_id_match:
        parsed_fields["Origin Tool Call ID"] = tool_id_match.group(1).strip()

    # 4. Extract infrastructure metadata context
    model_match = re.search(r'model_name([a-zA-Z0-9_\-:]+)', text)
    if model_match:
        parsed_fields["Engine Model"] = model_match.group(1).strip()

    time_match = re.search(r'created_at(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)', text)
    if time_match:
        parsed_fields["Generated At"] = time_match.group(1).strip()

    if parsed_fields:
        return role, parsed_fields

    return role, None


def format_message(msg):
    """
    Extracts, cleans, and pretty-prints LangChain/LangGraph conversation message items.
    Safely converts dictionaries, objects, and raw msgpack ExtType structures into cleanly
    indented JSON representations with human-readable type tokens.
    """
    try:
        # Check if the object is an unparsed msgpack custom ExtType wrapper
        if hasattr(msg, "code") and hasattr(msg, "data"):
            try:
                # Isolate the data block, strip unprintable framing bytes, and string-decode
                raw_str = msg.data.decode('utf-8', errors='ignore')
                cleaned_text = "".join(ch for ch in raw_str if ch.isprintable() or ch in "\n\r\t").strip()

                # Attempt to structure the scrambled binary string block
                role, structured_data = parse_corrupted_binary_string(cleaned_text)
                role_token = f"[{role}]"

                if structured_data:
                    pretty_output = json.dumps(structured_data, indent=4)
                    formatted = f"      {role_token}:\n"
                    for line in pretty_output.splitlines():
                        formatted += f"          {line}\n"
                    return formatted.rstrip()

                return f"      [RAW MSG DATA]: {cleaned_text}"
            except Exception:
                pass
            return f"      [UNPARSED BINARY STATE OBJECT] (Type Code: {msg.code})"

        # Handle standard dictionary message formatting
        if isinstance(msg, dict):
            msg_type = msg.get("type", msg.get("id", ["Message"])[-1] if isinstance(msg.get("id"), list) else "Message")
            msg_type_str = str(msg_type).lower()

            if "human" in msg_type_str:
                role_token = "[Human]"
            elif "ai" in msg_type_str:
                role_token = "[AI]"
            elif "system" in msg_type_str:
                role_token = "[System]"
            elif "tool" in msg_type_str:
                role_token = "[Tool]"
            else:
                role_token = f"[{msg_type}]"

            pretty_msg = json.dumps(msg, indent=4)
            formatted = f"      {role_token}:\n"
            for line in pretty_msg.splitlines():
                formatted += f"          {line}\n"
            return formatted.rstrip()

        else:
            # Fallback formatting for raw framework class instances
            msg_type = getattr(msg, "type", msg.__class__.__name__)
            content = getattr(msg, "content", "")

            msg_type_str = str(msg_type).lower()
            if "human" in msg_type_str:
                role_token = "[Human]"
            elif "ai" in msg_type_str:
                role_token = "[AI]"
            elif "system" in msg_type_str:
                role_token = "[System]"
            elif "tool" in msg_type_str:
                role_token = "[Tool]"
            else:
                role_token = f"[{msg_type}]"

            if isinstance(content, (dict, list)):
                content_str = json.dumps(content, indent=4)
            else:
                content_str = str(content)

            if "\n" in content_str:
                formatted = f"      {role_token}:\n"
                for line in content_str.splitlines():
                    formatted += f"          {line}\n"
                return formatted.rstrip()
            else:
                return f"      {role_token}: {content_str}"

    except Exception as e:
        return f"      [UNPARSABLE MESSAGE BLOB]: {str(msg)}"


def dump_checkpoints_by_thread():
    query = """
    MATCH (c:Checkpoint)
    WITH c.thread_id AS thread_id, min(c.created_at) AS first_node_creation
    ORDER BY first_node_creation ASC
    MATCH (ch:Checkpoint {thread_id: thread_id})
    WITH thread_id, first_node_creation, ch
    ORDER BY ch.created_at ASC
    RETURN thread_id, first_node_creation, collect({
        checkpoint_id: ch.checkpoint_id,
        checkpoint_ns: ch.checkpoint_ns,
        created_at: ch.created_at,
        parent_checkpoint_id: ch.parent_checkpoint_id,
        metadata: ch.metadata,
        serde_type: ch.serde_type,
        checkpoint: ch.checkpoint
    }) AS checkpoints
    """

    print(f"Connecting to Neo4j instance at: {NEO4J_URI}")

    try:
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver:
            with driver.session() as session:
                result = session.run(query)
                records = list(result)

                if not records:
                    print("\n[!] No checkpoint records found in the target database.")
                    return

                print(f"\n[+] Found {len(records)} unique conversation thread(s). Sorting chronologically...\n")
                print("=" * 100)

                for record in records:
                    thread_id = record["thread_id"]
                    first_node_creation = record["first_node_creation"]
                    checkpoints = record["checkpoints"]

                    print(f"THREAD ID                    : {thread_id}")
                    print(f"Thread Initialized At        : {format_timestamp(first_node_creation)}")
                    print(f"Total Checkpoints in History : {len(checkpoints)}")
                    print("-" * 100)

                    for idx, cp in enumerate(checkpoints, 1):
                        cp_id = cp.get("checkpoint_id", "N/A")
                        cp_ns = cp.get("checkpoint_ns", "N/A")
                        created_at = cp.get("created_at")
                        parent_id = cp.get("parent_checkpoint_id", "None (Root)")
                        serde_type = cp.get("serde_type", "N/A")
                        metadata_raw = cp.get("metadata", "{}")
                        checkpoint_raw = cp.get("checkpoint")

                        print(f"  [{idx}] Checkpoint ID : {cp_id}")
                        print(f"      Namespace     : {cp_ns}")
                        print(f"      Created At    : {format_timestamp(created_at)}")
                        print(f"      Parent ID     : {parent_id}")
                        print(f"      Serde Type    : {serde_type}")

                        try:
                            if isinstance(metadata_raw, str):
                                metadata = json.loads(metadata_raw)
                            else:
                                metadata = metadata_raw
                            print("      Metadata      :")
                            metadata_str = json.dumps(metadata, indent=4)
                            for line in metadata_str.splitlines():
                                print(f"          {line}")
                        except Exception:
                            print(f"      Metadata (Raw): {metadata_raw}")

                        if checkpoint_raw:
                            print("      Messages Transcript:")
                            try:
                                if isinstance(checkpoint_raw, str):
                                    try:
                                        checkpoint_bytes = base64.b64decode(checkpoint_raw)
                                    except Exception:
                                        checkpoint_bytes = checkpoint_raw.encode('utf-8')
                                else:
                                    checkpoint_bytes = checkpoint_raw

                                checkpoint_data = None

                                if serde_type == "msgpack":
                                    import msgpack
                                    try:
                                        checkpoint_data = msgpack.loads(checkpoint_bytes, raw=False)
                                    except Exception:
                                        checkpoint_data = msgpack.loads(checkpoint_bytes, use_list=True, raw=True)
                                else:
                                    if HAS_LANGGRAPH:
                                        serializer = JsonPlusSerializer()
                                        payload_str = checkpoint_bytes.decode('utf-8') if isinstance(checkpoint_bytes,
                                                                                                     bytes) else checkpoint_bytes
                                        checkpoint_data = serializer.loads(payload_str)

                                if checkpoint_data and isinstance(checkpoint_data, dict):
                                    channel_values = checkpoint_data.get("channel_values", checkpoint_data.get("v", {}))
                                    messages = channel_values.get("messages", [])

                                    if not messages:
                                        for k, v in channel_values.items():
                                            if "message" in str(k).lower() and isinstance(v, list):
                                                messages = v
                                                break

                                    if messages:
                                        if not isinstance(messages, list):
                                            messages = [messages]
                                        for msg in messages:
                                            print(format_message(msg))
                                    else:
                                        print("          (No active messages found in this state channel)")
                                else:
                                    try:
                                        fallback_str = checkpoint_bytes.decode('utf-8', errors='ignore')
                                        cleaned_fallback = "".join(
                                            ch for ch in fallback_str if ch.isprintable() or ch in "\n\r\t").strip()
                                        role, structured_fallback = parse_corrupted_binary_string(cleaned_fallback)
                                        if structured_fallback:
                                            role_token = f"[{role}]"
                                            print(f"      {role_token}:")
                                            pretty_fallback = json.dumps(structured_fallback, indent=4)
                                            for line in pretty_fallback.splitlines():
                                                print(f"          {line}")
                                        elif len(cleaned_fallback) > 20:
                                            print(f"          [RAW FALLBACK TEXT]: {cleaned_fallback}")
                                        else:
                                            print(
                                                "          (Checkpoint state data parsed but contained no text blocks)")
                                    except Exception:
                                        print(
                                            "          (Checkpoint state payload could not be parsed as a dictionary structural root)")
                            except Exception as e:
                                print(f"          [!] Serialization unpack error: {e}")
                        else:
                            print("      Messages Transcript: (Empty state snapshot)")
                        print()

                    print("=" * 100)

    except Exception as e:
        print(f"\n[X] Database connection or execution error: {e}")


if __name__ == "__main__":
    dump_checkpoints_by_thread()