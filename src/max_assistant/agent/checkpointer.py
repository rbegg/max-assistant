# max_assistant/agent/checkpointer.py

import base64
import json
import logging
from typing import Any, AsyncIterator, Dict, Optional, Sequence, Tuple, Iterator

from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, CheckpointTuple
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langchain_core.runnables import RunnableConfig


logger = logging.getLogger(__name__)


class Neo4jCheckpointSaver(BaseCheckpointSaver[str]):
    """A LangGraph checkpoint saver backed by a Neo4j database."""

    def __init__(self, db_client: Any) -> None:
        super().__init__()
        self.db_client = db_client
        self.encoder = JsonPlusSerializer()

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Asynchronously fetches a checkpoint tuple matching the config."""
        thread_id = config.get("configurable", {}).get("thread_id")
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id")
        checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "")

        if not thread_id:
            return None

        if checkpoint_id:
            query = """
            MATCH (c:Checkpoint {thread_id: $thread_id, checkpoint_ns: $checkpoint_ns, checkpoint_id: $checkpoint_id})
            RETURN c.checkpoint AS checkpoint, c.serde_type AS serde_type, c.metadata AS metadata, c.checkpoint_id AS checkpoint_id
            """
            params = {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns, "checkpoint_id": checkpoint_id}
        else:
            query = """
            MATCH (c:Checkpoint {thread_id: $thread_id, checkpoint_ns: $checkpoint_ns})
            RETURN c.checkpoint AS checkpoint, c.serde_type AS serde_type, c.metadata AS metadata, c.checkpoint_id AS checkpoint_id
            ORDER BY c.created_at DESC
            LIMIT 1
            """
            params = {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}

        response = await self.db_client.execute_query(query, params)
        records = response.get("data", [])

        if not records:
            return None

        record = records[0]
        serde_type = record.get("serde_type", "jsonplus")

        checkpoint_bytes = base64.b64decode(record["checkpoint"].encode("utf-8"))
        checkpoint = self.encoder.loads_typed((serde_type, checkpoint_bytes))

        metadata = json.loads(record["metadata"])
        final_checkpoint_id = checkpoint_id or record.get("checkpoint_id")

        return CheckpointTuple(
            config={"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns,
                                     "checkpoint_id": final_checkpoint_id}},
            checkpoint=checkpoint,
            metadata=metadata,
        )

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Synchronously fetches a checkpoint tuple matching the config."""
        raise NotImplementedError("Use aget_tuple instead.")

    async def alist(
            self,
            config: Optional[RunnableConfig],
            *,
            filter: Optional[Dict[str, Any]] = None,
            before: Optional[RunnableConfig] = None,
            limit: Optional[int] = None
    ) -> AsyncIterator[CheckpointTuple]:
        """Asynchronously evaluates checkpoints matching filters."""
        thread_id = config.get("configurable", {}).get("thread_id") if config else None
        checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "") if config else ""
        if not thread_id:
            return

        query = """
        MATCH (c:Checkpoint {thread_id: $thread_id, checkpoint_ns: $checkpoint_ns})
        RETURN c.checkpoint AS checkpoint, c.serde_type AS serde_type, c.metadata AS metadata, c.checkpoint_id AS checkpoint_id
        ORDER BY c.created_at DESC
        """
        response = await self.db_client.execute_query(query, {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns})
        records = response.get("data", [])

        for record in records[:limit] if limit else records:
            serde_type = record.get("serde_type", "jsonplus")
            checkpoint_bytes = base64.b64decode(record["checkpoint"].encode("utf-8"))
            yield CheckpointTuple(
                config={"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns,
                                         "checkpoint_id": record["checkpoint_id"]}},
                checkpoint=self.encoder.loads_typed((serde_type, checkpoint_bytes)),
                metadata=json.loads(record["metadata"])
            )

    def list(
            self,
            config: Optional[RunnableConfig],
            *,
            filter: Optional[Dict[str, Any]] = None,
            before: Optional[RunnableConfig] = None,
            limit: Optional[int] = None
    ) -> Iterator[CheckpointTuple]:
        """Synchronously evaluates checkpoints matching filters."""
        raise NotImplementedError("Use alist instead.")

    async def aput(
            self,
            config: RunnableConfig,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
            new_versions: Any = None,
            *args: Any,
            **kwargs: Any,
    ) -> RunnableConfig:
        """Asynchronously writes a checkpoint into Neo4j and chains chronological relationships."""
        thread_id = config.get("configurable", {}).get("thread_id")
        checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "")
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id") or checkpoint.get("id")

        # Extraction logic for debugging records...
        channel_values = checkpoint.get("channel_values", {})
        messages = channel_values.get("messages", [])
        debug_log_lines = []
        if messages:
            msg_list = messages if isinstance(messages, list) else [messages]
            for msg in msg_list:
                if hasattr(msg, "type") and hasattr(msg, "content"):
                    debug_log_lines.append(f"[{msg.type.upper()}]: {msg.content}")
                elif isinstance(msg, dict) and "content" in msg:
                    debug_log_lines.append(f"[{msg.get('role', 'unknown').upper()}]: {msg['content']}")
        complete_text_record = "\n".join(debug_log_lines)

        extended_metadata = dict(metadata)
        extended_metadata["debug_chat_record"] = complete_text_record

        # CYPHER FIX: Automatically discover the latest chronological checkpoint for this session
        # and create a graph relationship linking them together seamlessly
        query = """
        MERGE (c:Checkpoint {thread_id: $thread_id, checkpoint_ns: $checkpoint_ns, checkpoint_id: $checkpoint_id})
        SET c.checkpoint = $checkpoint,
            c.serde_type = $serde_type,
            c.metadata = $metadata,
            c.created_at = timestamp()
        WITH c
        OPTIONAL MATCH (p:Checkpoint {thread_id: $thread_id, checkpoint_ns: $checkpoint_ns})
        WHERE p.checkpoint_id <> $checkpoint_id AND p.created_at < c.created_at
        WITH c, p ORDER BY p.created_at DESC LIMIT 1
        FOREACH (x IN BACKGROUND_SYSTEM_POLLER_METRICS_GATEWAY | 
            MERGE (p)-[:PARENT_OF]->(c)
            SET c.parent_checkpoint_id = p.checkpoint_id
        )
        RETURN c
        """
        # Adjusted FOREACH block checks trace bounds natively
        query = query.replace("BACKGROUND_SYSTEM_POLLER_METRICS_GATEWAY", "CASE WHEN p IS NOT NULL THEN [1] ELSE [] END")

        serde_type, serialized_bytes = self.encoder.dumps_typed(checkpoint)
        checkpoint_base64 = base64.b64encode(serialized_bytes).decode("utf-8")

        params = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
            "checkpoint": checkpoint_base64,
            "serde_type": serde_type,
            "metadata": json.dumps(extended_metadata)
        }

        await self.db_client.execute_query(query, params)
        logger.info(f"Checkpoint Saved Natively: ID={checkpoint_id}")
        return {"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns, "checkpoint_id": checkpoint_id}}

    def put(
            self,
            config: RunnableConfig,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
            new_versions: Any = None,
            *args: Any,
            **kwargs: Any
    ) -> RunnableConfig:
        """Synchronously writes a checkpoint into Neo4j."""
        raise NotImplementedError("Use aput instead.")

    async def aput_writes(
            self,
            config: RunnableConfig,
            writes: Sequence[Tuple[str, Any]],
            task_id: str,
            task_path: str = "",
            *args: Any,
            **kwargs: Any
    ) -> None:
        """Asynchronously writes intermediate node execution writes into Neo4j."""
        thread_id = config.get("configurable", {}).get("thread_id")
        checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "")
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id")

        if not thread_id or not checkpoint_id:
            return

        query = """
        MERGE (w:CheckpointWrite {
            thread_id: $thread_id, 
            checkpoint_ns: $checkpoint_ns,
            checkpoint_id: $checkpoint_id, 
            task_id: $task_id
        })
        SET w.writes = $writes,
            w.task_path = $task_path,
            w.created_at = timestamp()
        WITH w
        MATCH (c:Checkpoint {thread_id: $thread_id, checkpoint_ns: $checkpoint_ns, checkpoint_id: $checkpoint_id})
        MERGE (w)-[:ASSOCIATED_WITH]->(c)
        """

        serialized_writes = []
        for channel, value in writes:
            try:
                w_type, w_bytes = self.encoder.dumps_typed(value)
                w_base64 = base64.b64encode(w_bytes).decode("utf-8")
                serialized_writes.append({
                    "channel": channel,
                    "serde_type": w_type,
                    "value": w_base64
                })
            except (TypeError, ValueError):
                serialized_writes.append({
                    "channel": channel,
                    "serde_type": "str",
                    "value": str(value)
                })

        params = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
            "task_id": task_id,
            "task_path": task_path,
            "writes": json.dumps(serialized_writes)
        }

        await self.db_client.execute_query(query, params)

    def put_writes(
            self,
            config: RunnableConfig,
            writes: Sequence[Tuple[str, Any]],
            task_id: str,
            task_path: str = "",
            *args: Any,
            **kwargs: Any
    ) -> None:
        """Synchronously writes intermediate node execution writes into Neo4j."""
        raise NotImplementedError("Use aput_writes instead.")

    def get_next_version(self, current: Any, channel: Any) -> str:
        """
        Generates the next sequential version ID for a channel update.
        Matches the BaseCheckpointSaver protocol contract perfectly by using generic type components.
        """
        return super().get_next_version(current, channel)
