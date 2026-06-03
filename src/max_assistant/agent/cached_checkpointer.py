import logging
from typing import Any, AsyncIterator, Dict, Optional, Iterator
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple, Checkpoint, CheckpointMetadata

logger = logging.getLogger(__name__)


class CachedCheckpointSaver(BaseCheckpointSaver[str]):
    """
    A caching wrapper for BaseCheckpointSaver implementations.
    Optimized for immediately serving the latest checkpoint of a thread.
    """

    def __init__(self, base_saver: BaseCheckpointSaver[str]) -> None:
        super().__init__()
        self.base_saver = base_saver
        # Cache Key: (thread_id, checkpoint_ns) -> Value: CheckpointTuple
        self._latest_cache: Dict[tuple[str, str], CheckpointTuple] = {}

    @staticmethod
    def _get_cache_key(config: RunnableConfig) -> Optional[tuple[str, str]]:
        """Extracts a unique composite cache key from the configuration layer."""
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id")
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        if not thread_id:
            return None
        return str(thread_id), str(checkpoint_ns)

    # --- Safe Attribute Delegation Fallbacks ---

    @property
    def encoder(self) -> Any:
        """Delegates encoder lookup safely to the underlying native saver."""
        return getattr(self.base_saver, "encoder", None)

    @property
    def db_client(self) -> Any:
        """Delegates database client access safely to the underlying native saver."""
        return getattr(self.base_saver, "db_client", None)

    # --- Asynchronous Read Paths ---

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id")
        cache_key = self._get_cache_key(config)

        # Optimization: If searching for the latest record (no specific checkpoint_id requested)
        if cache_key and not checkpoint_id:
            if cache_key in self._latest_cache:
                logger.debug(f"Cache Hit (Async): Serving latest checkpoint for thread {cache_key[0]}")
                return self._latest_cache[cache_key]

        # Cache Miss or Specific History Lookup: Fall back to the real database
        checkpoint_tuple = await self.base_saver.aget_tuple(config)

        # Warm the cache if we fetched the absolute latest historical node
        if cache_key and not checkpoint_id and checkpoint_tuple:
            self._latest_cache[cache_key] = checkpoint_tuple

        return checkpoint_tuple

    async def alist(
            self,
            config: Optional[RunnableConfig],
            *,
            query_filter: Optional[Dict[str, Any]] = None,
            before: Optional[RunnableConfig] = None,
            limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        # History lists bypass the latest-record cache.
        # Fixed: Aligned parameter argument naming contract to avoid unexpected arguments.
        async for checkpoint in self.base_saver.alist(
            config, query_filter=query_filter, before=before, limit=limit
        ):
            yield checkpoint

    # --- Asynchronous Write Paths ---

    async def aput(
            self,
            config: RunnableConfig,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
            new_versions: Any = None,
            *args: Any,
            **kwargs: Any,
        ) -> RunnableConfig:
        # Fix Cache Bottleneck: Update local cache ahead of heavy persistent I/O wait boundaries
        cache_key = self._get_cache_key(config)
        if cache_key and not config.get("configurable", {}).get("checkpoint_id"):
            self._latest_cache[cache_key] = CheckpointTuple(
                config=config,
                checkpoint=checkpoint,
                metadata=metadata,
            )
            logger.debug(f"Cache Warmed Pre-emptively: Updated latest state for thread {cache_key[0]}")

        # Commit changes downstream to the persistent Neo4j Database
        return await self.base_saver.aput(config, checkpoint, metadata, new_versions, *args, **kwargs)

    async def aput_writes(
            self,
            config: RunnableConfig,
            writes: Any,
            task_id: str,
            task_path: str = "",
            *args: Any,
            **kwargs: Any,
    ) -> None:
        await self.base_saver.aput_writes(config, writes, task_id, task_path, *args, **kwargs)

    # --- Synchronous Contract Fallbacks ---

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id")
        cache_key = self._get_cache_key(config)

        if cache_key and not checkpoint_id:
            if cache_key in self._latest_cache:
                logger.debug(f"Cache Hit (Sync): Serving latest checkpoint for thread {cache_key[0]}")
                return self._latest_cache[cache_key]

        checkpoint_tuple = self.base_saver.get_tuple(config)
        if cache_key and not checkpoint_id and checkpoint_tuple:
            self._latest_cache[cache_key] = checkpoint_tuple
        return checkpoint_tuple

    def list(
            self,
            config: Optional[RunnableConfig],
            *,
            query_filter: Optional[Dict[str, Any]] = None,
            before: Optional[RunnableConfig] = None,
            limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        # Fixed: Aligned query parameter naming tokens
        return self.base_saver.list(config, query_filter=query_filter, before=before, limit=limit)

    def put(
            self,
            config: RunnableConfig,
            checkpoint: Checkpoint,
            metadata: CheckpointMetadata,
            new_versions: Any = None,
            *args: Any,
            **kwargs: Any,
    ) -> RunnableConfig:
        new_config = self.base_saver.put(config, checkpoint, metadata, new_versions, *args, **kwargs)
        cache_key = self._get_cache_key(new_config)
        if cache_key:
            self._latest_cache[cache_key] = CheckpointTuple(
                config=new_config,
                checkpoint=checkpoint,
                metadata=metadata,
            )
        return new_config

    def put_writes(
            self,
            config: RunnableConfig,
            writes: Any,
            task_id: str,
            task_path: str = "",
            *args: Any,
            **kwargs: Any,
    ) -> None:
        self.base_saver.put_writes(config, writes, task_id, task_path, *args, **kwargs)

    def get_next_version(self, current: Any, channel: Any) -> str:
        return self.base_saver.get_next_version(current, channel)