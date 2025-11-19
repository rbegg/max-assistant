import time
import httpx
import asyncio
import logging
from typing import Optional

from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


def create_llm_instance(
        model_name: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
) -> ChatOllama:
    """
    Synchronously initializes and returns a ChatOllama instance.
    """
    logger.info("=" * 50)
    logger.info("🚀 Initializing Ollama instance...")
    logger.info(f"   Model: {model_name}")
    logger.info(f"   Target: {base_url}")
    logger.info("=" * 50)

    llm = ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=temperature,
    )
    return llm


async def preload_model_async(
        llm: ChatOllama,
        ready_event: Optional[asyncio.Event] = None,
        keep_alive: str = "-1",
        max_retries: int = 10,
        retry_delay: int = 2
):
    """
    Asynchronously preloads a model in Ollama with retry logic.
    This is designed to be run as a background task. It will set the
    provided asyncio.Event upon completion or failure.
    """
    parser = StrOutputParser()
    chain = llm | parser

    retries = 0
    try:
        while retries < max_retries:
            try:
                logger.info(f"🔥 Sending async warm-up request to load '{llm.model}' into memory.")
                start_time = time.monotonic()

                await chain.ainvoke(
                    "Hi",
                    config=RunnableConfig(configurable={"keep_alive": keep_alive})
                )

                end_time = time.monotonic()
                duration = end_time - start_time

                logger.info(f"\n✅ Async warm-up complete! Model '{llm.model}' is ready.")
                logger.info(f"   Warm-up duration: {duration:.2f} seconds.")
                logger.info("-" * 50)
                return  # Warm-up successful

            except httpx.ConnectError:
                logging.warning("Connection to Ollama service failed during warm-up.")
                retries += 1
                if retries < max_retries:
                    logging.warning(f"Retrying warm-up in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    logging.error(
                        f"Exceeded maximum retries for warm-up of '{llm.model}'. The model may not be preloaded.")
                    return
            except Exception as e:
                logging.error(f"\n❌ FAILED TO WARM UP OLLAMA for model '{llm.model}'.")
                logging.error(f"   Error: {e}")
                return
    finally:
        if ready_event:
            ready_event.set()
