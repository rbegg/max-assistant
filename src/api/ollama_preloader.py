import time
import httpx
import asyncio
import logging
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser


async def warm_up_ollama_async(
        model_name: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        keep_alive: str = "-1",
        max_retries: int = 10,
        retry_delay: int = 2
) -> ChatOllama | None:
    """
    Asynchronously initializes and preloads a model in Ollama with retry logic.
    """
    logging.info("=" * 50)
    logging.info("🚀 Initializing Ollama (Async)...")
    logging.info(f"   Model: {model_name}")
    logging.info(f"   Target: {base_url}")
    logging.info("=" * 50)

    retries = 0
    while retries < max_retries:
        try:
            llm = ChatOllama(
                model=model_name,
                base_url=base_url,
                temperature=temperature,
            )
            parser = StrOutputParser()
            chain = llm | parser

            logging.info(f"🔥 Sending async warm-up request to load '{model_name}' into memory.")

            start_time = time.monotonic()

            await chain.ainvoke(
                "Hi",
                config={"configurable": {"keep_alive": keep_alive}}
            )

            end_time = time.monotonic()
            duration = end_time - start_time

            logging.info(f"\n✅ Async warm-up complete! Model is ready.")
            logging.info(f"   Warm-up duration: {duration:.2f} seconds.")
            logging.info("-" * 50)

            return llm

        except httpx.ConnectError:
            logging.warning(f"Connection to Ollama service failed or was lost.")
            retries += 1
            if retries < max_retries:
                logging.warning(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logging.error("Exceeded maximum retries. Exiting.")
                return None
        except Exception as e:
            logging.error("\n❌ FAILED TO WARM UP OLLAMA (Async).")
            logging.error(f"   Error: {e}")
            return None

    return None