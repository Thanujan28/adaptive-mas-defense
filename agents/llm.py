import os

from langchain_ollama import ChatOllama


def get_llm(temperature: float | None = None) -> ChatOllama:
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
        temperature=(
            temperature
            if temperature is not None
            else float(os.getenv("OLLAMA_TEMPERATURE", "0"))
        ),
        num_gpu=int(os.getenv("OLLAMA_NUM_GPU", "0")),
        num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "4096")),
        num_thread=int(os.getenv("OLLAMA_NUM_THREAD", "0")) or None,
        keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "5m"),
        top_p=1.0,
    )