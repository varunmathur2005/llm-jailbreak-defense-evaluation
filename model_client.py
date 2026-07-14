"""Thin wrapper around the Ollama Python client."""

from time import perf_counter

import ollama


class ModelClient:
    """Synchronous chat client for a single Ollama model."""

    def __init__(self, model: str, base_url: str):
        self.model = model
        self._client = ollama.Client(host=base_url)

    def chat(self, messages: list[dict]) -> str:
        """Send a messages list and return the assistant reply text."""
        text, _ = self.chat_with_stats(messages)
        return text

    def chat_with_stats(
        self, messages: list[dict], options: dict | None = None
    ) -> tuple[str, dict]:
        """Send a messages list and return (assistant reply text, timing/token stats)."""
        start = perf_counter()
        kwargs = {"model": self.model, "messages": messages}
        if options is not None:
            kwargs["options"] = options
        response = self._client.chat(**kwargs)
        latency_s = perf_counter() - start

        prompt_tokens = getattr(response, "prompt_eval_count", 0) or 0
        completion_tokens = getattr(response, "eval_count", 0) or 0
        total_tokens = prompt_tokens + completion_tokens
        eval_duration_ns = getattr(response, "eval_duration", 0) or 0
        eval_duration_s = eval_duration_ns / 1_000_000_000 if eval_duration_ns else 0

        stats = {
            "latency_s": latency_s,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "tokens_per_second": completion_tokens / eval_duration_s if eval_duration_s else 0,
        }
        return response.message.content, stats

    def list_local_models(self) -> list[str]:
        """Return the names of all models currently pulled in Ollama."""
        result = self._client.list()
        return [m.model for m in result.models]
