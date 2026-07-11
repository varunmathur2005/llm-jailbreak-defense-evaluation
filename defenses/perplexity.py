"""Plain perplexity-threshold input defense using a local causal language model."""

from dataclasses import dataclass
from math import exp, isfinite
from threading import Lock
from time import perf_counter
from typing import Any

import config
from defenses.base import BaseDefense


def extract_latest_user_text(messages: list[dict]) -> str:
    """Return the latest user message without including other conversation roles.

    An empty conversation, no user turn, or empty user content produces an empty
    string, which the scorer treats as a neutral, unscorable input. A malformed
    user message is an error handled by the configured defense failure policy.
    """
    text, _ = _extract_latest_user_text_with_status(messages)
    return text


def _extract_latest_user_text_with_status(messages: list[dict]) -> tuple[str, str]:
    """Return the latest user text and an explicit input-status label."""
    if not isinstance(messages, list):
        raise TypeError("messages must be a list")

    for message in reversed(messages):
        if not isinstance(message, dict):
            raise TypeError("each message must be a dictionary")
        role = message.get("role")
        if not isinstance(role, str):
            raise ValueError("each message must have a string role")
        if role != "user":
            continue
        if "content" not in message:
            raise ValueError("the latest user message has no content")
        content = message["content"]
        if not isinstance(content, str):
            raise TypeError("the latest user message content must be a string")
        return content, "empty_user_message" if not content else "scored"
    return "", "no_user_message"


@dataclass(frozen=True)
class PerplexityResult:
    """Perplexity score and the token counts used to calculate it."""

    score: float | None
    token_count: int
    predicted_token_count: int
    device: str
    model_loaded_this_query: bool = False
    model_load_latency_ms: float = 0


class HuggingFacePerplexityScorer:
    """Lazily loaded, reusable causal-language-model perplexity scorer."""

    def __init__(self, model_name: str = "gpt2", device: str = "auto", stride: int = 256):
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: auto, cpu, cuda, mps")
        if stride <= 0:
            raise ValueError("stride must be greater than zero")
        self.model_name = model_name
        self.requested_device = device
        self.stride = stride
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._resolved_device = None
        self._load_lock = Lock()

    @property
    def resolved_device(self) -> str:
        if self._resolved_device is None:
            self._load()
        return self._resolved_device

    def _load(self) -> tuple[bool, float]:
        if self._model is not None:
            return False, 0

        with self._load_lock:
            if self._model is not None:
                return False, 0

            start = perf_counter()
            import torch  # Imported lazily so unrelated defenses stay lightweight.
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = self._resolve_device(torch, self.requested_device)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            model.to(device)
            model.eval()

            # Publish the model last so another caller can never observe a
            # non-None model with partially initialized supporting state.
            self._torch = torch
            self._tokenizer = tokenizer
            self._resolved_device = device
            self._model = model
            return True, (perf_counter() - start) * 1000

    @staticmethod
    def _resolve_device(torch_module: Any, requested: str) -> str:
        if requested == "auto":
            if torch_module.cuda.is_available():
                return "cuda"
            mps = getattr(torch_module.backends, "mps", None)
            if mps is not None and mps.is_available():
                return "mps"
            return "cpu"
        if requested == "cuda" and not torch_module.cuda.is_available():
            raise RuntimeError("PERPLEXITY_DEVICE is 'cuda', but CUDA is unavailable")
        if requested == "mps":
            mps = getattr(torch_module.backends, "mps", None)
            if mps is None or not mps.is_available():
                raise RuntimeError("PERPLEXITY_DEVICE is 'mps', but MPS is unavailable")
        return requested

    def score(self, text: str) -> PerplexityResult:
        """Calculate aggregate strided causal perplexity without truncation."""
        loaded_this_query, load_latency_ms = self._load()
        encoded = self._tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(self._resolved_device)
        token_count = int(input_ids.size(1))
        if token_count <= 1:
            # Causal perplexity has no next-token prediction for zero/one token.
            return PerplexityResult(
                None,
                token_count,
                0,
                self._resolved_device,
                loaded_this_query,
                load_latency_ms,
            )

        context_length = self._context_length()
        if self.stride > context_length - 1:
            raise ValueError(
                f"stride {self.stride} exceeds the maximum {context_length - 1} "
                f"for {self.model_name}'s context length {context_length}"
            )
        stride = self.stride
        total_negative_log_likelihood = 0.0
        predicted_token_count = 0
        previous_end = 0

        with self._torch.inference_mode():
            for begin in range(0, token_count, stride):
                end = min(begin + context_length, token_count)
                new_token_count = end - previous_end
                window_ids = input_ids[:, begin:end]
                labels = window_ids.clone()
                labels[:, :-new_token_count] = -100

                # Causal LM loss shifts labels left, so count only labels that
                # actually have a preceding token in this window.
                evaluated = int((labels[:, 1:] != -100).sum().item())
                if evaluated:
                    output = self._model(input_ids=window_ids, labels=labels)
                    loss = float(output.loss.item())
                    if not isfinite(loss):
                        raise ValueError("causal language model returned a non-finite loss")
                    total_negative_log_likelihood += loss * evaluated
                    predicted_token_count += evaluated

                previous_end = end
                if end == token_count:
                    break

        if predicted_token_count == 0:
            return PerplexityResult(
                None,
                token_count,
                0,
                self._resolved_device,
                loaded_this_query,
                load_latency_ms,
            )
        score = exp(total_negative_log_likelihood / predicted_token_count)
        if not isfinite(score):
            raise ValueError("calculated perplexity is not finite")
        return PerplexityResult(
            score,
            token_count,
            predicted_token_count,
            self._resolved_device,
            loaded_this_query,
            load_latency_ms,
        )

    def _context_length(self) -> int:
        candidates = [
            getattr(self._model.config, "max_position_embeddings", None),
            getattr(self._model.config, "n_positions", None),
            getattr(self._tokenizer, "model_max_length", None),
        ]
        usable = [int(value) for value in candidates if value and int(value) < 1_000_000]
        if not usable:
            raise ValueError(f"Cannot determine context length for {self.model_name}")
        context_length = min(usable)
        if context_length < 2:
            raise ValueError("causal language model context length must be at least two")
        return context_length


class PerplexityDefense(BaseDefense):
    """Block user prompts whose local GPT-style perplexity exceeds a threshold."""

    def __init__(
        self,
        client,
        scorer=None,
        threshold: float = config.PERPLEXITY_THRESHOLD,
        failure_policy: str = config.PERPLEXITY_FAILURE_POLICY,
        blocked_response: str = config.PERPLEXITY_BLOCKED_RESPONSE,
    ):
        super().__init__(client)
        if failure_policy not in {"allow", "block", "raise"}:
            raise ValueError("failure_policy must be one of: allow, block, raise")
        self.threshold = threshold
        self.failure_policy = failure_policy
        self.blocked_response = blocked_response
        self.scorer = scorer or HuggingFacePerplexityScorer(
            model_name=config.PERPLEXITY_MODEL,
            device=config.PERPLEXITY_DEVICE,
            stride=config.PERPLEXITY_STRIDE,
        )

    def query(self, messages: list[dict]) -> tuple[str, dict]:
        start = perf_counter()
        result = None
        error = ""
        input_status = "error"
        try:
            text, input_status = _extract_latest_user_text_with_status(messages)
            result = self.scorer.score(text)
            if result.predicted_token_count == 0:
                input_status = (
                    input_status
                    if input_status != "scored"
                    else "insufficient_tokens"
                )
        except Exception as exc:
            if self.failure_policy == "raise":
                raise RuntimeError(f"Perplexity input defense failed: {exc}") from exc
            error = f"{type(exc).__name__}: {exc}"

        latency_s = perf_counter() - start
        if result is not None:
            blocked = result.score is not None and result.score > self.threshold
        else:
            blocked = self.failure_policy == "block"
        metadata = self._metadata(result, blocked, latency_s, error, input_status)

        if blocked:
            metadata["target_stats"] = _empty_target_stats()
            return self.blocked_response, metadata

        response, target_stats = self.client.chat_with_stats(messages)
        metadata["target_invoked"] = True
        metadata["target_stats"] = target_stats
        return response, metadata

    def _metadata(
        self,
        result: PerplexityResult | None,
        blocked: bool,
        latency_s: float,
        error: str,
        input_status: str,
    ) -> dict:
        return {
            "defense_name": "perplexity",
            "defense_stage": "input",
            "defense_blocked": blocked,
            "defense_latency_s": latency_s,
            "target_invoked": False,
            "perplexity_status": input_status,
            "perplexity_measured": result is not None and result.score is not None,
            "perplexity_score": result.score if result is not None else None,
            "perplexity_threshold": self.threshold,
            "perplexity_token_count": result.token_count if result is not None else 0,
            "perplexity_predicted_token_count": (
                result.predicted_token_count if result is not None else 0
            ),
            "perplexity_model": self.scorer.model_name,
            "perplexity_device": (
                result.device
                if result is not None
                else getattr(self.scorer, "requested_device", "unknown")
            ),
            "perplexity_stride": self.scorer.stride,
            "perplexity_latency_ms": latency_s * 1000,
            "perplexity_model_loaded_this_query": (
                result.model_loaded_this_query if result is not None else False
            ),
            "perplexity_model_load_latency_ms": (
                result.model_load_latency_ms if result is not None else 0
            ),
            "perplexity_failure_policy": self.failure_policy,
            "perplexity_error": error,
        }


def _empty_target_stats() -> dict:
    return {
        "latency_s": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "tokens_per_second": 0,
    }
