"""Llama Guard 3 input/output filtering through Ollama's native chat format."""

from dataclasses import asdict, dataclass
import re
from time import perf_counter

import config
from defenses.base import BaseDefense
from model_client import ModelClient


BLOCKED_RESPONSE = config.LLAMA_GUARD_BLOCKED_RESPONSE
_VALID_MODES = {"input", "output", "both"}
_VALID_FAILURE_POLICIES = {"allow", "block", "raise"}
_CATEGORY_RE = re.compile(r"\bS(?:[1-9]|1[0-3])\b", re.IGNORECASE)
_ANY_CATEGORY_RE = re.compile(r"\bS\d+\b", re.IGNORECASE)


@dataclass(frozen=True)
class GuardDecision:
    """One serializable Llama Guard classification result."""

    label: str
    categories: list[str]
    raw_output: str
    parse_status: str
    error: str
    latency_s: float = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tokens_per_second: float = 0
    model: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def parse_guard_output(raw_output: str) -> GuardDecision:
    """Parse Llama Guard's exact safe/unsafe label and optional S1-S13 codes."""
    if not isinstance(raw_output, str):
        return _parse_error("", "Guard output must be a string")

    raw = raw_output
    stripped = raw.strip()
    if not stripped:
        return _parse_error(raw, "Guard output is empty")

    lines = stripped.splitlines()
    label = lines[0].strip().lower()
    if label not in {"safe", "unsafe"}:
        return _parse_error(raw, "First line must be exactly 'safe' or 'unsafe'")

    remainder = "\n".join(lines[1:]).strip()
    codes = [match.group(0).upper() for match in _ANY_CATEGORY_RE.finditer(remainder)]
    invalid = [code for code in codes if _CATEGORY_RE.fullmatch(code) is None]
    if invalid:
        return _parse_error(raw, f"Invalid Guard categories: {', '.join(invalid)}")
    if label == "safe" and remainder:
        return _parse_error(raw, "A safe result must not include categories or extra text")
    if label == "unsafe" and remainder:
        cleaned = _ANY_CATEGORY_RE.sub("", remainder)
        if cleaned.replace(",", "").strip():
            return _parse_error(raw, "Unsafe category lines contain unexpected text")

    categories = []
    for code in codes:
        if code not in categories:
            categories.append(code)
    return GuardDecision(label, categories, raw, "parsed", "")


def _parse_error(raw_output: str, error: str) -> GuardDecision:
    return GuardDecision("unknown", [], raw_output, "malformed", error)


def build_input_guard_conversation(messages: list[dict]) -> tuple[list[dict], str]:
    """Copy the conversation through its latest user turn for input moderation."""
    copied = _validated_copy(messages)
    latest_user = next(
        (index for index in range(len(copied) - 1, -1, -1) if copied[index]["role"] == "user"),
        None,
    )
    if latest_user is None:
        return [], "no_user_message"
    return copied[: latest_user + 1], "classifiable"


def build_output_guard_conversation(
    messages: list[dict], response: str
) -> tuple[list[dict], str]:
    """Copy a user-facing conversation and append exactly one target response."""
    if not isinstance(response, str):
        raise TypeError("target response must be a string")
    copied = _validated_copy(messages)
    if not any(message["role"] == "user" for message in copied):
        return [], "no_user_message"
    return copied + [{"role": "assistant", "content": response}], "classifiable"


def _validated_copy(messages: list[dict]) -> list[dict]:
    if not isinstance(messages, list):
        raise TypeError("messages must be a list")
    copied = []
    for message in messages:
        if not isinstance(message, dict):
            raise TypeError("each message must be a dictionary")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str):
            raise ValueError("each message must have a string role")
        if not isinstance(content, str):
            raise TypeError("each message must have string content")
        copied.append({"role": role, "content": content})
    return copied


class GuardClassifier:
    """Standalone native-conversation Llama Guard classifier."""

    def __init__(self, guard_client: ModelClient):
        self._guard = guard_client

    @property
    def model(self) -> str:
        return getattr(self._guard, "model", config.GUARD_MODEL)

    def classify_conversation(self, conversation: list[dict]) -> GuardDecision:
        start = perf_counter()
        try:
            raw, stats = self._guard.chat_with_stats(
                conversation, options={"temperature": 0}
            )
        except Exception as exc:
            return GuardDecision(
                "unknown",
                [],
                "",
                "provider_error",
                f"{type(exc).__name__}: {exc}",
                latency_s=perf_counter() - start,
                model=self.model,
            )

        parsed = parse_guard_output(raw)
        return GuardDecision(
            parsed.label,
            parsed.categories,
            parsed.raw_output,
            parsed.parse_status,
            parsed.error,
            latency_s=stats.get("latency_s", 0),
            prompt_tokens=stats.get("prompt_tokens", 0),
            completion_tokens=stats.get("completion_tokens", 0),
            total_tokens=stats.get("total_tokens", 0),
            tokens_per_second=stats.get("tokens_per_second", 0),
            model=self.model,
        )

    # Backward-compatible output-classification methods used by BothDefense.
    def classify(self, messages: list[dict], response: str) -> str:
        label, _ = self.classify_with_stats(messages, response)
        return label

    def classify_with_stats(self, messages: list[dict], response: str) -> tuple[str, dict]:
        conversation, status = build_output_guard_conversation(messages, response)
        if status != "classifiable":
            return "unknown", _empty_stats()
        decision = self.classify_conversation(conversation)
        return decision.label, _decision_stats(decision)


class LlamaGuardDefense(BaseDefense):
    """Filter target inputs, outputs, or both with native Ollama Llama Guard."""

    def __init__(
        self,
        client: ModelClient,
        guard_client: ModelClient,
        mode: str = config.LLAMA_GUARD_MODE,
        failure_policy: str = config.LLAMA_GUARD_FAILURE_POLICY,
        blocked_response: str = config.LLAMA_GUARD_BLOCKED_RESPONSE,
    ):
        super().__init__(client)
        if mode not in _VALID_MODES:
            raise ValueError("mode must be one of: input, output, both")
        if failure_policy not in _VALID_FAILURE_POLICIES:
            raise ValueError("failure_policy must be one of: allow, block, raise")
        self.mode = mode
        self.failure_policy = failure_policy
        self.blocked_response = blocked_response
        self._classifier = GuardClassifier(guard_client)

    def query(self, messages: list[dict]) -> tuple[str, dict]:
        input_decision = None
        output_decision = None

        if self.mode in {"input", "both"}:
            input_decision = self._classify_input(messages)
            action = self._action(input_decision, "input")
            if action == "block":
                return self.blocked_response, self._metadata(
                    input_decision, None, False, True, False, _empty_stats()
                )

        # Target errors deliberately remain outside Guard error handling.
        response, target_stats = self.client.chat_with_stats(messages)

        if self.mode in {"output", "both"}:
            output_decision = self._classify_output(messages, response)
            action = self._action(output_decision, "output")
            replaced = action == "block"
            final_response = self.blocked_response if replaced else response
        else:
            replaced = False
            final_response = response

        return final_response, self._metadata(
            input_decision,
            output_decision,
            True,
            replaced,
            replaced,
            target_stats,
            original_response=response,
        )

    def _classify_input(self, messages: list[dict]) -> GuardDecision:
        try:
            conversation, status = build_input_guard_conversation(messages)
        except Exception as exc:
            return _conversation_error(exc, self._classifier.model)
        if status != "classifiable":
            return GuardDecision(
                "unknown", [], "", "unclassifiable", "", model=self._classifier.model
            )
        return self._classifier.classify_conversation(conversation)

    def _classify_output(self, messages: list[dict], response: str) -> GuardDecision:
        try:
            conversation, status = build_output_guard_conversation(messages, response)
        except Exception as exc:
            return _conversation_error(exc, self._classifier.model)
        if status != "classifiable":
            return GuardDecision(
                "unknown", [], "", "unclassifiable", "", model=self._classifier.model
            )
        return self._classifier.classify_conversation(conversation)

    def _action(self, decision: GuardDecision, stage: str) -> str:
        if decision.label == "unsafe":
            return "block"
        if decision.label == "safe":
            return "allow"
        if self.failure_policy == "raise":
            detail = decision.error or decision.parse_status
            raise RuntimeError(f"Llama Guard {stage} classification failed: {detail}")
        return self.failure_policy

    def _metadata(
        self,
        input_decision: GuardDecision | None,
        output_decision: GuardDecision | None,
        target_invoked: bool,
        blocked: bool,
        response_replaced: bool,
        target_stats: dict,
        original_response: str = "",
    ) -> dict:
        decisions = [d for d in (input_decision, output_decision) if d is not None]
        guard_stats = _aggregate_stats(decisions)
        final_decision = output_decision or input_decision
        return {
            "defense_name": "llama_guard",
            "defense_stage": {"input": "input", "output": "output", "both": "input_output"}[
                self.mode
            ],
            "defense_blocked": blocked,
            "target_invoked": target_invoked,
            "response_replaced": response_replaced,
            "blocked_by_guard": blocked,
            "original_response": original_response,
            "guard_label": final_decision.label if final_decision else "",
            "llama_guard_mode": self.mode,
            "llama_guard_model": self._classifier.model,
            "llama_guard_failure_policy": self.failure_policy,
            "input_guard": input_decision.to_dict() if input_decision else None,
            "output_guard": output_decision.to_dict() if output_decision else None,
            "guard_total_latency_s": guard_stats["latency_s"],
            "defense_latency_s": guard_stats["latency_s"],
            "guard_stats": guard_stats,
            "target_stats": target_stats,
        }


def _conversation_error(exc: Exception, model: str) -> GuardDecision:
    return GuardDecision(
        "unknown", [], "", "conversation_error", f"{type(exc).__name__}: {exc}", model=model
    )


def _decision_stats(decision: GuardDecision) -> dict:
    return {
        "latency_s": decision.latency_s,
        "prompt_tokens": decision.prompt_tokens,
        "completion_tokens": decision.completion_tokens,
        "total_tokens": decision.total_tokens,
        "tokens_per_second": decision.tokens_per_second,
    }


def _aggregate_stats(decisions: list[GuardDecision]) -> dict:
    completion_tokens = sum(d.completion_tokens for d in decisions)
    if len(decisions) == 1:
        tokens_per_second = decisions[0].tokens_per_second
    elif completion_tokens and all(
        d.completion_tokens == 0 or d.tokens_per_second > 0 for d in decisions
    ):
        evaluation_time_s = sum(
            d.completion_tokens / d.tokens_per_second
            for d in decisions
            if d.completion_tokens
        )
        tokens_per_second = (
            completion_tokens / evaluation_time_s if evaluation_time_s else 0
        )
    else:
        # Per-stage rates remain available in input_guard/output_guard when an
        # aggregate cannot be reconstructed consistently.
        tokens_per_second = 0
    return {
        "latency_s": sum(d.latency_s for d in decisions),
        "prompt_tokens": sum(d.prompt_tokens for d in decisions),
        "completion_tokens": completion_tokens,
        "total_tokens": sum(d.total_tokens for d in decisions),
        "tokens_per_second": tokens_per_second,
    }


def _empty_stats() -> dict:
    return _aggregate_stats([])
