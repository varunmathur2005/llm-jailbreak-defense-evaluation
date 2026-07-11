"""Defense registry — add new defense classes here."""

from model_client import ModelClient
from defenses.base import BaseDefense
from defenses.none import NoneDefense
from defenses.self_reminder import SelfReminderDefense
from defenses.llama_guard import LlamaGuardDefense
from defenses.both import BothDefense
from defenses.perplexity import PerplexityDefense

REGISTRY: dict[str, type[BaseDefense]] = {
    "none": NoneDefense,
    "self_reminder": SelfReminderDefense,
    "llama_guard_input": LlamaGuardDefense,
    "llama_guard": LlamaGuardDefense,
    "llama_guard_both": LlamaGuardDefense,
    "both": BothDefense,
    "perplexity": PerplexityDefense,
}

_LLAMA_GUARD_MODES = {
    "llama_guard_input": "input",
    "llama_guard": "output",
    "llama_guard_both": "both",
}


def defense_requires_guard(name: str) -> bool:
    """Return whether a registered defense needs the configured Guard model."""
    return name in _LLAMA_GUARD_MODES or name == "both"


def get_defense(
    name: str,
    client: ModelClient,
    guard_client: ModelClient | None = None,
) -> BaseDefense:
    """Instantiate a defense by name."""
    if name not in REGISTRY:
        raise ValueError(f"Unknown defense '{name}'. Choose from: {list(REGISTRY)}")
    if defense_requires_guard(name):
        if guard_client is None:
            raise ValueError(f"The '{name}' defense requires a guard_client.")
    if name in _LLAMA_GUARD_MODES:
        return LlamaGuardDefense(client, guard_client, mode=_LLAMA_GUARD_MODES[name])
    if name == "both":
        return REGISTRY[name](client, guard_client)
    return REGISTRY[name](client)
