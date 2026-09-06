from dataclasses import dataclass
import copy
from pathlib import Path
from typing import Any, Callable, Optional

import yaml


@dataclass
class ResourceBudget:
    token_limit: int = 200_000
    tool_limit: int = 50
    tool_timeout_seconds: float = 30.0
    tool_result_tokens: int = 2_000
    memory_capacity: int = 100
    tokens_used: int = 0
    tools_used: int = 0
    timed_out_tools: int = 0

    def record_tokens(self, amount: int) -> int:
        amount = max(0, int(amount))
        if self.tokens_used + amount > self.token_limit:
            raise RuntimeError("LLM token budget exhausted.")
        self.tokens_used += amount
        return amount

    def record_tool(self) -> int:
        if self.tools_used >= self.tool_limit:
            raise RuntimeError("Tool invocation budget exhausted.")
        self.tools_used += 1
        return self.tools_used

    @property
    def token_budget_remaining(self) -> int:
        return max(0, self.token_limit - self.tokens_used)

    @property
    def tool_budget_remaining(self) -> int:
        return max(0, self.tool_limit - self.tools_used)

    def as_dict(self) -> dict:
        return {
            "tokens_used": self.tokens_used,
            "token_limit": self.token_limit,
            "tools_used": self.tools_used,
            "tool_limit": self.tool_limit,
            "timed_out_tools": self.timed_out_tools,
            "token_budget_remaining": self.token_budget_remaining,
            "tool_budget_remaining": self.tool_budget_remaining,
            "tool_timeout_seconds": self.tool_timeout_seconds,
            "tool_result_tokens": self.tool_result_tokens,
            "memory_capacity": self.memory_capacity,
        }

    def reset(self):
        self.tokens_used = 0
        self.tools_used = 0
        self.timed_out_tools = 0


def response_token_count(response: Any) -> int:
    """Read Ollama/LangChain usage metadata with a deterministic fallback."""
    for attribute in ("usage_metadata", "response_metadata"):
        metadata = getattr(response, attribute, None) or {}
        if not isinstance(metadata, dict):
            continue
        for key in ("total_tokens", "total_token_count"):
            if metadata.get(key) is not None:
                return int(metadata[key])
        token_usage = metadata.get("token_usage")
        if isinstance(token_usage, dict) and token_usage.get("total_tokens") is not None:
            return int(token_usage["total_tokens"])

    content = str(getattr(response, "content", "") or "")
    return len(content.split())


def load_resource_budget(config_path: str = "configs/config.yaml") -> ResourceBudget:
    with Path(config_path).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    values = config.get("budgets", {})
    return ResourceBudget(
        token_limit=int(values.get("token_limit", 200_000)),
        tool_limit=int(values.get("tool_limit", 50)),
        tool_timeout_seconds=float(values.get("tool_timeout_seconds", 30)),
        tool_result_tokens=int(values.get("tool_result_tokens", 2_000)),
        memory_capacity=int(values.get("memory_capacity", 100)),
    )


class TrackedLLM:
    def __init__(
        self,
        llm: Any,
        agent_name: str,
        budget: ResourceBudget,
        event_callback: Optional[Callable[..., None]] = None,
    ):
        self._llm = llm
        self._agent_name = agent_name
        self._budget = budget
        self._event_callback = event_callback

    def invoke(self, prompt: str, *args, **kwargs):
        if self._budget.token_budget_remaining <= 0:
            if self._event_callback is not None:
                self._event_callback(
                    self._agent_name, 0, "llm_rejected", "token_budget_exhausted"
                )
            raise RuntimeError("LLM token budget exhausted before invocation.")
        response = self._llm.invoke(prompt, *args, **kwargs)
        tokens = response_token_count(response)
        try:
            self._budget.record_tokens(tokens)
        except RuntimeError:
            if self._event_callback is not None:
                self._event_callback(
                    self._agent_name, tokens, "llm_rejected", "token_budget_exhausted"
                )
            raise
        if self._event_callback is not None:
            self._event_callback(self._agent_name, tokens, "llm_usage", "consumed")
        return response

    def __getattr__(self, name):
        return getattr(self._llm, name)


def truncate_tool_result(result: Any, max_tokens: int = 2_000) -> Any:
    """Truncate textual tool output before it enters an agent prompt."""
    remaining = [max(0, int(max_tokens))]

    def truncate(value):
        if isinstance(value, str):
            words = value.split()
            selected = words[:remaining[0]]
            remaining[0] -= len(selected)
            return " ".join(selected)
        if isinstance(value, dict):
            return {
                key: truncate(item)
                for key, item in copy.deepcopy(value).items()
            }
        if isinstance(value, list):
            return [truncate(item) for item in value]
        return value

    return truncate(result)
