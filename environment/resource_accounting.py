from dataclasses import dataclass
import copy
from typing import Any, Callable, Optional


@dataclass
class ResourceBudget:
    token_limit: int = 200_000
    tool_limit: int = 50
    tool_timeout_seconds: float = 30.0
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
        response = self._llm.invoke(prompt, *args, **kwargs)
        tokens = self._budget.record_tokens(response_token_count(response))
        if self._event_callback is not None:
            self._event_callback(self._agent_name, tokens)
        return response

    def __getattr__(self, name):
        return getattr(self._llm, name)


def truncate_tool_result(result: Any, max_tokens: int = 2_000) -> Any:
    """Truncate textual tool output before it enters an agent prompt."""
    if isinstance(result, str):
        return " ".join(result.split()[:max_tokens])
    if isinstance(result, dict):
        truncated = copy.deepcopy(result)
        for key, value in truncated.items():
            if isinstance(value, str):
                truncated[key] = " ".join(value.split()[:max_tokens])
        return truncated
    if isinstance(result, list):
        return [truncate_tool_result(item, max_tokens) for item in result]
    return result
