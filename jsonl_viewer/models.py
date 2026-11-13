"""Data models for parsed Claude Code JSONL sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# Standard Claude pricing per 1M tokens (USD)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "opus": {
        "input": 15.00,
        "output": 75.00,
        "cache_write": 18.75,
        "cache_read": 1.50,
    },
    "sonnet": {
        "input": 3.00,
        "output": 15.00,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "haiku": {
        "input": 0.80,
        "output": 4.00,
        "cache_write": 1.00,
        "cache_read": 0.08,
    },
}


def get_model_tier(model_name: str | None) -> str:
    """Resolve model string (e.g. 'claude-sonnet-4-20250514') to pricing tier."""
    if not model_name:
        return "sonnet"
    low = model_name.lower()
    if "opus" in low:
        return "opus"
    if "haiku" in low:
        return "haiku"
    return "sonnet"


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    model: str = ""

    @property
    def total_input_context(self) -> int:
        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    @property
    def total_tokens(self) -> int:
        return self.total_input_context + self.output_tokens

    @property
    def cache_hit_rate(self) -> float:
        total_in = self.total_input_context
        if total_in == 0:
            return 0.0
        return (self.cache_read_input_tokens / total_in) * 100.0

    def estimated_cost_usd(self, default_model: str = "sonnet") -> float:
        tier = get_model_tier(self.model or default_model)
        rates = MODEL_PRICING[tier]
        return (
            self.input_tokens * rates["input"]
            + self.output_tokens * rates["output"]
            + self.cache_creation_input_tokens * rates["cache_write"]
            + self.cache_read_input_tokens * rates["cache_read"]
        ) / 1_000_000.0

    def add(self, other: TokenUsage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        if not self.model and other.model:
            self.model = other.model


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime | None = None
    result_timestamp: datetime | None = None
    result_content: str = ""
    is_error: bool = False
    has_result: bool = False
    is_sidechain: bool = False
    extra_result_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float | None:
        if self.timestamp and self.result_timestamp:
            delta = (self.result_timestamp - self.timestamp).total_seconds() * 1000.0
            return max(0.0, delta)
        # Fallback if extra_result_meta has durationMs
        if "durationMs" in self.extra_result_meta:
            try:
                return float(self.extra_result_meta["durationMs"])
            except (TypeError, ValueError):
                pass
        if "totalDurationMs" in self.extra_result_meta:
            try:
                return float(self.extra_result_meta["totalDurationMs"])
            except (TypeError, ValueError):
                pass
        return None


@dataclass
class RawNode:
    """Represents a raw JSONL event line for building the UUID conversation tree."""

    line_no: int
    uuid: str
    parent_uuid: str | None
    node_type: str  # user, assistant, system, summary
    timestamp: datetime | None
    is_sidechain: bool = False
    agent_id: str | None = None
    label: str = ""
    tool_names: list[str] = field(default_factory=list)
    has_thinking: bool = False
    is_tool_result_only: bool = False
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class Turn:
    """A logical turn in the conversation timeline."""

    index: int
    role: str  # "user", "assistant", "system", "summary"
    uuid: str
    parent_uuid: str | None = None
    message_id: str | None = None
    timestamp: datetime | None = None
    end_timestamp: datetime | None = None
    is_sidechain: bool = False
    agent_id: str | None = None
    model: str | None = None
    text_blocks: list[str] = field(default_factory=list)
    thinking_blocks: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    line_numbers: list[int] = field(default_factory=list)


@dataclass
class Session:
    """Complete parsed representation of a Claude Code JSONL session."""

    path: Path
    session_id: str
    project_name: str
    cwd: str = ""
    git_branch: str = ""
    version: str = ""
    primary_model: str = ""
    summaries: list[str] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    raw_nodes: list[RawNode] = field(default_factory=list)
    total_usage: TokenUsage = field(default_factory=TokenUsage)
    total_cost_usd: float = 0.0
    start_time: datetime | None = None
    end_time: datetime | None = None
    line_count: int = 0
    parse_errors: int = 0

    @property
    def duration_seconds(self) -> float | None:
        if self.start_time and self.end_time:
            return max(0.0, (self.end_time - self.start_time).total_seconds())
        return None

    @property
    def user_prompt_count(self) -> int:
        return sum(1 for t in self.turns if t.role == "user" and not t.is_sidechain)

    @property
    def assistant_turn_count(self) -> int:
        return sum(1 for t in self.turns if t.role == "assistant")

    @property
    def all_tool_calls(self) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for t in self.turns:
            calls.extend(t.tool_calls)
        return calls

    @property
    def first_user_prompt(self) -> str:
        for t in self.turns:
            if t.role == "user" and t.text_blocks and not t.is_sidechain:
                text = " ".join(t.text_blocks).strip()
                if text:
                    return text
        if self.summaries:
            return self.summaries[0]
        return "(no user prompt)"
