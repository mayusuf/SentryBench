"""Defense interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..schemas import Channel, DefenseEvent, ToolCall


@dataclass
class GateDecision:
    action: str  # "allow" | "block" | "require_human"
    score: float = 0.0
    reason: str = ""
    component: str = "defense"

    def to_event(self, turn: int) -> DefenseEvent:
        return DefenseEvent(
            component=self.component,
            action=self.action,
            score=self.score,
            reason=self.reason,
            turn=turn,
        )


@dataclass
class DefenseContext:
    """Mutable per-episode context shared with the defense."""

    system_role: str
    channels: list[Channel]
    available_tools: list[str]
    # running signals
    injection_score: float = 0.0
    events: list[DefenseEvent] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class Defense:
    """Base no-op defense. Subclasses override the hooks they need."""

    name = "base"

    def pre_prompt(self, ctx: DefenseContext) -> tuple[str, list[Channel]]:
        """Return (possibly transformed) system_role and channels."""
        return ctx.system_role, ctx.channels

    def gate_tool_call(
        self, call: ToolCall, ctx: DefenseContext, turn: int
    ) -> GateDecision:
        """Decide whether a proposed tool call may execute."""
        return GateDecision(action="allow", component=self.name)

    def on_tool_result(
        self, tool: str, result: dict[str, Any], ctx: DefenseContext, turn: int
    ) -> dict[str, Any]:
        """Inspect/transform a tool result (e.g. poisoning detection)."""
        return result
