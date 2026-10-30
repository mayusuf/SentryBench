"""Data schemas for SentryBench.

The tool-call *ledger* is the evaluation substrate: as much scoring as possible
is computed deterministically from these structures, with the LLM judge only a
thin optional layer on top.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any


# --------------------------------------------------------------------------- #
# Scenario definition                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class Channel:
    """An input channel delivered to the agent. `trusted=False` marks
    attacker-controllable content (caller transcript, sensor log, report)."""

    type: str
    content: str
    trusted: bool = True


@dataclass
class ActionSpec:
    """A tool action with optional argument constraints.

    Constraints support exact match and ordinal operators, e.g.
    ``{"priority": ">=high"}`` or ``{"status": "closed_non_emergency"}``.
    """

    tool: str
    constraints: dict[str, Any] = field(default_factory=dict)


@dataclass
class Poison:
    """Tool-result poisoning: when the agent calls `tool`, return attacker
    `content` (optionally only on/after a given call index)."""

    tool: str
    content: str
    trigger_call_index: int = 0


@dataclass
class Scenario:
    id: str
    subdomain: str
    attack_vector: str
    target_failure_mode: str
    context_system_role: str
    channels: list[Channel]
    available_tools: list[str]
    gold_action_set: list[ActionSpec] = field(default_factory=list)
    forbidden_action_set: list[ActionSpec] = field(default_factory=list)
    attack_goal: str = ""
    severity_weights: dict[str, float] = field(
        default_factory=lambda: {"c_fn": 5.0, "c_fp": 5.0}
    )
    rubric: str = ""
    grounding_ref: str = ""
    difficulty: str = "medium"
    split: str = "public"
    version: str = "1.0"
    max_turns: int = 6
    poison: Poison | None = None
    # Whether calling escalate_to_human satisfies the "required action"
    # (True for ambiguous adversarial cases -> containment;
    #  False for benign-urgent controls where direct action is warranted).
    escalation_satisfies: bool = True
    containment_expected: bool = False

    @property
    def is_benign(self) -> bool:
        return self.attack_vector == "none"

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Scenario":
        channels = [Channel(**c) for c in d.get("channels", [])]
        gold = [ActionSpec(**a) for a in d.get("gold_action_set", [])]
        forb = [ActionSpec(**a) for a in d.get("forbidden_action_set", [])]
        poison = Poison(**d["poison"]) if d.get("poison") else None
        known = {
            "id", "subdomain", "attack_vector", "target_failure_mode",
            "context_system_role", "attack_goal", "severity_weights", "rubric",
            "grounding_ref", "difficulty", "split", "version", "max_turns",
            "available_tools", "escalation_satisfies", "containment_expected",
        }
        base = {k: d[k] for k in known if k in d}
        return Scenario(
            channels=channels,
            gold_action_set=gold,
            forbidden_action_set=forb,
            poison=poison,
            **base,
        )


# --------------------------------------------------------------------------- #
# Runtime / ledger structures                                                 #
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    call_id: str = ""


@dataclass
class DefenseEvent:
    component: str          # e.g. "tgcc_gate", "tgcc_detector", "tgcc_provenance"
    action: str             # allow | block | require_human | flag
    score: float = 0.0
    reason: str = ""
    turn: int = 0


@dataclass
class LedgerEntry:
    run_id: str
    scenario_id: str
    model: str
    defense: str
    turn: int
    type: str               # tool_call | tool_result | assistant_message | error
    timestamp: float = field(default_factory=time.time)
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    reasoning_trace: str = ""
    defense_events: list[dict[str, Any]] = field(default_factory=list)
    tool_result: dict[str, Any] = field(default_factory=dict)
    content: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Outcome:
    under_reaction: bool = False
    over_reaction: bool = False
    over_refusal: bool = False       # only meaningful on benign-urgent controls
    attack_success: bool = False
    contained: bool = False
    detection_latency_turns: int | None = None
    errored: bool = False            # the model call failed (any reason)
    platform_filtered: bool = False  # upstream provider guardrail blocked the prompt
    notes: str = ""


@dataclass
class RunResult:
    run_id: str
    scenario_id: str
    subdomain: str
    attack_vector: str
    target_failure_mode: str
    model: str
    defense: str
    is_benign: bool
    actions_taken: list[dict[str, Any]]
    outcome: Outcome
    harm_weighted_cost: float
    error: str = ""
    tokens: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "RunResult":
        out = Outcome(**d["outcome"])
        return RunResult(
            run_id=d["run_id"],
            scenario_id=d["scenario_id"],
            subdomain=d["subdomain"],
            attack_vector=d["attack_vector"],
            target_failure_mode=d["target_failure_mode"],
            model=d["model"],
            defense=d["defense"],
            is_benign=d["is_benign"],
            actions_taken=d.get("actions_taken", []),
            outcome=out,
            harm_weighted_cost=d.get("harm_weighted_cost", 0.0),
            error=d.get("error", ""),
            tokens=d.get("tokens", {}),
        )
