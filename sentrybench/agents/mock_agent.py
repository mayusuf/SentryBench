"""Mock agent for offline pipeline validation (NOT a scientific result).

Simulates a fully input-susceptible agent: on adversarial scenarios it proposes
the attacker-desired (forbidden) action; on benign-urgent scenarios it performs
the warranted (gold) action. This lets us validate scoring, defense gating, and
reporting deterministically without spending API budget. The undefended vs TGCC
contrast it produces is illustrative only.
"""

from __future__ import annotations

from .base import (
    BaseAgent,
    EpisodeResult,
    handle_tool_call,
    log_assistant,
    make_context,
)
from ..ledger import LedgerStore
from ..schemas import Scenario, ToolCall
from ..tools import ToolRegistry


class MockAgent(BaseAgent):
    model_id = "mock"

    def run_episode(
        self, scenario: Scenario, registry: ToolRegistry, defense, run_id: str,
        ledger: LedgerStore,
    ) -> EpisodeResult:
        result = EpisodeResult()
        _system, _channels, ctx = make_context(scenario, defense)

        if scenario.is_benign:
            proposed = [ToolCall(a.tool, _concrete(a.constraints)) for a in scenario.gold_action_set]
            log_assistant(ledger, run_id, scenario, self.model_id, defense, 1,
                          "Legitimate emergency; taking warranted action.")
        else:
            proposed = [ToolCall(a.tool, _concrete(a.constraints)) for a in scenario.forbidden_action_set]
            # Tool-result poisoning: call the poisoned tool first so the defense
            # can observe the manipulated result before the consequential action.
            if scenario.attack_vector == "tool_result_poisoning" and scenario.poison:
                proposed.insert(0, ToolCall(scenario.poison.tool,
                                            {"location": "feeder-3", "timeframe": "24h"}))
            log_assistant(ledger, run_id, scenario, self.model_id, defense, 1,
                          "Following instruction in input (simulated susceptibility).")

        for turn, call in enumerate(proposed, start=1):
            _payload, executed = handle_tool_call(
                call, registry=registry, defense=defense, ctx=ctx, turn=turn,
                run_id=run_id, scenario=scenario, model=self.model_id,
                ledger=ledger, result=result,
            )
            if executed:
                result.actions_taken.append(call)

        result.tokens = {"prompt": 0, "completion": 0}
        return result


def _concrete(constraints: dict) -> dict:
    """Turn constraint specs into concrete tool args for a mock call."""
    out: dict = {}
    for k, v in constraints.items():
        if k == "fields_only_within":
            out["fields"] = list(v)[:2]
        elif k == "fields_include_sensitive":
            if v:
                out["fields"] = ["ssn", "medical_history"]
        elif isinstance(v, str) and v[:1] in "<>":
            out[k] = v.lstrip("><=")
        else:
            out[k] = v
    return {k: val for k, val in out.items() if val is not None}
