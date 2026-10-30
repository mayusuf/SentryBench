"""Preview partial results by re-scoring completed episodes from the ledger.

The runner buffers `run_results.jsonl` until it closes, but the ledger is
flushed on every write. This reconstructs EpisodeResults from the ledger and
applies the real scorer + aggregator so we can preview the frontier mid-run.

Usage: python -m sentrybench.preview [tag]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .agents.base import EpisodeResult
from .config import FINAL_DIR, REPO_ROOT
from .judge import score_outcome
from .metrics import aggregate
from .scenarios import load_scenarios
from .schemas import RunResult, ToolCall


def _load_scenarios_any() -> dict:
    scen = {}
    for d in [REPO_ROOT / "data" / "scenarios_hf", REPO_ROOT / "data" / "scenarios"]:
        if d.exists():
            for s in load_scenarios(d):
                scen[s.id] = s
    return scen


def reconstruct(tag: str) -> list[RunResult]:
    results_dir = FINAL_DIR / (f"results_{tag}" if tag else "results")
    ledger = results_dir / "ledger.jsonl"
    scen = _load_scenarios_any()
    by_run: dict[str, list[dict]] = {}
    for line in ledger.open(encoding="utf-8"):
        e = json.loads(line)
        by_run.setdefault(e["run_id"], []).append(e)

    out: list[RunResult] = []
    for run_id, entries in by_run.items():
        model, defense, sid = run_id.split(":", 2)
        s = scen.get(sid)
        if not s:
            continue
        ep = EpisodeResult()
        gate_turns = []
        for e in entries:
            if e["type"] != "tool_call":
                continue
            devs = e.get("defense_events", [])
            if any(d.get("action") == "require_human" for d in devs):
                ep.human_routed = True
            if any(d.get("action") in ("block", "require_human") for d in devs):
                gate_turns.append(e.get("turn", 0))
            tr = e.get("tool_result", {})
            executed = tr.get("executed") is not False and tr.get("status") not in (
                "routed_to_human", "blocked_by_policy")
            if executed:
                ep.actions_taken.append(ToolCall(e["tool"], e.get("args", {})))
                if e["tool"] == "escalate_to_human":
                    ep.escalated = True
        if gate_turns:
            ep.first_gate_turn = min(gate_turns)
        outcome, cost = score_outcome(s, ep)
        out.append(RunResult(
            run_id=run_id, scenario_id=sid, subdomain=s.subdomain,
            attack_vector=s.attack_vector, target_failure_mode=s.target_failure_mode,
            model=model, defense=defense, is_benign=s.is_benign,
            actions_taken=[{"name": a.name, "args": a.args} for a in ep.actions_taken],
            outcome=outcome, harm_weighted_cost=cost,
        ))
    return out


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    tag = argv[0] if argv else "hf"
    results = reconstruct(tag)
    summary = aggregate(results)
    print(f"PREVIEW tag={tag} — reconstructed {len(results)} episodes from ledger\n")
    hdr = f"{'model':>7} {'defense':>8} {'n_adv':>6} {'ASR':>7} {'FN_urgent':>10} {'FP_ambig':>9} {'contain':>8} {'harm':>6}"
    print(hdr)
    for c in summary["cells"]:
        print(f"{c['model']:>7} {c['defense']:>8} {c['n_adv_executed']:>6} "
              f"{c['asr']*100:>6.1f}% {c['over_refusal_rate']*100:>9.1f}% "
              f"{c['benign_over_reaction_rate']*100:>8.1f}% {c['containment_rate']*100:>7.1f}% "
              f"{c['harm_weighted_severity_mean']:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
