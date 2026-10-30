"""Benchmark orchestration: scenarios x models x defenses -> scored results."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .agents.base import build_agent
from .defenses import build_defense
from .judge import score_outcome
from .ledger import LedgerStore, write_manifest
from .metrics import aggregate
from .schemas import RunResult, Scenario
from .tools import ToolRegistry


def _progress_line(idx: int, total: int, scenario: Scenario, rr: RunResult) -> str:
    o = rr.outcome
    kind = "adv" if not rr.is_benign else (
        "ben-amb" if scenario.target_failure_mode == "over_reaction" else "ben-urg")
    if o.platform_filtered:
        disp = "FILTERED"
    elif o.errored:
        disp = "ERR"
    elif rr.is_benign:
        disp = "OVER-REFUSE" if o.over_refusal else ("OVER-REACT" if o.over_reaction else "ok")
    else:
        disp = "ATTACK!" if o.attack_success else ("contained" if o.contained else "resisted")
    acts = ",".join(a["name"] for a in rr.actions_taken) or "-"
    return (f"[{idx:>4}/{total}] {rr.model:>6} | {rr.defense:>7} | {scenario.id:<12} | "
            f"{kind:<7} | {disp:<11} | {acts}")


def _seed_for(scenario_id: str) -> int:
    return int(hashlib.sha256(scenario_id.encode()).hexdigest(), 16) % (2**32)


def run_benchmark(
    scenarios: list[Scenario],
    models: list[str],
    defenses: list[str],
    out_dir: Path,
    limit: int | None = None,
    append: bool = False,
) -> tuple[list[RunResult], dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = LedgerStore(out_dir / "ledger.jsonl")
    results_path = out_dir / "run_results.jsonl"

    if limit:
        scenarios = scenarios[:limit]

    agents = {m: build_agent(m) for m in models}
    defense_objs = {d: build_defense(d) for d in defenses}

    results: list[RunResult] = []
    started = time.time()
    total = len(scenarios) * len(models) * len(defenses)
    idx = 0

    with results_path.open("a" if append else "w", encoding="utf-8") as rf:
        for scenario in scenarios:
            for model in models:
                for dname in defenses:
                    run_id = f"{model}:{dname}:{scenario.id}"
                    registry = ToolRegistry(_seed_for(scenario.id), scenario.poison)
                    defense = defense_objs[dname]
                    try:
                        episode = agents[model].run_episode(
                            scenario, registry, defense, run_id, ledger
                        )
                    except Exception as e:  # keep the run alive
                        from .agents.base import EpisodeResult
                        episode = EpisodeResult(error=f"episode: {e}")

                    outcome, cost = score_outcome(scenario, episode)
                    if episode.error:
                        outcome.errored = True
                        elow = episode.error.lower()
                        if (
                            "content_filter" in elow
                            or "responsibleaipolicy" in elow
                            or "content management policy" in elow
                        ):
                            outcome.platform_filtered = True
                        # The model never produced a decision, so do not
                        # attribute behavioral outcomes or harm to it. The
                        # disposition is tracked via errored/platform_filtered.
                        outcome.attack_success = False
                        outcome.under_reaction = False
                        outcome.over_reaction = False
                        outcome.over_refusal = False
                        outcome.contained = False
                        cost = 0.0
                    rr = RunResult(
                        run_id=run_id,
                        scenario_id=scenario.id,
                        subdomain=scenario.subdomain,
                        attack_vector=scenario.attack_vector,
                        target_failure_mode=scenario.target_failure_mode,
                        model=model,
                        defense=dname,
                        is_benign=scenario.is_benign,
                        actions_taken=[{"name": a.name, "args": a.args} for a in episode.actions_taken],
                        outcome=outcome,
                        harm_weighted_cost=cost,
                        error=episode.error,
                        tokens=episode.tokens,
                    )
                    results.append(rr)
                    rf.write(json.dumps(rr.to_dict(), ensure_ascii=False) + "\n")
                    rf.flush()
                    idx += 1
                    print(_progress_line(idx, total, scenario, rr), flush=True)

    ledger.close()
    summary = aggregate(results)
    summary["meta"] = {
        "models": models,
        "defenses": defenses,
        "n_scenarios": len(scenarios),
        "n_runs": len(results),
        "n_benign": sum(1 for s in scenarios if s.is_benign),
        "n_adversarial": sum(1 for s in scenarios if not s.is_benign),
        "wall_time_sec": round(time.time() - started, 2),
    }

    (out_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_manifest(out_dir / "manifest.json", {
        "benchmark": "SentryBench",
        "models": models,
        "defenses": defenses,
        "n_scenarios": len(scenarios),
        "scenario_ids": [s.id for s in scenarios],
    })
    return results, summary
