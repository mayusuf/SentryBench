"""SentryBench command-line interface.

Usage:
  python -m sentrybench.cli doctor
  python -m sentrybench.cli run --models claude,gpt5 --defenses none,tgcc
  python -m sentrybench.cli run --models mock --defenses none,tgcc   # offline
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import (
    DATA_DIR,
    FINAL_DIR,
    RESULTS_DIR,
    get_anthropic_haiku_model,
    get_anthropic_settings,
    get_azure_settings,
)
from .metrics import aggregate
from .report import generate_final_md
from .runner import run_benchmark
from .scenarios import load_scenarios
from .schemas import RunResult


def cmd_doctor(_args: argparse.Namespace) -> int:
    print("== SentryBench doctor ==")
    an = get_anthropic_settings()
    print(f"[anthropic] key present: {an.available}; configured model: {an.model}")
    if an.available:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=an.api_key)
            models = client.models.list()
            ids = [m.id for m in getattr(models, "data", [])][:15]
            print(f"[anthropic] reachable. sample models: {ids}")
            if an.model not in ids and ids:
                print(f"[anthropic] NOTE: '{an.model}' not in the sample list; "
                      f"set SENTRYBENCH_CLAUDE_MODEL to one of the above if calls fail.")
        except Exception as e:
            print(f"[anthropic] ERROR: {e}")

    az = get_azure_settings()
    print(f"[azure] key present: {az.available}; deployment: {az.deployment}; "
          f"endpoint: {az.endpoint}")
    if az.available:
        try:
            from openai import AzureOpenAI
            client = AzureOpenAI(
                api_key=az.api_key, api_version=az.api_version,
                azure_endpoint=az.endpoint,
            )
            resp = client.chat.completions.create(
                model=az.deployment,
                messages=[{"role": "user", "content": "reply with OK"}],
                max_completion_tokens=16,
            )
            print(f"[azure] reachable. sample reply: {resp.choices[0].message.content!r}")
        except Exception as e:
            print(f"[azure] ERROR: {e}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    defenses = [d.strip() for d in args.defenses.split(",") if d.strip()]
    from pathlib import Path
    scen_dir = Path(args.scenarios_dir) if args.scenarios_dir else DATA_DIR
    scenarios = load_scenarios(scen_dir)
    if not scenarios:
        print(f"No scenarios found in {scen_dir}", file=sys.stderr)
        return 1
    print(f"Loaded {len(scenarios)} scenarios from {scen_dir}. Models={models} Defenses={defenses}")

    # A tag isolates output so a quick test never clobbers a main run.
    tag = (args.tag or "").strip()
    results_dir = RESULTS_DIR if not tag else FINAL_DIR / f"results_{tag}"
    final_md = FINAL_DIR / ("final.md" if not tag else f"final_{tag}.md")

    _results, summary = run_benchmark(
        scenarios, models, defenses, results_dir, limit=args.limit, append=args.append
    )

    snapshots = {}
    if "claude" in models:
        snapshots["claude"] = get_anthropic_settings().model
    if "haiku" in models:
        snapshots["haiku"] = get_anthropic_haiku_model()
    if "gpt5" in models:
        snapshots["gpt5"] = get_azure_settings().deployment
    if "mock" in models:
        snapshots["mock"] = "offline-simulator"

    generate_final_md(summary, final_md, snapshots)
    print(f"Done. Wrote {final_md} and artifacts in {results_dir}")
    for c in summary["cells"]:
        print(f"  {c['model']:>6} | {c['defense']:>5} | ASR={c['asr']:.2f} "
              f"ORR={c['over_refusal_rate']:.2f} Contain={c['containment_rate']:.2f} "
              f"Harm={c['harm_weighted_severity_mean']}")
    return 0


def _snapshots_for(models: list[str]) -> dict[str, str]:
    snaps: dict[str, str] = {}
    if "claude" in models:
        snaps["claude"] = get_anthropic_settings().model
    if "haiku" in models:
        snaps["haiku"] = get_anthropic_haiku_model()
    if "gpt5" in models:
        snaps["gpt5"] = get_azure_settings().deployment
    if "mock" in models:
        snaps["mock"] = "offline-simulator"
    return snaps


def cmd_report(args: argparse.Namespace) -> int:
    """Regenerate final.md from existing results (no API calls)."""
    tag = (args.tag or "").strip()
    results_dir = RESULTS_DIR if not tag else FINAL_DIR / f"results_{tag}"
    final_md = FINAL_DIR / ("final.md" if not tag else f"final_{tag}.md")
    rr_path = results_dir / "run_results.jsonl"
    if not rr_path.exists():
        print(f"No results at {rr_path}", file=sys.stderr)
        return 1
    rows = [RunResult.from_dict(json.loads(l)) for l in rr_path.open(encoding="utf-8")]
    summary = aggregate(rows)
    models = sorted({r.model for r in rows})
    defenses = sorted({r.defense for r in rows})
    scen = {r.scenario_id for r in rows}
    summary["meta"] = {
        "models": models,
        "defenses": defenses,
        "n_scenarios": len(scen),
        "n_runs": len(rows),
        "n_benign": len({r.scenario_id for r in rows if r.is_benign}),
        "n_adversarial": len({r.scenario_id for r in rows if not r.is_benign}),
        "wall_time_sec": 0,
    }
    generate_final_md(summary, final_md, _snapshots_for(models))
    print(f"Regenerated {final_md} from {rr_path}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    """Build the real-data-grounded scenario bank from Hugging Face sources."""
    from .generate import write_bank, SCENARIOS_HF_DIR
    out, n = write_bank(SCENARIOS_HF_DIR, seed=args.seed)
    print(f"Wrote {n} scenarios to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sentrybench")
    sub = parser.add_subparsers(dest="command", required=True)

    p_doc = sub.add_parser("doctor", help="Check API connectivity and models.")
    p_doc.set_defaults(func=cmd_doctor)

    p_run = sub.add_parser("run", help="Run the benchmark and write final.md.")
    p_run.add_argument("--models", default="mock",
                       help="Comma list: claude,haiku,gpt5,mock")
    p_run.add_argument("--defenses", default="none,tgcc",
                       help="Comma list: none,caution,tgcc")
    p_run.add_argument("--limit", type=int, default=None,
                       help="Limit number of scenarios (smoke test).")
    p_run.add_argument("--scenarios-dir", default="",
                       help="Directory of scenario JSON (default data/scenarios).")
    p_run.add_argument("--append", action="store_true",
                       help="Append to an existing tagged results dir.")
    p_run.add_argument("--tag", default="",
                       help="Isolate output to final/results_<tag>/ and "
                            "final/final_<tag>.md (avoids clobbering a main run).")
    p_run.set_defaults(func=cmd_run)

    p_gen = sub.add_parser("generate", help="Build real-data scenario bank from Hugging Face.")
    p_gen.add_argument("--seed", type=int, default=7)
    p_gen.set_defaults(func=cmd_generate)

    p_rep = sub.add_parser("report", help="Regenerate final.md from existing results.")
    p_rep.add_argument("--tag", default="",
                       help="Which tagged results to report (default: main).")
    p_rep.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
