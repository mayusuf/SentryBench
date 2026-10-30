"""Scenario loading and validation."""

from __future__ import annotations

import json
from pathlib import Path

from .schemas import Scenario


def load_scenarios(directory: Path) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for fp in sorted(directory.glob("*.json")):
        raw = json.loads(fp.read_text(encoding="utf-8"))
        items = raw if isinstance(raw, list) else [raw]
        for item in items:
            scenarios.append(Scenario.from_dict(item))
    _validate_unique_ids(scenarios)
    return scenarios


def _validate_unique_ids(scenarios: list[Scenario]) -> None:
    seen: set[str] = set()
    for s in scenarios:
        if s.id in seen:
            raise ValueError(f"Duplicate scenario id: {s.id}")
        seen.add(s.id)
