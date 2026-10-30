"""Ledger store: append-only JSONL of every tool call, tool result, and
defense event, plus a run manifest for reproducibility."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .schemas import LedgerEntry


class LedgerStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def append(self, entry: LedgerEntry) -> None:
        self._fh.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {**manifest, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
