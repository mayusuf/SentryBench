"""Real-data ingestion from Hugging Face via the datasets-server REST API.

We deliberately avoid the heavy `datasets`/`pyarrow` stack (poor wheels on
Python 3.14) and instead page rows over HTTP, caching raw rows under
`data/raw/` so scenario generation is reproducible and offline-repeatable.

Sources (all real, publicly viewable):
  * deepset/prompt-injections                 — real prompt-injection strings
  * jackhhao/jailbreak-classification         — real jailbreak prompts
  * community-datasets/disaster_response_messages — real categorized disaster msgs
  * spikecodes/911-call-transcripts           — real 911 call transcripts
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .config import REPO_ROOT

RAW_DIR = REPO_ROOT / "data" / "raw"
_BASE = "https://datasets-server.huggingface.co/rows"


def _token() -> str | None:
    tok = os.getenv("HF_API_KEY") or os.getenv("Hf_api_key") or os.getenv("HF_TOKEN")
    if tok:
        return tok
    env = REPO_ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("hf_api_key"):
                return line.split("=", 1)[1].strip()
    return None


def _get(url: str) -> dict[str, Any]:
    hdr = {}
    tok = _token()
    if tok:
        hdr["Authorization"] = f"Bearer {tok}"
    req = urllib.request.Request(url, headers=hdr)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.load(r)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch_rows(
    dataset: str,
    split: str = "train",
    config: str = "default",
    max_rows: int = 600,
    cache_name: str | None = None,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    """Page rows (100 at a time) from the datasets-server, with caching."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = RAW_DIR / (cache_name or dataset.replace("/", "__")) 
    cache = cache.with_suffix(".json")
    if cache.exists() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    offset = 0
    ds = urllib.parse.quote(dataset)
    cfg = urllib.parse.quote(config)
    while len(rows) < max_rows:
        length = min(100, max_rows - len(rows))
        url = f"{_BASE}?dataset={ds}&config={cfg}&split={split}&offset={offset}&length={length}"
        data = _get(url)
        batch = [r["row"] for r in data.get("rows", [])]
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        if offset >= data.get("num_rows_total", offset):
            break
    cache.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


# --- Convenience loaders ---------------------------------------------------- #
def load_injections(max_rows: int = 546) -> list[str]:
    rows = fetch_rows("deepset/prompt-injections", max_rows=max_rows,
                      cache_name="deepset_prompt_injections")
    return [r["text"].strip() for r in rows if r.get("label") == 1 and r.get("text")]


def load_jailbreaks(max_rows: int = 600) -> list[str]:
    rows = fetch_rows("jackhhao/jailbreak-classification", max_rows=max_rows,
                      cache_name="jailbreak_classification")
    return [r["prompt"].strip() for r in rows
            if r.get("type") == "jailbreak" and r.get("prompt")]


def load_disaster(max_rows: int = 2000) -> list[dict[str, Any]]:
    rows = fetch_rows("community-datasets/disaster_response_messages",
                      max_rows=max_rows, cache_name="disaster_response_messages")
    # Keep English translated `message`, drop PII-flagged and unrelated noise.
    out = []
    for r in rows:
        msg = (r.get("message") or "").strip()
        if len(msg) < 12 or r.get("PII") == 1 or r.get("related") != 1:
            continue
        out.append(r)
    return out


def load_911(max_rows: int = 250) -> list[str]:
    rows = fetch_rows("spikecodes/911-call-transcripts", max_rows=max_rows,
                      cache_name="911_transcripts")
    transcripts = []
    for r in rows:
        msgs = r.get("messages") or []
        caller = " ".join(
            (m.get("content") or "") for m in msgs if m.get("role") == "user"
        ).strip()
        if len(caller) >= 20:
            transcripts.append(caller)
    return transcripts
