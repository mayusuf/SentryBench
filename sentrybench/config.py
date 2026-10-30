"""Configuration and secrets loading.

Loads `.env` and exposes typed settings for the Anthropic (Claude) and
Azure OpenAI (GPT-5) backends. Model snapshots are pinned here for
reproducibility and can be overridden via environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover - dotenv optional at import time
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "scenarios"
FINAL_DIR = REPO_ROOT / "final"
RESULTS_DIR = FINAL_DIR / "results"


@dataclass
class AnthropicSettings:
    api_key: str | None
    model: str
    max_tokens: int = 2048
    temperature: float = 0.0

    @property
    def available(self) -> bool:
        return bool(self.api_key)


@dataclass
class AzureOpenAISettings:
    api_key: str | None
    endpoint: str | None
    api_version: str
    deployment: str
    max_completion_tokens: int = 8192
    # GPT-5 / reasoning models only support the default temperature (1).
    # Left as None so we omit the parameter entirely.
    temperature: float | None = None

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.endpoint and self.deployment)


def get_anthropic_settings() -> AnthropicSettings:
    return AnthropicSettings(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        # Candidate default; use `cli.py doctor` to confirm/override via
        # SENTRYBENCH_CLAUDE_MODEL if this snapshot is unavailable on your key.
        model=os.getenv("SENTRYBENCH_CLAUDE_MODEL", "claude-sonnet-4-5-20250929"),
    )


def get_anthropic_haiku_model() -> str:
    """A fast, low-cost *real* Claude snapshot for quick smoke tests."""
    return os.getenv("SENTRYBENCH_CLAUDE_HAIKU_MODEL", "claude-haiku-4-5-20251001")


def get_azure_settings() -> AzureOpenAISettings:
    return AzureOpenAISettings(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5"),
    )


# Logical model ids used across the harness -> backend selector.
MODEL_BACKENDS = {
    "claude": "anthropic",
    "haiku": "anthropic",
    "gpt5": "azure_openai",
    "mock": "mock",
}
