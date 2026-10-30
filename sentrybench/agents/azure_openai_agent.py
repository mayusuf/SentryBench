"""Azure OpenAI (GPT-5) acting-agent adapter.

GPT-5 is a reasoning model: we use `max_completion_tokens` and omit
`temperature` (only the default is supported).
"""

from __future__ import annotations

import json

from .base import (
    BaseAgent,
    EpisodeResult,
    build_system_prompt,
    build_user_message,
    handle_tool_call,
    log_assistant,
    make_context,
)
from ..config import get_azure_settings
from ..ledger import LedgerStore
from ..schemas import Scenario, ToolCall
from ..tools import ToolRegistry


class AzureOpenAIAgent(BaseAgent):
    model_id = "gpt5"

    def __init__(self):
        self.settings = get_azure_settings()
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            from openai import AzureOpenAI
            self._client = AzureOpenAI(
                api_key=self.settings.api_key,
                api_version=self.settings.api_version,
                azure_endpoint=self.settings.endpoint,
            )
        return self._client

    def run_episode(
        self, scenario: Scenario, registry: ToolRegistry, defense, run_id: str,
        ledger: LedgerStore,
    ) -> EpisodeResult:
        result = EpisodeResult()
        system_role, channels, ctx = make_context(scenario, defense)
        system = build_system_prompt(system_role, scenario.available_tools)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": s.name,
                    "description": s.description,
                    "parameters": s.parameters,
                },
            }
            for s in registry.specs_for(scenario.available_tools)
        ]
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": build_user_message(channels)},
        ]

        try:
            client = self._client_lazy()
        except Exception as e:  # pragma: no cover
            result.error = f"client_init: {e}"
            return result

        p_tok = c_tok = 0
        for turn in range(1, scenario.max_turns + 1):
            kwargs = dict(
                model=self.settings.deployment,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_completion_tokens=self.settings.max_completion_tokens,
            )
            if self.settings.temperature is not None:
                kwargs["temperature"] = self.settings.temperature
            try:
                resp = client.chat.completions.create(**kwargs)
            except Exception as e:
                result.error = f"api: {e}"
                break

            if getattr(resp, "usage", None):
                p_tok += getattr(resp.usage, "prompt_tokens", 0) or 0
                c_tok += getattr(resp.usage, "completion_tokens", 0) or 0

            msg = resp.choices[0].message
            log_assistant(ledger, run_id, scenario, self.model_id, defense, turn,
                          msg.content or "")
            messages.append(msg)

            if not getattr(msg, "tool_calls", None):
                result.final_text = msg.content or ""
                break

            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                call = ToolCall(name=tc.function.name, args=args, call_id=tc.id)
                payload, executed = handle_tool_call(
                    call, registry=registry, defense=defense, ctx=ctx, turn=turn,
                    run_id=run_id, scenario=scenario, model=self.model_id,
                    ledger=ledger, result=result,
                )
                if executed:
                    result.actions_taken.append(call)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(payload),
                })

        result.tokens = {"prompt": p_tok, "completion": c_tok}
        return result
