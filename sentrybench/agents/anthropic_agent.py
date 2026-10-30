"""Claude (Anthropic) acting-agent adapter."""

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
from ..config import get_anthropic_settings
from ..ledger import LedgerStore
from ..schemas import Scenario, ToolCall
from ..tools import TOOL_SPECS, ToolRegistry


class AnthropicAgent(BaseAgent):
    model_id = "claude"

    def __init__(self, model_label: str = "claude", model_name: str | None = None):
        self.model_id = model_label
        self.settings = get_anthropic_settings()
        if model_name:
            self.settings.model = model_name
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.settings.api_key)
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
                "name": s.name,
                "description": s.description,
                "input_schema": s.parameters,
            }
            for s in registry.specs_for(scenario.available_tools)
        ]
        messages = [{"role": "user", "content": build_user_message(channels)}]

        try:
            client = self._client_lazy()
        except Exception as e:  # pragma: no cover
            result.error = f"client_init: {e}"
            return result

        in_tok = out_tok = 0
        for turn in range(1, scenario.max_turns + 1):
            try:
                resp = client.messages.create(
                    model=self.settings.model,
                    max_tokens=self.settings.max_tokens,
                    temperature=self.settings.temperature,
                    system=system,
                    messages=messages,
                    tools=tools,
                )
            except Exception as e:
                result.error = f"api: {e}"
                break

            if getattr(resp, "usage", None):
                in_tok += getattr(resp.usage, "input_tokens", 0) or 0
                out_tok += getattr(resp.usage, "output_tokens", 0) or 0

            assistant_content = []
            text_out = []
            tool_uses = []
            for block in resp.content:
                if block.type == "text":
                    text_out.append(block.text)
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    tool_uses.append(block)
                    assistant_content.append(
                        {"type": "tool_use", "id": block.id,
                         "name": block.name, "input": block.input}
                    )
            log_assistant(ledger, run_id, scenario, self.model_id, defense, turn,
                          " ".join(text_out))
            messages.append({"role": "assistant", "content": assistant_content})

            if not tool_uses:
                result.final_text = " ".join(text_out)
                break

            tool_result_blocks = []
            for tu in tool_uses:
                call = ToolCall(name=tu.name, args=dict(tu.input or {}), call_id=tu.id)
                payload, executed = handle_tool_call(
                    call, registry=registry, defense=defense, ctx=ctx, turn=turn,
                    run_id=run_id, scenario=scenario, model=self.model_id,
                    ledger=ledger, result=result,
                )
                if executed:
                    result.actions_taken.append(call)
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps(payload),
                })
            messages.append({"role": "user", "content": tool_result_blocks})

            if getattr(resp, "stop_reason", None) != "tool_use":
                break

        result.tokens = {"prompt": in_tok, "completion": out_tok}
        return result
