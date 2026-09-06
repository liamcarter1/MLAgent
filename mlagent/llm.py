"""Claude API wrapper: a manual tool-use loop behind a small protocol, plus a fake for tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from mlagent import config


class LLMError(RuntimeError):
    pass


class LLMRefused(LLMError):
    pass


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict], str]

    def to_api(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class LLMResult:
    text: str
    tool_calls: list[tuple[str, dict]] = field(default_factory=list)
    stop_reason: str = "end_turn"


class LLM(Protocol):
    def run(self, system: str, messages: list[dict], tools: list[ToolSpec]) -> LLMResult: ...


def _dispatch(name: str, inp: dict, tools: list[ToolSpec]) -> tuple[str, bool]:
    """Run a tool handler. Returns (content, is_error)."""
    by_name = {t.name: t for t in tools}
    if name not in by_name:
        raise LLMError(f"model called unknown tool {name!r}")
    try:
        return by_name[name].handler(dict(inp)), False
    except Exception as exc:  # noqa: BLE001 - tool errors go back to the model, not up the stack
        return f"Error: {exc}", True


class AnthropicLLM:
    def __init__(
        self,
        client: Any = None,
        model: str = config.MODEL_ID,
        max_tokens: int = config.MAX_TOKENS,
        effort: str = config.EFFORT,
        max_rounds: int = config.MAX_TOOL_ROUNDS,
    ):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.max_rounds = max_rounds

    def run(self, system: str, messages: list[dict], tools: list[ToolSpec]) -> LLMResult:
        history = list(messages)
        tool_calls: list[tuple[str, dict]] = []
        last_text = ""
        for _ in range(self.max_rounds):
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=history,
                tools=[t.to_api() for t in tools],
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
            texts = [b.text for b in response.content if b.type == "text"]
            if texts:
                last_text = "\n".join(texts)
            if response.stop_reason == "refusal":
                raise LLMRefused("the model declined this request")
            if response.stop_reason == "max_tokens":
                raise LLMError("response truncated at max_tokens")
            if response.stop_reason == "pause_turn":
                history.append({"role": "assistant", "content": response.content})
                continue
            uses = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not uses:
                return LLMResult(text=last_text, tool_calls=tool_calls, stop_reason="end_turn")
            history.append({"role": "assistant", "content": response.content})
            results = []
            for use in uses:
                content, is_error = _dispatch(use.name, use.input, tools)
                tool_calls.append((use.name, dict(use.input)))
                block = {"type": "tool_result", "tool_use_id": use.id, "content": content}
                if is_error:
                    block["is_error"] = True
                results.append(block)
            history.append({"role": "user", "content": results})
        raise LLMError(f"tool loop exceeded {self.max_rounds} rounds")


class FakeLLM:
    """Scripted stand-in. Each turn: list of ("text", str) or ("tool", name, input)."""

    def __init__(self, script: list[list[tuple]]):
        self.script = list(script)
        self.calls: list[dict] = []

    def run(self, system: str, messages: list[dict], tools: list[ToolSpec]) -> LLMResult:
        self.calls.append({"system": system, "messages": list(messages), "tools": list(tools)})
        tool_calls: list[tuple[str, dict]] = []
        last_text = ""
        while True:
            if not self.script:
                raise LLMError("FakeLLM script exhausted")
            turn = self.script.pop(0)
            uses = []
            for item in turn:
                if item[0] == "text":
                    last_text = item[1]
                elif item[0] == "tool":
                    uses.append((item[1], item[2]))
                else:
                    raise LLMError(f"bad FakeLLM item {item!r}")
            if not uses:
                return LLMResult(text=last_text, tool_calls=tool_calls, stop_reason="end_turn")
            for name, inp in uses:
                _dispatch(name, inp, tools)
                tool_calls.append((name, dict(inp)))


def ask_text(llm: LLM, system: str, prompt: str) -> str:
    return llm.run(system=system, messages=[{"role": "user", "content": prompt}], tools=[]).text
