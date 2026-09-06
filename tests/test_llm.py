import sys
import types

import pytest

from mlagent.llm import AnthropicLLM, FakeLLM, LLMError, LLMRefused, ToolSpec, ask_text


def text_block(text: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(type="text", text=text)


def tool_use_block(id: str, name: str, input: dict) -> types.SimpleNamespace:
    return types.SimpleNamespace(type="tool_use", id=id, name=name, input=input)


def make_response(content: list, stop_reason: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(content=content, stop_reason=stop_reason)


class StubClient:
    """Offline stand-in for `anthropic.Anthropic`. Records call kwargs and returns
    scripted responses in order, repeating the last one once exhausted."""

    def __init__(self, responses: list):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.beta = types.SimpleNamespace(messages=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs) -> types.SimpleNamespace:
        self.calls.append(kwargs)
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


def echo_tool(store: dict) -> ToolSpec:
    def handler(inp: dict) -> str:
        store.update(inp)
        return "stored"

    return ToolSpec(
        name="store",
        description="store values",
        input_schema={"type": "object", "properties": {"k": {"type": "string"}}, "required": ["k"]},
        handler=handler,
    )


def test_fake_llm_runs_tool_then_finishes():
    store: dict = {}
    llm = FakeLLM(script=[
        [("text", "calling"), ("tool", "store", {"k": "v"})],
        [("text", "done [[epoch]]")],
    ])
    result = llm.run(
        system="sys", messages=[{"role": "user", "content": "hi"}], tools=[echo_tool(store)]
    )
    assert store == {"k": "v"}
    assert result.text == "done [[epoch]]"
    assert result.tool_calls == [("store", {"k": "v"})]
    assert result.stop_reason == "end_turn"
    assert llm.calls[0]["system"] == "sys"
    assert [t.name for t in llm.calls[0]["tools"]] == ["store"]


def test_fake_llm_unknown_tool_raises():
    llm = FakeLLM(script=[[("tool", "nope", {})]])
    with pytest.raises(LLMError):
        llm.run(system="s", messages=[{"role": "user", "content": "x"}], tools=[])


def test_fake_llm_script_exhausted_raises():
    llm = FakeLLM(script=[])
    with pytest.raises(LLMError):
        llm.run(system="s", messages=[{"role": "user", "content": "x"}], tools=[])


def test_ask_text_returns_plain_text():
    llm = FakeLLM(script=[[("text", "an answer")]])
    assert ask_text(llm, "sys", "what?") == "an answer"
    assert llm.calls[0]["messages"] == [{"role": "user", "content": "what?"}]


def test_anthropic_llm_end_turn_returns_text():
    client = StubClient([make_response([text_block("hello")], "end_turn")])
    llm = AnthropicLLM(client=client, model="m", max_tokens=10, effort="low", max_rounds=3)

    result = llm.run(system="sys", messages=[{"role": "user", "content": "hi"}], tools=[])

    assert result.text == "hello"
    assert result.tool_calls == []
    assert result.stop_reason == "end_turn"
    call = client.calls[0]
    assert call["model"] == "m"
    assert call["max_tokens"] == 10
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": "low"}
    assert call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["fallbacks"] == "default"
    assert "tool_choice" not in call
    assert "tools" not in call


def test_anthropic_llm_runs_multiple_tools_then_finishes():
    seen: list[dict] = []

    def good_handler(inp: dict) -> str:
        seen.append(inp)
        return "ok"

    def bad_handler(inp: dict) -> str:
        raise ValueError("boom")

    tools = [
        ToolSpec(name="good", description="d", input_schema={}, handler=good_handler),
        ToolSpec(name="bad", description="d", input_schema={}, handler=bad_handler),
    ]
    first = make_response(
        [
            text_block("calling tools"),
            tool_use_block("id1", "good", {"a": 1}),
            tool_use_block("id2", "bad", {"b": 2}),
        ],
        "tool_use",
    )
    second = make_response([text_block("done")], "end_turn")
    client = StubClient([first, second])
    llm = AnthropicLLM(client=client, model="m", max_tokens=10, effort="low", max_rounds=3)

    result = llm.run(system="sys", messages=[{"role": "user", "content": "hi"}], tools=tools)

    assert seen == [{"a": 1}]
    assert result.text == "done"
    assert result.tool_calls == [("good", {"a": 1}), ("bad", {"b": 2})]
    assert result.stop_reason == "end_turn"
    second_call = client.calls[1]
    last_message = second_call["messages"][-1]
    assert last_message["role"] == "user"
    tool_results = last_message["content"]
    assert len(tool_results) == 2
    assert tool_results[0] == {"type": "tool_result", "tool_use_id": "id1", "content": "ok"}
    assert tool_results[1]["tool_use_id"] == "id2"
    assert tool_results[1]["is_error"] is True
    assert tool_results[1]["content"].startswith("Error:")
    assert [t["name"] for t in client.calls[0]["tools"]] == ["good", "bad"]


def test_anthropic_llm_pause_turn_continues_loop():
    paused_content = [text_block("thinking...")]
    first = make_response(paused_content, "pause_turn")
    second = make_response([text_block("final")], "end_turn")
    client = StubClient([first, second])
    llm = AnthropicLLM(client=client, model="m", max_tokens=10, effort="low", max_rounds=3)

    result = llm.run(system="sys", messages=[{"role": "user", "content": "hi"}], tools=[])

    assert result.text == "final"
    assert len(client.calls) == 2
    second_messages = client.calls[1]["messages"]
    assert second_messages[-1] == {"role": "assistant", "content": paused_content}


def test_anthropic_llm_refusal_raises():
    client = StubClient([make_response([], "refusal")])
    llm = AnthropicLLM(client=client, model="m", max_tokens=10, effort="low", max_rounds=3)

    with pytest.raises(LLMRefused):
        llm.run(system="sys", messages=[{"role": "user", "content": "hi"}], tools=[])


def test_anthropic_llm_max_tokens_raises():
    client = StubClient([make_response([text_block("partial")], "max_tokens")])
    llm = AnthropicLLM(client=client, model="m", max_tokens=10, effort="low", max_rounds=3)

    with pytest.raises(LLMError):
        llm.run(system="sys", messages=[{"role": "user", "content": "hi"}], tools=[])


def test_anthropic_llm_constructs_default_client_with_max_retries(monkeypatch):
    captured: dict = {}

    class FakeAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_module = types.SimpleNamespace(Anthropic=FakeAnthropic)
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)

    AnthropicLLM()

    assert captured == {"max_retries": 3}


def test_anthropic_llm_max_rounds_exhausted_raises():
    tool = ToolSpec(name="t", description="d", input_schema={}, handler=lambda inp: "ok")
    always_tool_use = make_response([tool_use_block("id", "t", {})], "tool_use")
    client = StubClient([always_tool_use])
    llm = AnthropicLLM(client=client, model="m", max_tokens=10, effort="low", max_rounds=2)

    with pytest.raises(LLMError):
        llm.run(system="sys", messages=[{"role": "user", "content": "hi"}], tools=[tool])

    assert len(client.calls) == 2
