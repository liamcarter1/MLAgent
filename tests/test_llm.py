import pytest

from mlagent.llm import FakeLLM, LLMError, ToolSpec, ask_text


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
    result = llm.run(system="sys", messages=[{"role": "user", "content": "hi"}], tools=[echo_tool(store)])
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
