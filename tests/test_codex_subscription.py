import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from tradingagents.llm_clients import create_llm_client
from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.llm_clients.codex_client import CodexChatModel, _strict_json_schema


@pytest.mark.unit
def test_codex_model_uses_subscription_provider_type():
    model = CodexChatModel(model="gpt-5.3-codex")
    assert model._llm_type == "codex-chatgpt-subscription"


@pytest.mark.unit
def test_codex_provider_does_not_require_api_key():
    assert get_api_key_env("codex_subscription") is None
    assert type(create_llm_client("codex_subscription", "gpt-5.3-codex")).__name__ == "CodexSubscriptionClient"


@pytest.mark.unit
def test_codex_provider_forwards_callbacks_to_structured_clones():
    callback = BaseCallbackHandler()
    model = create_llm_client(
        "codex_subscription", "gpt-5.6-sol", callbacks=[callback]
    ).get_llm()
    assert model.callbacks == [callback]

    class Verdict(BaseModel):
        rating: str

    structured = model.with_structured_output(Verdict)
    original_invoke = CodexChatModel.invoke

    def fake_invoke(self, messages, *args, **kwargs):
        assert self.callbacks is model.callbacks
        return AIMessage(content='{"rating":"Hold"}')

    CodexChatModel.invoke = fake_invoke
    try:
        assert structured.invoke("rate it").rating == "Hold"
    finally:
        CodexChatModel.invoke = original_invoke


@pytest.mark.unit
def test_codex_tool_schema_is_json_schema_compatible():
    class Tool:
        name = "lookup"
        description = "Lookup a value"
        args_schema = None
        args = {"type": "object", "properties": {"key": {"type": "string"}}}

    model = CodexChatModel(model="gpt-5.3-codex").bind_tools([Tool()])
    assert model.tools[0].name == "lookup"


@pytest.mark.unit
def test_codex_structured_output_schema_is_strict_at_every_object():
    schema = {
        "type": "object",
        "properties": {
            "decision": {
                "type": "object",
                "properties": {"rating": {"type": "string"}},
            }
        },
    }
    strict = _strict_json_schema(schema)
    assert strict["additionalProperties"] is False
    assert strict["required"] == ["decision"]
    assert strict["properties"]["decision"]["additionalProperties"] is False
    assert strict["properties"]["decision"]["required"] == ["rating"]
    assert _strict_json_schema(
        {"$ref": "#/$defs/Rating", "description": "Decision rating"}
    ) == {"$ref": "#/$defs/Rating"}
