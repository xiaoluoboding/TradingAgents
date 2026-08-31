import pytest

from tradingagents.llm_clients import create_llm_client
from tradingagents.llm_clients.api_key_env import get_api_key_env
from tradingagents.llm_clients.codex_client import CodexChatModel


@pytest.mark.unit
def test_codex_model_uses_subscription_provider_type():
    model = CodexChatModel(model="gpt-5.3-codex")
    assert model._llm_type == "codex-chatgpt-subscription"


@pytest.mark.unit
def test_codex_provider_does_not_require_api_key():
    assert get_api_key_env("codex_subscription") is None
    assert type(create_llm_client("codex_subscription", "gpt-5.3-codex")).__name__ == "CodexSubscriptionClient"


@pytest.mark.unit
def test_codex_tool_schema_is_json_schema_compatible():
    class Tool:
        name = "lookup"
        description = "Lookup a value"
        args_schema = None
        args = {"type": "object", "properties": {"key": {"type": "string"}}}

    model = CodexChatModel(model="gpt-5.3-codex").bind_tools([Tool()])
    assert model.tools[0].name == "lookup"
