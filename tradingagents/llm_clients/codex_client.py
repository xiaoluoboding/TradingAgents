"""LangChain adapter for the locally installed Codex app-server.

This provider deliberately does not read Codex's credential files.  The Codex
binary owns ChatGPT OAuth, token refresh, plan limits, and account state.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from typing import Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import PrivateAttr

from .base_client import BaseLLMClient


def _tool_schema(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "args_schema", None)
    if schema is not None and hasattr(schema, "model_json_schema"):
        schema = schema.model_json_schema()
    elif schema is not None and hasattr(schema, "schema"):
        schema = schema.schema()
    else:
        schema = getattr(tool, "args", {"type": "object"})
    return {
        "type": "function",
        "name": tool.name,
        "description": getattr(tool, "description", "") or tool.name,
        "inputSchema": schema,
    }


def _strict_json_schema(value: Any) -> Any:
    """Normalize Pydantic JSON Schema for Codex Structured Outputs."""
    if isinstance(value, list):
        return [_strict_json_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized = {key: _strict_json_schema(item) for key, item in value.items()}
    if "$ref" in normalized:
        # Codex Structured Outputs rejects annotations next to a reference.
        return {"$ref": normalized["$ref"]}
    if normalized.get("type") == "object" or "properties" in normalized:
        properties = normalized.get("properties", {})
        normalized["additionalProperties"] = False
        normalized["required"] = list(properties)
    return normalized


class CodexChatModel(BaseChatModel):
    """Synchronous stdio JSON-RPC bridge to Codex app-server."""

    model: str
    codex_bin: str = "codex"
    cwd: str | None = None
    tools: tuple[Any, ...] = ()
    effort: str | None = None
    timeout: float = 600.0
    _output_schema: dict[str, Any] | None = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "codex-chatgpt-subscription"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "CodexChatModel":
        if kwargs:
            # Codex app-server owns tool choice; silently ignoring a LangChain
            # tool_choice avoids sending an OpenAI-specific payload to Codex.
            kwargs.pop("tool_choice", None)
        # Keep callback handlers shared with the parent model so bound and
        # structured-output calls remain visible to the run-level auditor.
        clone = self.model_copy(deep=False)
        clone.tools = tuple(tools)
        return clone

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        del kwargs
        if hasattr(schema, "model_json_schema"):
            output_schema = schema.model_json_schema()
        elif hasattr(schema, "schema"):
            output_schema = schema.schema()
        else:
            output_schema = schema
        clone = self.model_copy(deep=False)
        clone._output_schema = _strict_json_schema(output_schema)

        def parse(result: AIMessage) -> Any:
            value = result.content
            if isinstance(value, str):
                value = json.loads(value.strip().removeprefix("```json").removesuffix("```").strip())
            if hasattr(schema, "model_validate"):
                return schema.model_validate(value)
            return schema.parse_obj(value)

        return RunnableLambda(lambda messages: parse(clone.invoke(messages)))

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        binary = shutil.which(self.codex_bin) or self.codex_bin
        proc = subprocess.Popen(
            [binary, "app-server", "--listen", "stdio://"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
            cwd=self.cwd,
        )
        try:
            def send(message: dict[str, Any]) -> None:
                assert proc.stdin is not None
                proc.stdin.write(json.dumps(message) + "\n")
                proc.stdin.flush()

            def receive(expected_id: str | None = None) -> dict[str, Any]:
                assert proc.stdout is not None
                while True:
                    line = proc.stdout.readline()
                    if not line:
                        raise RuntimeError("Codex app-server exited unexpectedly")
                    item = json.loads(line)
                    if expected_id is None or str(item.get("id")) == expected_id:
                        return item
                    if item.get("method") == "item/tool/call":
                        self._handle_tool_call(send, item)

            def request(method: str, params: dict[str, Any]) -> dict[str, Any]:
                request_id = str(uuid.uuid4())
                send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
                result = receive(request_id)
                if "error" in result:
                    detail = result["error"]
                    message = f"Codex {method} failed: {detail}"
                    if method in {"thread/start", "turn/start"} and any(
                        word in str(detail).lower() for word in ("auth", "login", "unauthorized", "token")
                    ):
                        message += ". Run `codex login` and retry; ChatGPT OAuth is managed by Codex CLI."
                    raise RuntimeError(message)
                return result.get("result", {})

            request("initialize", {
                "clientInfo": {"name": "tradingagents", "title": "TradingAgents", "version": "0.4.0"},
                # The app-server gates `thread/start` parameters and output
                # schemas behind this capability, even for turns without
                # dynamic tools. Keep it enabled for this adapter.
                "capabilities": {"experimentalApi": True},
            })
            send({"jsonrpc": "2.0", "method": "initialized", "params": {}})
            thread = request("thread/start", {
                "model": self.model,
                "cwd": self.cwd or os.getcwd(),
                "approvalPolicy": "never",
                "dynamicTools": [{
                    "type": "namespace",
                    "name": "tradingagents",
                    "description": "TradingAgents market-data and research tools",
                    "tools": [_tool_schema(t) for t in self.tools],
                }] if self.tools else [],
            })["thread"]["id"]
            text = "\n\n".join(self._message_text(message) for message in messages if self._message_text(message))
            turn = request("turn/start", {
                "threadId": thread,
                "input": [{"type": "text", "text": text}],
                "model": self.model,
                **({"effort": self.effort} if self.effort else {}),
                **({"outputSchema": self._output_schema} if self._output_schema else {}),
            })["turn"]["id"]

            answers_by_item: dict[str, str] = {}
            answer_order: list[str] = []
            while True:
                event = receive()
                method = event.get("method")
                params = event.get("params", {})
                if method == "item/tool/call":
                    self._handle_tool_call(send, event)
                elif method == "item/agentMessage/delta":
                    item_id = str(params.get("itemId", "agent-message"))
                    if item_id not in answers_by_item:
                        answer_order.append(item_id)
                    answers_by_item[item_id] = answers_by_item.get(item_id, "") + params.get("delta", "")
                elif method == "item/completed":
                    item = params.get("item", {})
                    if item.get("type") == "agentMessage":
                        item_id = str(item.get("id", "agent-message"))
                        if item_id not in answers_by_item:
                            answer_order.append(item_id)
                        completed_text = item.get("text", item.get("message", ""))
                        if completed_text:
                            answers_by_item[item_id] = completed_text
                elif method == "error" and params.get("turnId") in (None, turn):
                    detail = params.get("error", {})
                    raise RuntimeError(
                        "Codex turn failed: " + str(detail.get("message") or detail)
                    )
                elif method == "turn/completed" and params.get("turn", {}).get("id") == turn:
                    completed = params.get("turn", {})
                    if completed.get("status") != "completed":
                        detail = completed.get("error") or completed.get("status")
                        raise RuntimeError(f"Codex turn failed: {detail}")
                    break
            answer = answers_by_item[answer_order[-1]] if answer_order else ""
            if not answer.strip():
                raise RuntimeError("Codex turn completed without an assistant message")
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=answer))])
        finally:
            proc.terminate()
            proc.wait(timeout=5)

    @staticmethod
    def _message_text(message: BaseMessage) -> str:
        prefix = "system" if isinstance(message, SystemMessage) else "user"
        if isinstance(message, ToolMessage):
            prefix = "tool"
        return f"{prefix}: {message.content}"

    def _handle_tool_call(self, send: Any, event: dict[str, Any]) -> None:
        params = event.get("params", {})
        tool = next((tool for tool in self.tools if tool.name == params.get("tool")), None)
        try:
            if tool is None:
                raise ValueError(f"Unknown Codex dynamic tool: {params.get('tool')}")
            result = tool.invoke(params.get("arguments", {}))
            content = str(result)
            success = True
        except Exception as exc:  # tool errors are returned to Codex for recovery
            content = f"Tool error: {exc}"
            success = False
        send({"jsonrpc": "2.0", "id": event["id"], "result": {
            "contentItems": [{"type": "inputText", "text": content}],
            "success": success,
        }})


class CodexSubscriptionClient(BaseLLMClient):
    """Client using the user's ChatGPT Plus/Pro login managed by Codex CLI."""

    provider = "codex_subscription"

    def get_llm(self) -> CodexChatModel:
        self.warn_if_unknown_model()
        return CodexChatModel(
            model=self.model,
            codex_bin=self.kwargs.get("codex_bin", "codex"),
            cwd=self.kwargs.get("cwd"),
            effort=self.kwargs.get("reasoning_effort"),
            timeout=float(self.kwargs.get("timeout", 600)),
            callbacks=self.kwargs.get("callbacks"),
        )

    def validate_model(self) -> bool:
        return bool(self.model)
