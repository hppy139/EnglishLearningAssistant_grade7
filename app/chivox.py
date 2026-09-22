"""驰声 MCP（Streamable HTTP / JSON-RPC 2.0）客户端。

只做三件事：连接、列工具、调评测工具。
"""

from __future__ import annotations

import json
import os
import time
from typing import Any
from uuid import uuid4

import httpx

from . import config

PROTOCOL_VERSION = "2024-11-05"


class ChivoxMCPError(RuntimeError):
    pass


class ChivoxMCP:
    def __init__(
        self,
        api_key: str | None = None,
        url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.api_key = api_key or config.chivox_api_key()
        self.url = (url or config.chivox_mcp_url()).rstrip("/")
        if not self.api_key:
            raise ChivoxMCPError(
                "缺少 CHIVOX_API_KEY。请到 https://api-portal.cloud.chivox.com 申请后写入 .env。"
            )
        self.timeout = timeout if timeout is not None else config.chivox_timeout()
        self._session_id: str | None = None
        self._initialized = False
        # trust_env=False：忽略 shell 里的 http_proxy/all_proxy（集群上常导致请求直接失败）
        self._client = httpx.Client(timeout=self.timeout, trust_env=config.http_trust_env())

    # ---- 生命周期 ----
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ChivoxMCP":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- 底层 ----
    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": PROTOCOL_VERSION,
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    def _parse_body(self, response: httpx.Response) -> dict[str, Any]:
        sid = response.headers.get("mcp-session-id") or response.headers.get("Mcp-Session-Id")
        if sid:
            self._session_id = sid
        ctype = response.headers.get("content-type", "")
        text = response.text.strip()
        if "text/event-stream" in ctype or text.startswith("event:") or "data:" in text:
            last: dict[str, Any] | None = None
            for line in text.splitlines():
                if line.startswith("data:"):
                    payload = line[5:].strip()
                    if payload and payload != "[DONE]":
                        last = json.loads(payload)
            if last is None:
                raise ChivoxMCPError(f"SSE 响应为空: {text[:400]}")
            return last
        if not text:
            raise ChivoxMCPError(f"空响应 HTTP {response.status_code}")
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise ChivoxMCPError(f"无法解析 JSON: {text[:400]}") from exc

    def _rpc(self, method: str, params: dict[str, Any] | None = None, notify: bool = False) -> dict[str, Any]:
        body: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if not notify:
            body["id"] = str(uuid4())
        if params is not None:
            body["params"] = params
        response: httpx.Response | None = None
        for attempt in (1, 2):
            try:
                response = self._client.post(self.url, headers=self._headers(), json=body)
                break
            except httpx.TimeoutException as exc:
                if attempt == 2:
                    raise ChivoxMCPError(
                        f"评测服务响应超时（等待超过 {self.timeout:.0f} 秒）：多为音频过长或服务繁忙，"
                        "请把单题音频压到 15 秒以内再试"
                    ) from exc
                time.sleep(1.0)
            except httpx.HTTPError as exc:
                raise ChivoxMCPError(f"无法连接 {self.url}: {exc}") from exc
        if response is None:
            raise ChivoxMCPError("评测服务无响应，请稍后重试")
        if response.status_code in (401, 403):
            raise ChivoxMCPError(f"鉴权失败 HTTP {response.status_code}，请检查 CHIVOX_API_KEY。")
        if response.status_code >= 400:
            raise ChivoxMCPError(f"MCP 请求失败 HTTP {response.status_code}: {response.text[:300]}")
        if notify:
            return {}
        data = self._parse_body(response)
        if data.get("error"):
            err = data["error"]
            raise ChivoxMCPError(f"MCP error {err.get('code')}: {err.get('message')}")
        return data.get("result") or {}

    def ensure(self) -> None:
        if self._initialized:
            return
        self._rpc(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "grade1-oral-placement", "version": "2.0.0"},
            },
        )
        try:
            self._rpc("notifications/initialized", {}, notify=True)
        except ChivoxMCPError:
            pass
        self._initialized = True

    def list_tools(self) -> list[dict[str, Any]]:
        self.ensure()
        result = self._rpc("tools/list", {})
        return result.get("tools") or []

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.ensure()
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        contents = result.get("content") or []
        texts: list[str] = []
        for item in contents:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text") or "")
            elif isinstance(item, str):
                texts.append(item)
        joined = "\n".join(t for t in texts if t).strip()
        if not joined:
            return result
        try:
            return json.loads(joined)
        except json.JSONDecodeError:
            return joined

    # ---- 业务封装 ----
    def evaluate(
        self,
        tool: str,
        ref_text: str,
        audio_base64: str | None = None,
        audio_url: str | None = None,
        accent: int | None = None,
        rank: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Any:
        if not audio_base64 and not audio_url:
            raise ChivoxMCPError("需要 audio_base64 或 audio_url 之一。")
        if not ref_text:
            raise ChivoxMCPError("缺少 ref_text（评测参考文本）。")
        args: dict[str, Any] = {"ref_text": ref_text}
        if audio_base64:
            args["audio_base64"] = audio_base64
        if audio_url:
            args["audio_url"] = audio_url
        args["accent"] = accent if accent is not None else config.chivox_accent()
        args["rank"] = rank if rank is not None else config.chivox_rank()
        if extra:
            args.update(extra)
        return self.call_tool(tool, args)

    def create_stream_session(self, core_type: str, ref_text: str, **kwargs: Any) -> Any:
        """实时录音流式评测：返回 session_id，客户端用 WebSocket 推音频。"""
        self.ensure()
        args: dict[str, Any] = {"core_type": core_type, "ref_text": ref_text}
        args.update({k: v for k, v in kwargs.items() if v is not None})
        return self.call_tool("create_stream_session", args)


def has_key() -> bool:
    return bool(os.getenv("CHIVOX_API_KEY"))
