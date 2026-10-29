"""Minimal OpenAI-compatible chat client for Nemotron (Ollama, NIM, vLLM, build.nvidia.com).

Only the standard library is used so the case generator runs outside the Isaac Sim
Python environment.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass


@dataclass
class ChatReply:
    content: str
    reasoning: str
    usage: dict
    raw: dict


class NemotronClient:
    """Chat-completions client with JSON-schema structured output.

    Args:
        base_url: OpenAI-compatible API root, e.g. ``http://localhost:11434/v1`` (Ollama)
            or ``https://integrate.api.nvidia.com/v1`` (NVIDIA API catalog).
        model: Model name on that server, e.g. ``nemotron-3.5-lightning:30b``.
        api_key_env: Name of the environment variable holding the API key, if any.
        timeout_s: Per-request timeout.
    """

    def __init__(self, base_url: str, model: str, api_key_env: str | None = None, timeout_s: float = 1800.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self._api_key = os.environ.get(api_key_env, "") if api_key_env else ""
        if api_key_env and not self._api_key:
            raise RuntimeError(f"environment variable {api_key_env} is empty")

    @property
    def is_ollama(self) -> bool:
        return urllib.parse.urlparse(self.base_url).port == 11434

    def _post(self, url: str, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{url} -> HTTP {e.code}: {e.read().decode(errors='replace')[:500]}") from e

    def chat_json(
        self, messages: list[dict], schema: dict, temperature: float = 0.6, seed: int | None = None
    ) -> ChatReply:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_schema", "json_schema": {"name": "catalog", "schema": schema}},
        }
        if seed is not None:
            body["seed"] = seed
        data = self._post(f"{self.base_url}/chat/completions", body)
        msg = data["choices"][0]["message"]
        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        return ChatReply(content=msg.get("content") or "", reasoning=reasoning, usage=data.get("usage", {}), raw=data)

    def unload(self) -> None:
        """Free GPU memory held by an Ollama model so Isaac Sim gets the whole GPU."""
        if not self.is_ollama:
            return
        root = self.base_url.rsplit("/v1", 1)[0]
        try:
            self._post(f"{root}/api/generate", {"model": self.model, "keep_alive": 0})
        except Exception as e:  # noqa: BLE001 - best effort
            print(f"[WARN] could not unload {self.model}: {e}")


def parse_json_reply(text: str) -> dict:
    """Parse a JSON object, tolerating code fences or leading <think> blocks."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("reply contains no JSON object")
    return json.loads(text[start : end + 1])
