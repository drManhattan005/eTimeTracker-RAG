"""
app/infrastructure/llm.py
──────────────────────────
Thin HTTP wrapper around the local Ollama server supporting qwen2.5:1.5b.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass

import requests

log = logging.getLogger(__name__)


@dataclass
class StreamStats:
    done_reason: str = ""
    prompt_eval_count: int = 0
    eval_count: int = 0
    eval_duration_ms: float = 0.0
    total_duration_ms: float = 0.0


class OllamaClient:
    def __init__(
        self,
        model_name: str = "qwen2.5:1.5b",
        base_url: str = "http://localhost:11434",
        timeout: int | tuple[int, int] = (10, 180),
        keep_alive: str = "10m",
        options: dict | None = None,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.options: dict = options or {}
        self.last_stream_stats: StreamStats | None = None

    def _base_payload(self, prompt: str, stream: bool, system: str | None = None) -> dict:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            "keep_alive": self.keep_alive,
            "options": self.options,
        }
        if system:
            payload["system"] = system
        return payload

    def generate(self, prompt: str, system: str | None = None) -> str:
        log.debug(
            "[ollama] blocking generate | model=%s options=%s prompt_len=%d",
            self.model_name,
            self.options,
            len(prompt),
        )
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=self._base_payload(prompt, stream=False, system=system),
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        log.debug(
            "[ollama] blocking done | done_reason=%s eval_count=%s",
            data.get("done_reason"),
            data.get("eval_count"),
        )
        return data.get("response", "").strip()

    def generate_stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        self.last_stream_stats = None
        log.info(
            "[ollama] stream start | model=%s options=%s prompt_len=%d",
            self.model_name,
            self.options,
            len(prompt),
        )

        try:
            with requests.post(
                f"{self.base_url}/api/generate",
                json=self._base_payload(prompt, stream=True, system=system),
                timeout=self.timeout,
                stream=True,
            ) as resp:
                resp.raise_for_status()

                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue

                    try:
                        data = json.loads(raw_line)
                    except (json.JSONDecodeError, ValueError) as exc:
                        log.warning("[ollama] malformed line skipped: %s | err=%s", raw_line[:120], exc)
                        continue

                    chunk = data.get("response", "")
                    if chunk:
                        yield chunk

                    if data.get("done", False):
                        stats = StreamStats(
                            done_reason=data.get("done_reason", ""),
                            prompt_eval_count=data.get("prompt_eval_count", 0),
                            eval_count=data.get("eval_count", 0),
                            eval_duration_ms=round(
                                data.get("eval_duration", 0) / 1_000_000, 1
                            ),
                            total_duration_ms=round(
                                data.get("total_duration", 0) / 1_000_000, 1
                            ),
                        )
                        self.last_stream_stats = stats
                        log.info(
                            "[ollama] stream done | reason=%s prompt_tokens=%d "
                            "eval_tokens=%d eval_ms=%s total_ms=%s",
                            stats.done_reason,
                            stats.prompt_eval_count,
                            stats.eval_count,
                            stats.eval_duration_ms,
                            stats.total_duration_ms,
                        )
                        return

        except requests.Timeout:
            log.error("[ollama] stream timed out | model=%s", self.model_name)
            raise
        except requests.RequestException as exc:
            log.error("[ollama] stream request error: %s", exc)
            raise
