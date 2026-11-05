"""
app/infrastructure/llm.py
──────────────────────────
Production Google Gemini Cloud API connector using the official Google GenAI SDK
(google-genai) with streaming support and direct REST fallback.
"""
from __future__ import annotations

import json
import logging
import os
import time
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


class GeminiClient:
    """
    Client for Google Gemini cloud models (e.g. gemini-2.5-flash, gemini-2.0-flash).
    Integrates with the official `google-genai` SDK and provides a resilient
    direct REST fallback when needed.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: str | None = None,
        temperature: float = 0.1,
        max_output_tokens: int = 1024,
        timeout: int = 60,
    ) -> None:
        self.model_name = model_name
        self.api_key = (api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.last_stream_stats: StreamStats | None = None

        self._sdk_client = None
        self._init_sdk()

    def _init_sdk(self) -> None:
        if not self.api_key:
            return

        try:
            from google import genai

            self._sdk_client = genai.Client(api_key=self.api_key)
            log.info(
                "[gemini] Initialized official google-genai SDK client for model=%s",
                self.model_name,
            )
        except ImportError:
            log.info(
                "[gemini] 'google-genai' SDK not found in environment; using direct Gemini REST API. "
                "To use the official SDK, run `pip install google-genai`."
            )
            self._sdk_client = None

    def _ensure_api_key(self) -> str:
        key = (self.api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
        if not key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please add GEMINI_API_KEY to your .env file "
                "or export it as an environment variable.\n"
                "You can generate an API key at: https://aistudio.google.com/app/apikey"
            )
        if not self.api_key:
            self.api_key = key
            self._init_sdk()
        return key

    def generate(self, prompt: str, system: str | None = None) -> str:
        """
        Generate non-streaming response from Gemini.
        """
        api_key = self._ensure_api_key()
        t0 = time.monotonic()
        log.debug(
            "[gemini] blocking generate | model=%s prompt_len=%d",
            self.model_name,
            len(prompt),
        )

        # 1. Attempt official google-genai SDK first
        if self._sdk_client is not None:
            try:
                from google.genai import types

                config = types.GenerateContentConfig(
                    system_instruction=system if system else None,
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                )
                response = self._sdk_client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )

                text = response.text or ""
                prompt_tokens = 0
                eval_tokens = 0
                if getattr(response, "usage_metadata", None):
                    prompt_tokens = (
                        getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                    )
                    eval_tokens = (
                        getattr(response.usage_metadata, "candidates_token_count", 0) or 0
                    )

                total_ms = round((time.monotonic() - t0) * 1000, 1)
                self.last_stream_stats = StreamStats(
                    done_reason="stop",
                    prompt_eval_count=prompt_tokens,
                    eval_count=eval_tokens,
                    eval_duration_ms=total_ms,
                    total_duration_ms=total_ms,
                )
                return text.strip()
            except Exception as exc:
                log.warning("[gemini] SDK generate call failed (%s); trying REST fallback", exc)

        # 2. Resilient direct REST API call
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent?key={api_key}"
        )
        payload: dict = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()

            text = ""
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts)

            usage = data.get("usageMetadata", {})
            total_ms = round((time.monotonic() - t0) * 1000, 1)
            self.last_stream_stats = StreamStats(
                done_reason="stop",
                prompt_eval_count=usage.get("promptTokenCount", 0),
                eval_count=usage.get("candidatesTokenCount", 0),
                eval_duration_ms=total_ms,
                total_duration_ms=total_ms,
            )
            return text.strip()
        except requests.RequestException as exc:
            log.error("[gemini] REST generate error: %s", exc)
            raise

    def generate_stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        """
        Stream response tokens from Gemini.
        """
        api_key = self._ensure_api_key()
        self.last_stream_stats = None
        t0 = time.monotonic()
        log.info(
            "[gemini] stream start | model=%s prompt_len=%d",
            self.model_name,
            len(prompt),
        )

        # 1. Attempt official google-genai SDK first
        if self._sdk_client is not None:
            try:
                from google.genai import types

                config = types.GenerateContentConfig(
                    system_instruction=system if system else None,
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                )
                response_stream = self._sdk_client.models.generate_content_stream(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )

                prompt_tokens = 0
                eval_tokens = 0
                done_reason = "stop"

                for chunk in response_stream:
                    if getattr(chunk, "usage_metadata", None):
                        prompt_tokens = getattr(
                            chunk.usage_metadata, "prompt_token_count", prompt_tokens
                        ) or prompt_tokens
                        eval_tokens = getattr(
                            chunk.usage_metadata, "candidates_token_count", eval_tokens
                        ) or eval_tokens

                    if getattr(chunk, "candidates", None) and chunk.candidates:
                        c = chunk.candidates[0]
                        if getattr(c, "finish_reason", None):
                            done_reason = str(c.finish_reason)

                    chunk_text = chunk.text
                    if chunk_text:
                        yield chunk_text

                total_ms = round((time.monotonic() - t0) * 1000, 1)
                self.last_stream_stats = StreamStats(
                    done_reason=done_reason,
                    prompt_eval_count=prompt_tokens,
                    eval_count=eval_tokens,
                    eval_duration_ms=total_ms,
                    total_duration_ms=total_ms,
                )
                log.info(
                    "[gemini] stream done | reason=%s prompt_tokens=%d eval_tokens=%d total_ms=%s",
                    done_reason,
                    prompt_tokens,
                    eval_tokens,
                    total_ms,
                )
                return
            except Exception as exc:
                log.warning("[gemini] SDK stream call failed (%s); trying REST fallback", exc)

        # 2. Resilient direct REST streaming via Server-Sent Events (SSE)
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:streamGenerateContent?alt=sse&key={api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        try:
            with requests.post(url, json=payload, timeout=self.timeout, stream=True) as resp:
                resp.raise_for_status()
                prompt_tokens = 0
                eval_tokens = 0
                done_reason = "stop"

                for line in resp.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode("utf-8") if isinstance(line, bytes) else line
                    if decoded.startswith("data: "):
                        json_str = decoded[6:]
                        try:
                            chunk_data = json.loads(json_str)
                        except json.JSONDecodeError:
                            continue

                        usage = chunk_data.get("usageMetadata", {})
                        if usage:
                            prompt_tokens = usage.get("promptTokenCount", prompt_tokens)
                            eval_tokens = usage.get("candidatesTokenCount", eval_tokens)

                        candidates = chunk_data.get("candidates", [])
                        if candidates:
                            c = candidates[0]
                            if "finishReason" in c:
                                done_reason = c["finishReason"]
                            parts = c.get("content", {}).get("parts", [])
                            text = "".join(p.get("text", "") for p in parts)
                            if text:
                                yield text

                total_ms = round((time.monotonic() - t0) * 1000, 1)
                self.last_stream_stats = StreamStats(
                    done_reason=done_reason,
                    prompt_eval_count=prompt_tokens,
                    eval_count=eval_tokens,
                    eval_duration_ms=total_ms,
                    total_duration_ms=total_ms,
                )
                log.info(
                    "[gemini] stream done | reason=%s prompt_tokens=%d eval_tokens=%d total_ms=%s",
                    done_reason,
                    prompt_tokens,
                    eval_tokens,
                    total_ms,
                )
        except requests.Timeout:
            log.error("[gemini] stream timed out | model=%s", self.model_name)
            raise
        except requests.RequestException as exc:
            log.error("[gemini] stream request error: %s", exc)
            raise


# Backward compatibility alias
OllamaClient = GeminiClient
