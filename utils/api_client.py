#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Client Wrapper

Unified interface supporting multiple API formats:
- OpenAI Chat Completions (DMXAPI standard)
- Gemini API
- Claude API

Reference documentation:
- https://doc.dmxapi.cn/fanwei.html (OpenAI compatible)
- https://doc.dmxapi.cn/gemini-chat.html (Gemini)
- https://doc.dmxapi.cn/claude-chat.html (Claude)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI, OpenAI

from config import Config
from utils.logger import logger


# Base URLs for different providers
OPENAI_BASE_URL = "https://www.dmxapi.cn/v1"
GEMINI_BASE_URL = "https://www.dmxapi.cn/v1beta"
CLAUDE_BASE_URL = "https://www.dmxapi.cn/v1"


def _detect_provider(base_url: str, model_name: str) -> str:
    """Detect the upstream provider from base_url (preferred) or model_name.

    OpenRouter is OpenAI-compatible, so when the base_url points at
    openrouter.ai we always use the OpenAI Chat Completions endpoint
    regardless of the underlying model. Otherwise we fall back to name
    sniffing for legacy DMX-API Gemini / Claude endpoints.
    """
    base = (base_url or "").lower()
    if "openrouter.ai" in base:
        return "openai"
    name = model_name.lower()
    if "gemini" in name:
        return "gemini"
    if "claude" in name:
        return "claude"
    return "openai"


def _build_headers(api_key: str) -> Dict[str, str]:
    """Build request headers."""
    return {
        "Authorization": api_key,
        "Content-Type": "application/json",
    }


def _build_gemini_headers(api_key: str) -> Dict[str, str]:
    """Build headers for Gemini API."""
    return {
        "Content-Type": "application/json",
    }


def _build_claude_headers(api_key: str) -> Dict[str, str]:
    """Build headers for Claude API."""
    return {
        "Accept": "application/json",
        "Authorization": api_key,
        "Content-Type": "application/json",
    }


async def call_api(
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    *,
    model_config: Optional[Dict[str, Any]] = None,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    **extra: Any,
) -> str:
    """
    Unified API call supporting OpenAI, Gemini, and Claude formats.

    :param prompt: Single prompt string (used if messages is None)
    :param messages: List of message dicts with role/content keys
    :param model_config: Model configuration dict
    :param max_output_tokens: Max output tokens
    :param temperature: Sampling temperature
    """
    cfg = model_config or {}
    model_name = cfg.get("model", "unknown")
    api_key = cfg.get("api_key") or Config.OPENAI_API_KEY
    base_url = cfg.get("base_url") or Config.OPENAI_BASE_URL
    model_type = _detect_provider(base_url, model_name)

    logger.info(
        f"Calling API: model={model_name}, type={model_type}"
    )

    if model_type == "gemini":
        return await _call_gemini(prompt, messages, model_name, api_key, max_output_tokens, temperature, **extra)
    elif model_type == "claude":
        return await _call_claude(prompt, messages, model_name, api_key, max_output_tokens, temperature, **extra)
    else:
        return await _call_openai(prompt, messages, cfg, model_name, api_key, max_output_tokens, temperature, **extra)


async def _call_openai(
    prompt: Optional[str],
    messages: Optional[List[Dict[str, str]]],
    cfg: Dict[str, Any],
    model_name: str,
    api_key: str,
    max_output_tokens: Optional[int],
    temperature: Optional[float],
    **extra: Any,
) -> str:
    """Call OpenAI-compatible API (Chat Completions)."""
    base_url = cfg.get("base_url") or Config.OPENAI_BASE_URL
    # Normalize base_url
    base_url = base_url.strip().rstrip("/")
    for suffix in ["/responses", "/chat/completions"]:
        if base_url.endswith(suffix):
            base_url = base_url[: -len(suffix)].rstrip("/")
    # Ensure /v1
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=Config.API_READ_TIMEOUT,
        max_retries=0,
    )

    # OpenRouter strongly recommends (and some providers require) attaching
    # HTTP-Referer and X-Title. Auto-attach them when the base URL points
    # at openrouter.ai. Users may still override via `extra` below.
    if "openrouter.ai" in base_url:
        default_headers = {
            "HTTP-Referer": "https://mathmodelhelper.local",
            "X-Title": "MathModelAgent",
        }
    else:
        default_headers = {}

    # Build messages
    if messages:
        chat_messages = messages.copy()
    elif prompt:
        chat_messages = [{"role": "user", "content": prompt}]
    else:
        chat_messages = [{"role": "user", "content": ""}]

    # Handle system message
    for msg in chat_messages:
        if msg.get("role") == "system":
            msg["role"] = "developer"

    kwargs: Dict[str, Any] = {
        "model": model_name,
        "messages": chat_messages,
    }

    if max_output_tokens is not None:
        kwargs["max_tokens"] = max_output_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature

    # Pass through other parameters
    for k, v in extra.items():
        if v is not None:
            kwargs[k] = v

    try:
        # Re-create the client with default_headers only when we actually need them;
        # this avoids mutating a shared client.
        if default_headers:
            client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=Config.API_READ_TIMEOUT,
                max_retries=0,
                default_headers=default_headers,
            )
        response = await client.chat.completions.create(**kwargs)
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content or ""
        return ""
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        raise


async def _call_gemini(
    prompt: Optional[str],
    messages: Optional[List[Dict[str, str]]],
    model_name: str,
    api_key: str,
    max_output_tokens: Optional[int],
    temperature: Optional[float],
    **extra: Any,
) -> str:
    """Call Gemini API."""
    import httpx

    # Build prompt from messages or prompt
    if messages:
        # Convert messages to Gemini format
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                parts.append({"text": content})
            elif role == "assistant":
                parts.append({"text": content})
            # Skip system messages in Gemini format (use generation_config instead)
        text_input = "\n".join(p.get("text", "") for p in parts if "text" in p)
    else:
        text_input = prompt or ""

    # Build request payload
    payload: Dict[str, Any] = {
        "contents": [{
            "role": "user",
            "parts": [{"text": text_input}]
        }]
    }

    # Add generation config
    generation_config: Dict[str, Any] = {}
    if max_output_tokens is not None:
        generation_config["maxOutputTokens"] = max_output_tokens
    if temperature is not None:
        generation_config["temperature"] = temperature
    if generation_config:
        payload["generationConfig"] = generation_config

    # Pass through extra parameters
    for k, v in extra.items():
        if v is not None and k not in payload:
            payload[k] = v

    url = f"{GEMINI_BASE_URL}/models/{model_name}:generateContent?key={api_key}"

    try:
        async with httpx.AsyncClient(timeout=Config.API_READ_TIMEOUT) as client:
            response = await client.post(
                url,
                headers=_build_gemini_headers(api_key),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            # Extract text from response
            candidates = data.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                texts = [p.get("text", "") for p in parts if "text" in p]
                return "".join(texts)
            return ""
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        raise


async def _call_claude(
    prompt: Optional[str],
    messages: Optional[List[Dict[str, str]]],
    model_name: str,
    api_key: str,
    max_output_tokens: Optional[int],
    temperature: Optional[float],
    **extra: Any,
) -> str:
    """Call Claude API."""
    import httpx

    # Build messages
    claude_messages: List[Dict[str, Any]] = []
    system_prompt = ""

    if messages:
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_prompt = content
            else:
                # Claude uses human/assistant instead of user/assistant
                claude_role = "user" if role == "user" else "assistant"
                claude_messages.append({
                    "role": claude_role,
                    "content": content
                })
    else:
        claude_messages = [{"role": "user", "content": prompt or ""}]

    # Build request payload
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": claude_messages,
        "stream": False,
    }

    if system_prompt:
        payload["system"] = system_prompt

    if max_output_tokens is not None:
        payload["max_tokens"] = max_output_tokens
    if temperature is not None:
        payload["temperature"] = temperature

    # Pass through extra parameters
    for k, v in extra.items():
        if v is not None and k not in payload:
            payload[k] = v

    url = f"{CLAUDE_BASE_URL}/messages"

    try:
        async with httpx.AsyncClient(timeout=Config.API_READ_TIMEOUT) as client:
            response = await client.post(
                url,
                headers=_build_claude_headers(api_key),
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            # Extract text from response
            content = data.get("content", [])
            texts = []
            for item in content:
                if item.get("type") == "text":
                    texts.append(item.get("text", ""))
            return "".join(texts)
    except Exception as e:
        logger.error(f"Claude API call failed: {e}")
        raise


def call_api_sync(
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    *,
    model_config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> str:
    """Sync wrapper for call_api."""
    return asyncio.run(call_api(
        prompt, messages,
        model_config=model_config,
        **kwargs
    ))
