#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NodeExecutor
============

Package "one LLM call + retry + streaming callback + manifest recording" into a minimal unit.

Design goals:
- Fully decoupled from meeting_manager: business only needs to provide prompt, it returns a string
  (or pushes streaming fragments asynchronously via stream_callback)
- Auto-record: call duration, estimated tokens, prompt hash, model ID
- Provide regenerate interface for UI to do "get another answer" or "regenerate"

Note: This module does not depend on any meeting_manager / phase concept, only depends on ai_agent.AIAgent,
config, utils.logger, utils.manifest.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ai_agent import AIAgent
from models import Message
from utils.logger import logger
from utils.manifest import ManifestWriter, fingerprint_prompt


# Streaming callback signature: (delta, full_text, finished, message_id) -> None
StreamCallback = Optional[Callable[[str, str, bool, str], None]]


@dataclass
class NodeExecutionResult:
    """Result of one node execution, convenient for upper layer to do persistence / coloring / billing"""
    message: Message
    prompt: str
    duration_ms: int
    finished: bool = True
    error: Optional[str] = None


def _estimate_tokens(s: str) -> int:
    """Rough token estimation. English ~4 characters ≈ 1 token."""
    return max(1, len(s) // 4)


class NodeExecutor:
    """Thin wrapper for single LLM call.

    Usage::

        executor = NodeExecutor(manifest=manifest_writer)
        result  = await executor.run(
            agent=critic,
            context=[...],
            topic="...",
            branch_reason="multi-model-compare",
        )
        print(result.message.content)

    For streaming::

        executor.run(
            agent=critic, ...,
            stream_callback=lambda d, f, fin, mid: socketio.emit("stream_chunk", ...)
        )
    """

    def __init__(
        self,
        manifest: Optional[ManifestWriter] = None,
        default_stream_callback: StreamCallback = None,
    ):
        self.manifest = manifest
        self.default_stream_callback = default_stream_callback

    # ---------------------------------------------------------------

    async def run(
        self,
        agent: AIAgent,
        context: List[Message],
        topic: str,
        *,
        parent_id: Optional[str] = None,
        branch_id: Optional[str] = None,
        per_turn_guidance: str = "",
        branch_reason: str = "",
        regenerate: bool = False,
        stream_callback: StreamCallback = None,
    ) -> NodeExecutionResult:
        """Call an agent once and produce one Message.

        Parameters
        ----------
        agent          : AIAgent instance (with bound model_config)
        context        : Assembled context messages
        topic          : Global topic (used to build system prompt)
        parent_id      : Parent node ID in DAG
        branch_id      : Parallel branch ID (same parent can have multiple)
        per_turn_guidance: Temporary guidance appended to system prompt
        regenerate     : Whether to try regenerating (with "avoid repetition" suffix)
        """
        # 1) Prepare prompt (here we reuse AIAgent internal _build_prompt;
        #    To record prompt_hash, we assemble separately once more.
        #    Because _build_prompt is private, but signatures match, local recalculation)
        prompt = self._build_recorded_prompt(agent, context, topic, per_turn_guidance, regenerate)

        # 2) Prepare streaming callback (for ai_agent internal hook)
        callback = stream_callback or self.default_stream_callback
        if callback is not None:
            agent.stream_callback = self._wrap_stream(callback)

        # 3) Actual call
        mid = uuid.uuid4().hex
        t0 = time.time()
        try:
            runner = getattr(agent, "generate_response_with_extras", None)
            if callable(runner):
                response = await runner(
                    context=context,
                    topic=topic,
                    per_turn_guidance=per_turn_guidance,
                    regenerate=regenerate,
                )
            else:
                # Backward compatible with old agent
                if per_turn_guidance or regenerate:
                    logger.debug(
                        "Agent missing generate_response_with_extras; "
                        "per_turn_guidance / regenerate will be ignored."
                    )
                response = await agent.generate_response(context, topic)
            error_text: Optional[str] = None
        except Exception as e:
            logger.error(f"NodeExecutor failed for agent={agent.name}: {e}", exc_info=True)
            response = ""
            error_text = str(e)
        duration_ms = int((time.time() - t0) * 1000)

        # 4) Calculate token estimation
        in_tokens = _estimate_tokens(prompt)
        out_tokens = _estimate_tokens(response)

        message = Message(
            role=agent.name,
            content=response,
            timestamp=time.time(),
            turn=context[-1].turn + 1 if context else 0,
            message_id=mid,
            parent_id=parent_id,
            branch_id=branch_id,
            model_id=agent.model_config.get("model"),
            provider=agent.model_config.get("provider"),
            prompt_hash=fingerprint_prompt(prompt),
            token_usage={"input": in_tokens, "output": out_tokens},
            metadata={
                "duration_ms": duration_ms,
                "branch_reason": branch_reason,
                "regenerate": regenerate,
                "error": error_text,
            },
        )

        if self.manifest is not None:
            # prompt hash already recorded on message via fingerprint_prompt,
            # here backup first-seen prompt template to manifest
            self.manifest.add_prompt_template(
                key=f"{agent.name}:{message.prompt_hash}",
                body=prompt,
            )

        return NodeExecutionResult(
            message=message,
            prompt=prompt,
            duration_ms=duration_ms,
            finished=error_text is None,
            error=error_text,
        )

    # ---------------------------------------------------------------

    def _build_recorded_prompt(
        self,
        agent: AIAgent,
        context: List[Message],
        topic: str,
        per_turn_guidance: str,
        regenerate: bool,
    ) -> str:
        """Consistent logic with AIAgent._build_prompt; additionally appends per_turn_guidance.

        Reason for not directly calling agent._build_prompt is that we need to externally observe
        the complete prompt predictably (future can also count tokens, integrate context engineering).
        """
        # Reuse existing assembly logic
        return agent.build_full_prompt(context=context, topic=topic, per_turn_guidance=per_turn_guidance)

    def _wrap_stream(self, cb: StreamCallback):
        """Wrap (delta, full, finished) three-parameter callback into what AIAgent expects."""
        message_id_holder = {"mid": uuid.uuid4().hex}

        def wrapped(delta: str, full_text: str, finished: bool):
            try:
                cb(delta, full_text, finished, message_id_holder["mid"])
            except Exception as e:
                logger.warning(f"NodeExecutor stream callback error: {e}")

        return wrapped
