#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic data model definitions.
Contains foundational data structures such as enum types and data classes.

v2: Adds parent_id / model_id / branch_id / token_usage / prompt_hash fields
    to support DAG-style meeting visualization, reproducibility, and multi-model comparison.
"""

from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any


class RoleType(Enum):
    """Role type enum."""
    MODERATOR = "Moderator"
    EXPERT = "Expert"
    ANALYST = "Analyst"


@dataclass
class Message:
    """Message data class.

    Legacy fields:
        role:       Speaker role name (Project Manager / Domain Expert / Critic / ...)
        content:    Message body
        timestamp:  Speaking timestamp
        turn:       Round number (globally incrementing)

    v2 additions (to support AIBRANCH / multi-model comparison / reproducibility):
        message_id:  Unique message ID (typically = uuid4().hex)
        parent_id:   Parent message ID; forms the DAG. None means root.
        branch_id:   Branch ID; used when multiple models answer the same parent in parallel.
        model_id:    Actual model invoked (e.g. gemini-2.5-pro)
        provider:    Model provider (openai / gemini / anthropic / ...)
        prompt_hash: SHA256 of the rendered prompt template (for reproducibility / comparison)
        token_usage: Estimated token consumption (input / output) for cost tracking.
        metadata:    Free-form extension fields (stop_reason / latency_ms / branch_reason, etc.)
        extra:       Backward-compat field for legacy code.
    """
    role: str
    content: str
    timestamp: float
    turn: int

    # ---- v2 additions ----
    message_id: str = field(default_factory=lambda: _gen_id())
    parent_id: Optional[str] = None
    branch_id: Optional[str] = None
    model_id: Optional[str] = None
    provider: Optional[str] = None
    prompt_hash: Optional[str] = None
    token_usage: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serializable dict."""
        return asdict(self)

    def short(self) -> str:
        """Short summary for debugging."""
        return (
            f"[{self.message_id[:6]}] {self.role} "
            f"(turn={self.turn}, branch={self.branch_id or '-'})"
        )


def _gen_id() -> str:
    import uuid
    return uuid.uuid4().hex
