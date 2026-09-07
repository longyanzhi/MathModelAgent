#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replication Manifest Module
=========================

Responsible for generating a manifest at the start of each meeting, recording all "external / environment" parameters
that can affect results, so that a meeting can be fully replicated based on that manifest.

Follows the principles of AIbranch paper regarding replication manifest:
- Explicitly list prompt template version, models used, hyperparameters
- Preserve Agent configuration snapshot (no longer dependent on config.py at that time)
- Persist with conversation

File structure::

    runs/
      <run_id>/
        manifest.json      # Written by this module
        conversation.json  # Written by conversation_storage
        prompts/<hash>.txt # Optional: actual prompt text used for each node

Does not depend on any specific application logic, can be imported independently.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import logger


# Default manifest and conversation stored in the same root directory
DEFAULT_RUNS_ROOT = Path(os.getcwd()) / "runs"


@dataclass
class PromptTemplateRef:
    """A reference to a used prompt template"""
    path: Optional[str] = None       # Template path relative to repo (if any)
    version: Optional[str] = None    # Custom version number
    text_sha256: Optional[str] = None
    body: Optional[str] = None       # Full prompt text (optional, kept if not too large)

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update((self.path or "").encode("utf-8"))
        h.update(b"\0")
        h.update((self.version or "").encode("utf-8"))
        h.update(b"\0")
        h.update((self.text_sha256 or "").encode("utf-8"))
        return h.hexdigest()[:16]


@dataclass
class AgentSpec:
    """Configuration snapshot of an agent in the meeting (no longer affected by config.py at that time)"""
    name: str
    role: str
    model: Optional[str] = None
    provider: Optional[str] = None
    base_url: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Manifest:
    run_id: str
    created_at: str
    system: Dict[str, Any]
    topic: str
    max_turns: Optional[int]
    phase_turns: Dict[str, int]
    selected_models: Dict[str, Any]
    agents: List[AgentSpec] = field(default_factory=list)
    prompt_templates: Dict[str, PromptTemplateRef] = field(default_factory=dict)
    user_interventions: List[Dict[str, Any]] = field(default_factory=list)
    env: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
    team_template_id: Optional[str] = None
    team_template_label: Optional[str] = None

    # Convenient for UI display
    def short_id(self) -> str:
        return self.run_id[:8]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


class ManifestWriter:
    """Maintains a meeting's manifest and writes to disk asynchronously."""

    def __init__(
        self,
        topic: str,
        max_turns: Optional[int],
        phase_turns: Dict[str, int],
        selected_models: Optional[Dict[str, Any]],
        agents: Optional[List[AgentSpec]] = None,
        runs_root: Optional[Path] = None,
        team_template_id: Optional[str] = None,
        team_template_label: Optional[str] = None,
    ):
        self.run_id = uuid.uuid4().hex
        self.topic = topic
        self.max_turns = max_turns
        self.phase_turns = phase_turns
        self.selected_models = selected_models or {}

        self.manifest = Manifest(
            run_id=self.run_id,
            created_at=datetime.utcnow().isoformat() + "Z",
            system={
                "app": "MathModelAgent",
                "schema_version": "1.0",
            },
            topic=topic,
            max_turns=max_turns,
            phase_turns=phase_turns,
            selected_models=selected_models or {},
            agents=list(agents or []),
            team_template_id=team_template_id,
            team_template_label=team_template_label,
        )
        self._collect_system_info()
        self._runs_root = Path(runs_root) if runs_root else DEFAULT_RUNS_ROOT


    def add_agent(self, agent: AgentSpec):
        self.manifest.agents.append(agent)

    def add_prompt_template(self, key: str, body: str, *, path: Optional[str] = None,
                            version: Optional[str] = None):
        sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
        ref = PromptTemplateRef(
            path=path,
            version=version,
            text_sha256=sha,
            body=body if len(body) <= 200_000 else None,  # Store only sha if too large
        )
        self.manifest.prompt_templates[key] = ref

    def has_prompt_template(self, sha: str) -> bool:
        return any(t.text_sha256 == sha for t in self.manifest.prompt_templates.values())

    def record_user_intervention(self, kind: str, payload: Dict[str, Any]):
        """Record a user intervention (send message / phase confirmation / stop)"""
        self.manifest.user_interventions.append({
            "kind": kind,
            "at": datetime.utcnow().isoformat() + "Z",
            **payload,
        })

    def update_extra(self, **kwargs):
        self.manifest.extra.update(kwargs)

    def run_dir(self) -> Path:
        return self._runs_root / self.run_id

    def save(self) -> Path:
        d = self.run_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / "manifest.json"
        path.write_text(
            json.dumps(self.manifest.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Manifest saved: {path}")
        return path

    def _collect_system_info(self):
        info = {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
        }
        try:
            from config import Config  # Avoid circular import, get at runtime
            info["openai_base_url"] = getattr(Config, "OPENAI_BASE_URL", None)
            info["api_read_timeout"] = getattr(Config, "API_READ_TIMEOUT", None)
            info["api_retry_delay"] = getattr(Config, "API_RETRY_DELAY", None)
        except Exception:
            pass
        self.manifest.env = info


def fingerprint_prompt(body: str) -> str:
    """Generate fingerprint for a prompt, used by Message.prompt_hash / Manifest"""
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
