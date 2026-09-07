#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Team Templates
==============

MathModelAgent lets users configure a "team template" for a meeting. Each
team template defines a set of role slots (Project Manager / Domain Expert /
Critic / Model Architect / Paper Writer or their domain-specific equivalents),
the models each slot is allowed to use, and a default model. Built-in presets
cover competition, scientific research and industry consulting. Users can
create custom templates which are persisted to ``data/custom_templates.json``.

This module is intentionally framework-light: it just defines data classes and
provides I/O helpers. Domain dispatch happens in
``models.agent_roles.AgentRoleFactory.create_from_slot``.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_ALLOWED_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-thinking"]


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #

# All phases (used as default when phase_participation is empty)
ALL_PHASES = ["Problem Analysis", "Model Design", "Model Building", "Paper Writing"]


@dataclass
class AgentSlot:
    """One role slot inside a team template."""

    role_name: str
    system_prompt: str = ""
    allowed_models: List[str] = field(default_factory=list)
    default_model: str = ""
    required: bool = True
    weight: float = 1.0
    # Phase participation: which phases this role actively speaks in.
    # Empty list means all phases (the default). Each item must match one of
    # ALL_PHASES. Examples:
    #   - []                    → all phases (the default)
    #   - ["Paper Writing"]     → only Paper Writing
    #   - ["Model Design", "Model Building"] → design & build only
    phase_participation: List[str] = field(default_factory=list)
    # Functional category used for phase routing when multiple EXPERT agents exist.
    # Supported values: "analysis", "design", "build", "write", "general"
    functional_category: str = "general"

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d


@dataclass
class TeamTemplate:
    """A team template (preset or user-defined)."""

    template_id: str
    label: str
    description: str = ""
    applicable_problems: List[str] = field(default_factory=list)
    slots: List[AgentSlot] = field(default_factory=list)
    is_builtin: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "TeamTemplate":
        raw_slots = d.get("slots", []) or []
        slots: List[AgentSlot] = []
        for s in raw_slots:
            if not isinstance(s, dict):
                continue
            # Back-compat: legacy JSON files (saved before phase_participation /
            # functional_category existed) lack these keys. Inject defaults so
            # deserialisation never crashes.
            s.setdefault("system_prompt", "")
            s.setdefault("phase_participation", [])
            s.setdefault("functional_category", "general")
            s.setdefault("allowed_models", [])
            s.setdefault("default_model", "")
            s.setdefault("required", True)
            s.setdefault("weight", 1.0)
            slots.append(AgentSlot(**s))
        return cls(
            template_id=d["template_id"],
            label=d.get("label", ""),
            description=d.get("description", ""),
            applicable_problems=list(d.get("applicable_problems", [])),
            slots=slots,
            is_builtin=bool(d.get("is_builtin", False)),
        )


# --------------------------------------------------------------------------- #
# Built-in preset factories
# --------------------------------------------------------------------------- #

def _competition_slots() -> List[AgentSlot]:
    """MCM/ICM competition preset - keeps the historical 5-role structure."""
    return [
        AgentSlot(
            role_name="Project Manager",
            system_prompt="",
            allowed_models=list(DEFAULT_ALLOWED_MODELS),
            default_model="gemini-2.5-flash",
            phase_participation=[],        # all phases
            functional_category="general",
        ),
        AgentSlot(
            role_name="Domain Expert",
            system_prompt="",
            allowed_models=list(DEFAULT_ALLOWED_MODELS),
            default_model="gemini-2.5-flash-thinking",
            phase_participation=["Problem Analysis", "Model Design"],  # analysis + design
            functional_category="analysis",
        ),
        AgentSlot(
            role_name="Critic",
            system_prompt="",
            allowed_models=list(DEFAULT_ALLOWED_MODELS),
            default_model="gemini-2.5-flash-thinking",
            phase_participation=[],        # all phases
            functional_category="general",
        ),
        AgentSlot(
            role_name="Model Architect",
            system_prompt="",
            allowed_models=list(DEFAULT_ALLOWED_MODELS),
            default_model="gemini-2.5-flash",
            phase_participation=["Model Design", "Model Building"],  # design + build
            functional_category="build",
        ),
        AgentSlot(
            role_name="Paper Writer",
            system_prompt="",
            allowed_models=list(DEFAULT_ALLOWED_MODELS),
            default_model="gemini-2.5-flash",
            phase_participation=["Paper Writing"],  # write only
            functional_category="write",
        ),
    ]


def _scientific_research_slots() -> List[AgentSlot]:
    """Research-oriented preset - emphasises reproducibility and method critique."""
    return [
        AgentSlot(
            role_name="PI",
            system_prompt="",
            allowed_models=list(DEFAULT_ALLOWED_MODELS),
            default_model="gemini-2.5-flash",
            phase_participation=[],        # all phases
            functional_category="general",
        ),
        AgentSlot(
            role_name="Senior Researcher",
            system_prompt="",
            allowed_models=list(DEFAULT_ALLOWED_MODELS),
            default_model="gemini-2.5-flash-thinking",
            phase_participation=["Problem Analysis", "Model Design"],  # analysis + design
            functional_category="analysis",
        ),
        AgentSlot(
            role_name="Methodologist",
            system_prompt="",
            allowed_models=list(DEFAULT_ALLOWED_MODELS),
            default_model="gemini-2.5-flash-thinking",
            phase_participation=[],        # all phases (covers critique across phases)
            functional_category="general",
        ),
        AgentSlot(
            role_name="Solution Architect",
            system_prompt="",
            allowed_models=list(DEFAULT_ALLOWED_MODELS),
            default_model="gemini-2.5-flash",
            phase_participation=["Model Design", "Model Building"],  # design + build
            functional_category="build",
        ),
        AgentSlot(
            role_name="Report Writer",
            system_prompt="",
            allowed_models=list(DEFAULT_ALLOWED_MODELS),
            default_model="gemini-2.5-flash",
            phase_participation=["Paper Writing"],  # write only
            functional_category="write",
        ),
    ]


def _industry_slots() -> List[AgentSlot]:
    """Industry consulting preset - emphasises risks and decision support."""
    return [
        AgentSlot(
            role_name="Engagement Lead",
            system_prompt="",
            allowed_models=list(DEFAULT_ALLOWED_MODELS),
            default_model="gemini-2.5-flash",
            phase_participation=[],        # all phases
            functional_category="general",
        ),
        AgentSlot(
            role_name="Subject Matter Expert",
            system_prompt="",
            allowed_models=list(DEFAULT_ALLOWED_MODELS),
            default_model="gemini-2.5-flash-thinking",
            phase_participation=["Problem Analysis", "Model Design"],  # analysis + design
            functional_category="analysis",
        ),
        AgentSlot(
            role_name="Risk Reviewer",
            system_prompt="",
            allowed_models=list(DEFAULT_ALLOWED_MODELS),
            default_model="gemini-2.5-flash-thinking",
            phase_participation=[],        # all phases (covers risk review across phases)
            functional_category="general",
        ),
        AgentSlot(
            role_name="Solution Architect",
            system_prompt="",
            allowed_models=list(DEFAULT_ALLOWED_MODELS),
            default_model="gemini-2.5-flash",
            phase_participation=["Model Design", "Model Building"],  # design + build
            functional_category="build",
        ),
        AgentSlot(
            role_name="Report Writer",
            system_prompt="",
            allowed_models=list(DEFAULT_ALLOWED_MODELS),
            default_model="gemini-2.5-flash",
            phase_participation=["Paper Writing"],  # write only
            functional_category="write",
        ),
    ]


PRESET_TEAMS: List[TeamTemplate] = [
    TeamTemplate(
        template_id="competition",
        label="Competition Team (MCM/ICM)",
        description=(
            "Five-role team aligned with the MCM/ICM competition workflow: "
            "Project Manager, Domain Expert, Critic, Model Architect and "
            "Paper Writer. Output uses the MCM/ICM 12-section LaTeX template."
        ),
        applicable_problems=["competition"],
        slots=_competition_slots(),
        is_builtin=True,
    ),
    TeamTemplate(
        template_id="scientific_research",
        label="Research Team",
        description=(
            "Five-role team for academic / scientific research problems. "
            "Emphasises reproducibility, method critique and rigorous reporting. "
            "Roles: PI, Senior Researcher, Methodologist, Solution Architect, "
            "Report Writer."
        ),
        applicable_problems=["scientific_research", "general"],
        slots=_scientific_research_slots(),
        is_builtin=True,
    ),
    TeamTemplate(
        template_id="industry",
        label="Industry Consulting Team",
        description=(
            "Five-role team for industry / policy consulting problems. "
            "Emphasises risks, stakeholder analysis and decision support. "
            "Roles: Engagement Lead, Subject Matter Expert, Risk Reviewer, "
            "Solution Architect, Report Writer."
        ),
        applicable_problems=["industry", "general"],
        slots=_industry_slots(),
        is_builtin=True,
    ),
]


# --------------------------------------------------------------------------- #
# Custom template I/O
# --------------------------------------------------------------------------- #

def get_custom_templates_path() -> Path:
    """Path to ``data/custom_templates.json`` (created on first save)."""
    return Path(os.getcwd()) / "data" / "custom_templates.json"


def load_custom_templates() -> List[TeamTemplate]:
    """Load user-defined custom templates from disk. Returns [] on any error."""
    path = get_custom_templates_path()
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        raw = payload.get("templates", []) if isinstance(payload, dict) else []
        return [TeamTemplate.from_dict(item) for item in raw]
    except Exception:
        # Corrupt file - skip rather than crashing the meeting start path
        return []


def save_custom_templates(templates: List[TeamTemplate]) -> Path:
    """Persist user-defined templates (overwrites file). Returns the path."""
    path = get_custom_templates_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"templates": [t.to_dict() for t in templates]}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def add_custom_template(template: TeamTemplate) -> TeamTemplate:
    """Add a custom template (force ``is_builtin=False`` and assign a fresh id)."""
    builtin_ids = {t.template_id for t in PRESET_TEAMS}
    existing = load_custom_templates()
    existing_ids = {t.template_id for t in existing}

    template.is_builtin = False
    if not template.template_id or template.template_id in builtin_ids or template.template_id in existing_ids:
        template.template_id = f"custom_{uuid.uuid4().hex[:8]}"

    existing.append(template)
    save_custom_templates(existing)
    return template


def delete_custom_template(template_id: str) -> bool:
    """Delete a custom template by id. Returns True on success."""
    builtin_ids = {t.template_id for t in PRESET_TEAMS}
    if template_id in builtin_ids:
        return False
    existing = load_custom_templates()
    kept = [t for t in existing if t.template_id != template_id]
    if len(kept) == len(existing):
        return False
    save_custom_templates(kept)
    return True


def list_all_templates() -> List[TeamTemplate]:
    """Return all templates (builtins first, then custom)."""
    return list(PRESET_TEAMS) + load_custom_templates()


def find_template(template_id: str) -> Optional[TeamTemplate]:
    """Look up a template by id across builtin + custom. Returns None if missing."""
    for t in PRESET_TEAMS:
        if t.template_id == template_id:
            return t
    for t in load_custom_templates():
        if t.template_id == template_id:
            return t
    return None
