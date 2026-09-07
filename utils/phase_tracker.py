#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase Progress Tracker
Used to track discussion progress for each phase, avoid unnecessary repetition
"""

import difflib
from typing import Dict, Any


class PhaseProgressTracker:
    """Track per-phase discussion progress to avoid unnecessary repetition."""

    def __init__(self):
        self.phase_targets = {
            "Problem Analysis": 3.0,
            "Model Design": 4.0,
            "Model Building": 3.5,
            "Paper Writing": 3.0,
        }
        self.phase_keywords = {
            "Problem Analysis": ["problem", "challenge", "difficulty", "obstacle", "risk", "bottleneck", "root cause", "key"],
            "Model Design": ["model", "design", "solution", "method", "strategy", "architecture", "framework", "paradigm"],
            "Model Building": ["build", "implement", "construct", "develop", "encode", "code", "execute", "deploy"],
            "Paper Writing": ["paper", "write", "writing", "document", "report", "LaTeX", "section", "references"],
        }
        self.redundancy_limit = 2
        self.phase_state: Dict[str, Dict[str, Any]] = {}

    def _ensure_phase(self, phase: str):
        if phase not in self.phase_state:
            self.phase_state[phase] = {
                "score": 0.0,
                "turns": 0,
                "redundancy": 0,
                "seen_keywords": set(),
                "recent_contents": [],
                "last_gain": 0.0,
            }

    def start_phase(self, phase: str):
        self.phase_state[phase] = {
            "score": 0.0,
            "turns": 0,
            "redundancy": 0,
            "seen_keywords": set(),
            "recent_contents": [],
            "last_gain": 0.0,
        }

    def register_turn(self, phase: str):
        if phase not in self.phase_targets:
            return
        self._ensure_phase(phase)
        self.phase_state[phase]["turns"] += 1

    def dynamic_min_turns(self, phase: str) -> int:
        if phase not in self.phase_targets:
            return 1
        self._ensure_phase(phase)
        # Paper writing phase requires at least 6 turns to ensure complete 25-page paper
        if phase == "Paper Writing":
            return 6
        base = 2
        score = self.phase_state[phase]["score"]
        target = self.phase_targets[phase]
        if score >= target * 0.6:
            return 1
        return base

    def note_message(self, phase: str, agent_name: str, content: str):
        if phase not in self.phase_targets:
            return
        self._ensure_phase(phase)
        state = self.phase_state[phase]
        content_lower = content.lower()
        keywords = self.phase_keywords.get(phase, [])

        new_hits = 0
        keyword_hits = 0
        for kw in keywords:
            if kw in content_lower:
                keyword_hits += 1
                if kw not in state["seen_keywords"]:
                    new_hits += 1
                    state["seen_keywords"].add(kw)

        gain = 0.0
        if new_hits:
            gain += new_hits * 1.0
        elif keyword_hits:
            gain += 0.2

        content_length = len(content)
        if content_length >= 220:
            gain += 0.5
        elif content_length >= 160:
            gain += 0.3
        elif content_length >= 120:
            gain += 0.1

        similarity = 0.0
        for prev in state["recent_contents"]:
            similarity = max(similarity, difflib.SequenceMatcher(None, prev, content).ratio())

        if similarity >= 0.9:
            state["redundancy"] += 1
            gain -= 0.6
        elif similarity >= 0.85:
            state["redundancy"] += 1
            gain -= 0.3
        else:
            state["redundancy"] = max(0, state["redundancy"] - 1)

        state["recent_contents"].append(content)
        if len(state["recent_contents"]) > 6:
            state["recent_contents"].pop(0)

        state["score"] += gain
        state["last_gain"] = gain

    def should_advance(self, phase: str, current_turn: int) -> bool:
        if phase not in self.phase_targets:
            return False
        self._ensure_phase(phase)

        min_turns = self.dynamic_min_turns(phase)
        if current_turn < min_turns:
            return False

        state = self.phase_state[phase]
        target = self.phase_targets[phase]

        if state["score"] >= target:
            return True

        if state["redundancy"] >= self.redundancy_limit:
            return True

        return False

    def get_state(self, phase: str) -> Dict[str, Any]:
        return self.phase_state.get(phase, {})
