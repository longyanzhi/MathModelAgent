#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRunner
===========

Meeting flow state machine executor. Extracts the "phase → agent sequence → decision" logic
from meeting_manager.

Design principles:
- Only responsible for "which node is the next step / whether jump is allowed / whether completion conditions are met".
- Does not directly call LLM (handled by NodeExecutor).
- Does not directly handle SocketIO (handled by injected callbacks from caller).

Node representation:

    GraphNode(phase, agent_name, kind)
    - kind ∈ {"speak", "decision", "merge"}:
        "speak"    -> Execute one LLM call
        "decision" -> Call UserInterventionPolicy.wait_for_phase_decision
        "merge"    -> Trigger phase transition to next phase
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Dict, List, Optional

from utils.logger import logger


class NodeKind(str, Enum):
    SPEAK = "speak"
    DECISION = "decision"
    MERGE = "merge"


@dataclass
class GraphNode:
    """A node in the meeting graph"""
    phase: str
    agent_name: Optional[str]
    kind: NodeKind
    # Additional information for decision nodes
    criteria: Optional[List[str]] = None
    metadata: Dict[str, object] = field(default_factory=dict)


# Short alias
_S = NodeKind.SPEAK
_D = NodeKind.DECISION
_M = NodeKind.MERGE


# Default "Mathematical Modeling" four-phase flow graph, node order consistent with original meeting_manager.
def default_math_model_graph() -> List[GraphNode]:
    return [
        # ---- Problem Analysis ----
        GraphNode("Problem Analysis", "Domain Expert", _S),
        GraphNode("Problem Analysis", "Critic",        _S),
        GraphNode("Problem Analysis", "Project Manager", _S, criteria=["all_agents_spoke"]),
        GraphNode("Problem Analysis", None, _D),
        # ---- Model Design ----
        GraphNode("Model Design", "Domain Expert",    _S),
        GraphNode("Model Design", "Critic",           _S),
        GraphNode("Model Design", "Project Manager",  _S, criteria=["design_clear"]),
        GraphNode("Model Design", None, _D),
        # ---- Model Building ----
        GraphNode("Model Building", "Model Architect", _S),
        GraphNode("Model Building", "Critic",          _S),
        GraphNode("Model Building", "Project Manager", _S, criteria=["model_built"]),
        GraphNode("Model Building", None, _D),
        # ---- Paper Writing ----
        GraphNode("Paper Writing", "Paper Writer",    _S),
        GraphNode("Paper Writing", "Critic",          _S),
        GraphNode("Paper Writing", "Project Manager", _S, criteria=["paper_complete"]),
    ]


class GraphRunner:
    """Interprets and executes GraphNode list.

    Only responsible for "which node is next / whether to continue".
    Actual LLM calls, socket pushes are delegated to caller.
    """

    def __init__(self, nodes: Optional[List[GraphNode]] = None):
        self.nodes = nodes or default_math_model_graph()
        self.cursor = 0
        self._phase_advance_signals: Dict[str, bool] = {}

    # ---------- Iteration interface ----------

    def __iter__(self):
        return self

    def __next__(self) -> GraphNode:
        if self.cursor >= len(self.nodes):
            raise StopIteration
        node = self.nodes[self.cursor]
        self.cursor += 1
        return node

    def has_more(self) -> bool:
        return self.cursor < len(self.nodes)

    # ---------- Phase control ----------

    def signal_phase_advance(self, phase: str):
        self._phase_advance_signals[phase] = True

    def should_jump_to_next_phase(self, current_phase: str) -> bool:
        return self._phase_advance_signals.get(current_phase, False)

    def current_phase(self) -> Optional[str]:
        # Find phase corresponding to cursor position
        if self.cursor == 0:
            return None
        return self.nodes[self.cursor - 1].phase

    def next_agent_for_phase(self, phase: str, agents_lookup: Dict[str, object]) -> List[GraphNode]:
        """Return all unexecuted SPEAK nodes for a phase in order"""
        return [n for n in self.nodes if n.phase == phase and n.kind == _S]


    def all_phases_done(self) -> bool:
        return self.cursor >= len(self.nodes)


def build_minimal_round_graph(num_rounds: int = 3) -> List[GraphNode]:
    """Construct a minimal runnable graph (for testing GraphRunner without LLM dependency)."""
    graph: List[GraphNode] = []
    for r in range(num_rounds):
        phase = f"Round {r + 1}"
        graph.extend([
            GraphNode(phase, "Domain Expert", _S),
            GraphNode(phase, "Critic",        _S),
        ])
    return graph
