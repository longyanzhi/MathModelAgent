#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UserInterventionPolicy
Unified encapsulation of all "user may intervene in meeting" points:
1. Text messages sent by user during meeting (for injecting into meeting context)
2. User confirmation for phase advancement (continue / advance / rollback)
3. User stops meeting
4. User regenerates a node's response

Benefits of unification:
- Business code (meeting_manager / GraphRunner) only interacts with strategy interface,
  no need to handle socket events / asyncio.Event / field states directly.
- Strategy is "replaceable", easy to switch to fully automatic / rule-based in the future.

Design pattern borrowed from Actor Pattern: strategy only exposes await methods, no internal state exposure.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from models import Message
from utils.logger import logger
from utils.manifest import ManifestWriter


@dataclass
class PhaseDecision:
    """User's decision on current phase advancement"""
    advance: bool
    note: str = ""
    decided_at_phase: str = ""
    timestamp: float = 0.0


@dataclass
class UserMessageItem:
    """A message sent by user"""
    content: str
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = 0.0
    delivered: bool = False


class UserInterventionPolicy:
    """Composable user intervention policy collection.

    Usage::

        policy = UserInterventionPolicy(manifest=...)
        policy.attach_user_queue(session_id, session_user_messages_dict)
        result = await policy.wait_for_phase_decision("Problem Analysis", stop_check)
    """

    def __init__(self, manifest: Optional[ManifestWriter] = None):
        self.manifest = manifest
        # Store confirmation events and results for each phase
        self._phase_decision_events: Dict[str, asyncio.Event] = {}
        self._phase_decisions: Dict[str, PhaseDecision] = {}
        # Store user message events for convenient wake-up in wait_for_user_messages
        self._user_msg_event: asyncio.Event = asyncio.Event()
        # Global flag for user-initiated meeting stop
        self._stop_requested: bool = False
        # "Regenerate request queue" for each phase, key=phase, value=list[{agent_name, ...}]
        self._regenerate_requests: List[Dict[str, Any]] = []

    # =========================== User stop ===========================

    def request_stop(self):
        self._stop_requested = True
        logger.info("UserInterventionPolicy: stop requested")
        # Wake up all phase confirmation events to prevent meeting from hanging
        for ev in self._phase_decision_events.values():
            ev.set()
        self._user_msg_event.set()

    def is_stop_requested(self) -> bool:
        return self._stop_requested

    # =========================== User text messages ===========================

    def attach_user_queue(self, session_id: str, user_messages_dict: Dict[str, List[Dict[str, Any]]]):
        """Bind a session → user message queue, allowing strategy to check for new messages"""
        self._session_id = session_id
        self._user_messages_ref = user_messages_dict
        # Initialize as empty list
        user_messages_dict.setdefault(session_id, [])

    def post_user_message(self, session_id: str, content: str):
        """Called by socket handler, put user message into queue and wake up waiters"""
        ref = getattr(self, "_user_messages_ref", None)
        if ref is None:
            logger.warning("UserInterventionPolicy.post_user_message called before attach_user_queue")
            return
        item = UserMessageItem(content=content, timestamp=__import__("time").time())
        ref.setdefault(session_id, []).append({
            "content": item.content,
            "message_id": item.message_id,
            "timestamp": item.timestamp,
        })
        if self.manifest is not None:
            self.manifest.record_user_intervention(
                kind="user_message",
                payload={"session_id": session_id, "content": content[:2000], "mid": item.message_id},
            )
        self._user_msg_event.set()

    async def wait_for_user_messages(
        self,
        session_id: str,
        stop_check: Optional[Callable[[], bool]] = None,
        poll_interval: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """Wait until queue has user messages, then return all of them and clear queue.

        Continues waiting when stop_check is None or returns False.
        """
        ref = getattr(self, "_user_messages_ref", None)
        if ref is None:
            return []
        while True:
            if stop_check and stop_check():
                return []
            queue = ref.get(session_id, [])
            if queue:
                snapshot = list(queue)
                ref[session_id] = []
                self._user_msg_event.clear()
                return snapshot
            try:
                await asyncio.wait_for(self._user_msg_event.wait(), timeout=poll_interval)
            except asyncio.TimeoutError:
                pass
            self._user_msg_event.clear()

    # =========================== Phase decision ===========================

    async def wait_for_phase_decision(
        self,
        phase: str,
        stop_check: Optional[Callable[[], bool]] = None,
        timeout: Optional[float] = None,
        poll_interval: float = 0.3,
    ) -> PhaseDecision:
        """Wait for user's decision on current phase (continue / advance)."""
        event = asyncio.Event()
        self._phase_decision_events[phase] = event
        deadline = None
        start = __import__("time").time()
        while not event.is_set():
            if stop_check and stop_check():
                return PhaseDecision(advance=False, note="stopped", decided_at_phase=phase)
            if timeout is not None:
                deadline = start + timeout
                if __import__("time").time() > deadline:
                    return PhaseDecision(advance=False, note="timeout", decided_at_phase=phase)
            try:
                await asyncio.wait_for(event.wait(), timeout=poll_interval)
            except asyncio.TimeoutError:
                pass
        decision = self._phase_decisions.get(phase) or PhaseDecision(advance=False, decided_at_phase=phase)
        if self.manifest is not None:
            self.manifest.record_user_intervention(
                kind="phase_decision",
                payload={
                    "phase": phase,
                    "advance": decision.advance,
                    "note": decision.note,
                },
            )
        return decision

    def post_phase_decision(self, phase: str, advance: bool, note: str = ""):
        """Called by socket handler, inform user decision on phase."""
        decision = PhaseDecision(
            advance=advance,
            note=note,
            decided_at_phase=phase,
            timestamp=__import__("time").time(),
        )
        self._phase_decisions[phase] = decision
        ev = self._phase_decision_events.get(phase)
        if ev is not None:
            ev.set()

    # =========================== Regenerate a node ===========================

    def request_regenerate(self, agent_name: str, phase: str, message_id: str = ""):
        self._regenerate_requests.append({
            "agent_name": agent_name,
            "phase": phase,
            "message_id": message_id,
            "timestamp": __import__("time").time(),
        })
        if self.manifest is not None:
            self.manifest.record_user_intervention(
                kind="regenerate",
                payload={"agent_name": agent_name, "phase": phase, "message_id": message_id},
            )

    def pop_regenerate(self) -> Optional[Dict[str, Any]]:
        if self._regenerate_requests:
            return self._regenerate_requests.pop(0)
        return None
