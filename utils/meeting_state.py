#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Meeting State Management Module
"""

import threading
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime
from utils.logger import logger


class MeetingStatus(Enum):
    """Meeting status enumeration"""
    IDLE = "idle" # Idle state
    STARTING = "starting" # Starting state
    RUNNING = "running" # Running state
    PAUSED = "paused" # Paused state
    STOPPING = "stopping" # Stopping state
    COMPLETED = "completed" # Completed state
    ERROR = "error" # Error state


class MeetingState:
    """Meeting state management class"""
    
    def __init__(self, session_id: str, topic: str, max_turns: int):
        """
        Initialize meeting state
        :param session_id: Session ID
        :param topic: Meeting topic
        :param max_turns: Maximum number of turns
        """
        self.session_id = session_id # Session ID
        self.topic = topic # Meeting topic
        self.max_turns = max_turns # Maximum turns
        self.status = MeetingStatus.IDLE # Meeting status, initially idle
        self.current_turn = 0
        self.start_time: Optional[datetime] = None # Meeting start time
        self.end_time: Optional[datetime] = None # Meeting end time
        self.error_message: Optional[str] = None # Error message
        self._lock = threading.Lock()  # State lock for thread safety
        self._stop_requested = False  # Stop request flag, initially False
    
    def start(self):
        """Start meeting"""
        with self._lock:  # Lock for thread safety
            if self.status != MeetingStatus.IDLE:
                raise ValueError(f'Meeting status error, cannot start: {self.status}')
            self.status = MeetingStatus.STARTING
            self.start_time = datetime.now()
            self._stop_requested = False
            logger.info(f'Meeting started: {self.topic} (session: {self.session_id})')
    
    def set_running(self):
        """Set to running"""
        with self._lock:
            self.status = MeetingStatus.RUNNING
    
    def set_turn(self, turn: int):
        """Set current turn"""
        with self._lock:
            self.current_turn = turn
    
    def request_stop(self):
        """Request to stop meeting"""
        with self._lock:
            if self.status in [MeetingStatus.COMPLETED, MeetingStatus.ERROR, MeetingStatus.STOPPING]:
                return False
            self._stop_requested = True
            self.status = MeetingStatus.STOPPING
            logger.info(f'Stop requested: {self.topic} (session: {self.session_id})')
            return True
    
    def is_stop_requested(self) -> bool:
        """Check if stop was requested"""
        with self._lock:
            return self._stop_requested
    
    def complete(self):
        """Complete meeting"""
        with self._lock:
            self.status = MeetingStatus.COMPLETED
            self.end_time = datetime.now()
            logger.info(f'Meeting completed: {self.topic} (session: {self.session_id})')
    
    def set_error(self, error_message: str):
        """Set error state"""
        with self._lock:
            self.status = MeetingStatus.ERROR
            self.error_message = error_message
            self.end_time = datetime.now()
            logger.error(f'Meeting error: {self.topic} (session: {self.session_id}) - {error_message}')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        with self._lock:
            return {
                'session_id': self.session_id,
                'topic': self.topic,
                'max_turns': self.max_turns,
                'status': self.status.value,
                'current_turn': self.current_turn,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'error_message': self.error_message
            }


class MeetingStateManager:
    """Meeting state manager"""
    
    def __init__(self, max_concurrent: int = 10):
        """
        Initialize state manager
        :param max_concurrent: Maximum concurrent meetings
        """
        self._states: Dict[str, MeetingState] = {}
        self._lock = threading.Lock()
        self.max_concurrent = max_concurrent
    
    def create_state(self, session_id: str, topic: str, max_turns: int) -> MeetingState:
        """
        Create meeting state
        :param session_id: Session ID
        :param topic: Meeting topic
        :param max_turns: Maximum turns
        :return: Meeting state object
        """
        with self._lock:
            if len(self._states) >= self.max_concurrent:
                raise RuntimeError(f'Maximum concurrent meeting limit reached: {self.max_concurrent}')
            
            if session_id in self._states:
                raise ValueError(f'Meeting state already exists: {session_id}')
            
            state = MeetingState(session_id, topic, max_turns)
            self._states[session_id] = state
            logger.info(f'Created meeting state: {session_id} (current concurrent: {len(self._states)})')
            return state
    
    def get_state(self, session_id: str) -> Optional[MeetingState]:
        """Get meeting state"""
        with self._lock:
            return self._states.get(session_id)
    
    def remove_state(self, session_id: str):
        """Remove meeting state"""
        with self._lock:
            if session_id in self._states:
                del self._states[session_id]
                logger.info(f'Removed meeting state: {session_id} (current concurrent: {len(self._states)})')
    
    def get_all_states(self) -> Dict[str, MeetingState]:
        """Get all meeting states"""
        with self._lock:
            return self._states.copy()
    
    def get_concurrent_count(self) -> int:
        """Get current concurrent meeting count"""
        with self._lock:
            return len(self._states)
