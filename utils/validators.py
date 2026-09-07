#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Input Validation Utility Module
"""

import json
import re
from typing import Dict, Any, Optional, Tuple, List
from config import Config

MAX_CONTEXT_FILES = 6


def validate_topic(topic: str) -> Tuple[bool, Optional[str]]:
    """
    Validate meeting topic
    """
    if not topic:
        return False, 'Meeting topic cannot be empty'

    if not isinstance(topic, str):
        return False, 'Meeting topic must be a string'

    topic = topic.strip()

    if len(topic) == 0:
        return False, 'Meeting topic cannot be empty'

    if len(topic) > 10000:
        return False, 'Meeting topic length cannot exceed 10000 characters'

    dangerous_patterns = [
        r'<script',
        r'javascript:',
        r'on\w+\s*=',
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, topic, re.IGNORECASE):
            return False, 'Meeting topic contains disallowed characters'

    return True, None


def validate_max_turns(max_turns: Any) -> Tuple[bool, Optional[str], Optional[int]]:
    """
    Validate discussion rounds
    """
    try:
        turns = int(max_turns)
    except (ValueError, TypeError):
        return False, 'Discussion rounds must be an integer', None

    if turns < Config.MIN_TURNS:
        return False, f'Discussion rounds cannot be less than {Config.MIN_TURNS}', None

    if turns > Config.MAX_TURNS:
        return False, f'Discussion rounds cannot exceed {Config.MAX_TURNS}', None

    return True, None, turns


def _parse_context_files(value: Any) -> Tuple[bool, Optional[str], Optional[List[Dict[str, Optional[str]]]]]:
    """Parse pre-meeting document file list"""
    if value in (None, "", []):
        return True, None, []

    items = []
    try:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return True, None, []
            if text.startswith('['):
                parsed = json.loads(text)
            else:
                parsed = [item.strip() for item in re.split(r'[\n;,]+', text) if item.strip()]
        elif isinstance(value, list):
            parsed = value
        else:
            return False, 'Pre-meeting document format is incorrect, please provide string or list', None

        for entry in parsed:
            if isinstance(entry, str):
                cleaned = entry.strip()
                if cleaned:
                    items.append({'path': cleaned, 'alias': None})
            elif isinstance(entry, dict):
                path = entry.get('path') or entry.get('file') or entry.get('filepath')
                if not path or not isinstance(path, str) or not path.strip():
                    continue
                alias = entry.get('alias') or entry.get('name') or entry.get('label')
                alias = alias.strip() if isinstance(alias, str) else None
                items.append({'path': path.strip(), 'alias': alias})
            else:
                continue
    except json.JSONDecodeError:
        return False, 'Failed to parse pre-meeting document JSON format', None
    except Exception:
        return False, 'Failed to parse pre-meeting document, please check input format', None

    if len(items) > MAX_CONTEXT_FILES:
        return False, f'Pre-meeting documents support maximum of {MAX_CONTEXT_FILES} files', None

    return True, None, items


def validate_meeting_request(data: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Validate meeting request data
    """
    if not isinstance(data, dict):
        return False, 'Request data format error', None

    topic = data.get('topic', '')
    valid, error = validate_topic(topic)
    if not valid:
        return False, error, None

    max_turns = data.get('max_turns', Config.DEFAULT_TURNS)
    valid, error, validated_turns = validate_max_turns(max_turns)
    if not valid:
        return False, error, None

    valid, error, context_files = _parse_context_files(data.get('context_files'))
    if not valid:
        return False, error, None

    return True, None, {
        'topic': topic.strip(),
        'max_turns': validated_turns,
        'context_files': context_files or [],
    }
