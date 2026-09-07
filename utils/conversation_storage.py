#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conversation Storage Module
Save/load conversations to JSON files for resume capability
"""
import json
import os
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from utils.logger import logger

CONVERSATIONS_DIR = Path(os.getcwd()) / "conversations"
CONVERSATIONS_DIR.mkdir(exist_ok=True)


def save_conversation(
    session_id: str,
    name: str,
    topic: str,
    messages: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Save a conversation to a JSON file"""
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
    filepath = CONVERSATIONS_DIR / filename

    payload = {
        "filename": filename,
        "name": name,
        "session_id": session_id,
        "topic": topic,
        "created_at": datetime.now().isoformat(),
        "messages": messages,
        "metadata": metadata or {},
    }

    filepath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Saved conversation: {filename}, {len(messages)} messages")
    return {"success": True, "filename": filename, "filepath": str(filepath)}


def list_conversations() -> List[Dict[str, Any]]:
    """List all saved conversations"""
    items = []
    for f in CONVERSATIONS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            items.append({
                "filename": f.name,
                "name": data.get("name", f.stem),
                "topic": data.get("topic", ""),
                "created_at": data.get("created_at", ""),
                "message_count": len(data.get("messages", [])),
            })
        except Exception as e:
            logger.warning(f"Failed to read conversation {f}: {e}")
    return sorted(items, key=lambda x: x["created_at"], reverse=True)


def load_conversation(filename: str) -> Optional[Dict[str, Any]]:
    """Load a conversation by filename"""
    safe_name = Path(filename).name
    filepath = CONVERSATIONS_DIR / safe_name
    if not filepath.exists():
        return None
    try:
        return json.loads(filepath.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Failed to load conversation {filename}: {e}")
        return None


def delete_conversation(filename: str) -> bool:
    safe_name = Path(filename).name
    filepath = CONVERSATIONS_DIR / safe_name
    if filepath.exists():
        filepath.unlink()
        logger.info(f"Deleted conversation: {filename}")
        return True
    return False
