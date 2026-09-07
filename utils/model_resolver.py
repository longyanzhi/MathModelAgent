#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Resolver
==============

Single entry point for turning a model name (string) into a model config dict.
Used by team templates where the user picks a model by name from a dropdown.

Resolution order:
1. User-defined custom model (via utils.model_manager)
2. Built-in model in the model_library (treating the name as the actual model id)
3. Fallback to a cheap default text_dialogue model
"""

from __future__ import annotations

from typing import Any, Dict


def resolve_model_config_for_slot(model_name: str = "") -> Dict[str, Any]:
    """Resolve a model name to a model_config dict.

    Falls back gracefully - returns the cheapest available model config
    even if the requested name is unknown or empty.
    """
    from utils.model_manager import get_model_manager
    from model_library import get_model_config_by_category, ModelCategory

    if model_name:
        # 1) User-defined model
        try:
            manager = get_model_manager()
            cfg = manager.get_config_for_agent(model_name)
            if cfg:
                return dict(cfg)
        except Exception:
            pass

        # 2) Built-in model (treat name as the actual model id)
        try:
            cfg = get_model_config_by_category(ModelCategory.TEXT_DIALOGUE, model_name)
            if cfg:
                return dict(cfg)
        except Exception:
            pass

    # 3) Final fallback - cheapest text_dialogue default
    try:
        cfg = get_model_config_by_category(ModelCategory.TEXT_DIALOGUE)
        if cfg:
            return dict(cfg)
    except Exception:
        pass

    # 4) Last-resort hardcoded fallback
    from config import Config
    return {
        "provider": "openai",
        "model": "gemini-2.5-flash",
        "api_max_retries": Config.API_MAX_RETRIES,
    }
