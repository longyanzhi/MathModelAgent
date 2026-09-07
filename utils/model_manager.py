#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Management Module
Supports dynamic add/remove/update models and runtime selection
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from utils.logger import logger


MODELS_CONFIG_FILE = Path(os.getcwd()) / "config" / "user_models.json"


@dataclass
class CustomModel:
    """User-defined model configuration"""
    name: str            # Display name, e.g. "my-gpt-4"
    provider: str        # Provider: openai, anthropic, gemini, deepseek, custom
    model: str           # Actual model identifier
    category: str        # text_dialogue / deep_thinking / code_programming / academic_writing / vision
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    max_tokens: Optional[int] = None
    api_max_retries: int = 0
    vision_capable: bool = False
    enabled: bool = True
    description: str = ""

    def to_config(self) -> Dict[str, Any]:
        """Convert to AIAgent consumable config"""
        cfg = {
            "provider": self.provider,
            "model": self.model,
            "category": self.category,
            "api_max_retries": self.api_max_retries,
            "vision_capable": self.vision_capable,
        }
        if self.base_url:
            cfg["base_url"] = self.base_url
        if self.api_key:
            cfg["api_key"] = self.api_key
        if self.max_tokens is not None:
            cfg["max_tokens"] = self.max_tokens
        return cfg


class ModelManager:
    """Singleton model manager - manages user-added models"""

    def __init__(self):
        self.user_models: Dict[str, CustomModel] = {}
        self._load_user_models()

    def _load_user_models(self):
        if MODELS_CONFIG_FILE.exists():
            try:
                data = json.loads(MODELS_CONFIG_FILE.read_text(encoding="utf-8"))
                for m in data.get("models", []):
                    self.user_models[m["name"]] = CustomModel(**m)
                logger.info(f"Loaded {len(self.user_models)} user-defined models")
            except Exception as e:
                logger.error(f"Failed to load user models: {e}")

    def _save_user_models(self):
        try:
            MODELS_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "models": [asdict(m) for m in self.user_models.values()]
            }
            MODELS_CONFIG_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to save user models: {e}")

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {**asdict(m), "is_user_defined": True}
            for m in self.user_models.values()
        ]

    def get_model(self, name: str) -> Optional[CustomModel]:
        return self.user_models.get(name)

    def add_model(self, model_data: Dict[str, Any]) -> CustomModel:
        """Add or update a model"""
        name = model_data.get("name")
        if not name:
            raise ValueError("Model name is required")
        model = CustomModel(
            name=name,
            provider=model_data.get("provider", "openai"),
            model=model_data.get("model", name),
            category=model_data.get("category", "text_dialogue"),
            base_url=model_data.get("base_url"),
            api_key=model_data.get("api_key"),
            max_tokens=model_data.get("max_tokens"),
            api_max_retries=model_data.get("api_max_retries", 0),
            vision_capable=model_data.get("vision_capable", False),
            enabled=model_data.get("enabled", True),
            description=model_data.get("description", "")
        )
        self.user_models[name] = model
        self._save_user_models()
        logger.info(f"Added/Updated model: {name}")
        return model

    def remove_model(self, name: str) -> bool:
        if name in self.user_models:
            del self.user_models[name]
            self._save_user_models()
            logger.info(f"Removed model: {name}")
            return True
        return False

    def get_config_for_agent(self, name: str) -> Optional[Dict[str, Any]]:
        m = self.get_model(name)
        if m and m.enabled:
            return m.to_config()
        return None


_manager_instance: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ModelManager()
    return _manager_instance
