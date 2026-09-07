#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model library management module.

All model IDs are the canonical slugs exposed by OpenRouter
(https://openrouter.ai/models). Use the `provider/model` form exactly as
listed at openrouter.ai/models — OpenRouter forwards the request to the
underlying provider.

API base_url: https://openrouter.ai/api/v1  (OpenAI-compatible format)
Header for OpenRouter is `Authorization: Bearer <OPENAI_API_KEY>`, and we
also send the recommended `HTTP-Referer` and `X-Title` headers.
"""

from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass
from utils.logger import logger


class ModelCategory(Enum):
    """Model category."""
    TEXT_DIALOGUE = "text_dialogue"      # General text dialogue
    DEEP_THINKING = "deep_thinking"      # Requires reasoning
    CODE_PROGRAMMING = "code_programming"
    ACADEMIC_WRITING = "academic_writing"


@dataclass
class ModelInfo:
    """Model information.

    Prices are USD per million tokens (after converting from OpenRouter's
    per-token USD prices), so the values are directly comparable to the
    USD/M-token figures shown on openrouter.ai/models.
    """
    name: str
    provider: str
    input_price: float   # USD / M tokens
    output_price: float  # USD / M tokens
    advantage: str
    category: ModelCategory
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    context_length: int = 0
    supports_reasoning: bool = False
    is_multimodal: bool = False


# Default model used when no specific selection is made.
# Picked to be cheap and capable: google/gemini-3.8-flash.
DEFAULT_MODEL_NAME = "google/gemini-3.8-flash"

# OpenRouter base URL — chat completions endpoint.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _price_usd_per_million(per_token_usd: float) -> float:
    """Convert OpenRouter's per-token USD price into USD per million tokens."""
    return round(float(per_token_usd) * 1_000_000.0, 4)


class ModelLibrary:
    """Model library manager.

    Every model id below has been verified to exist on
    https://openrouter.ai/models (sampled 2026-09). If a model becomes
    unavailable on OpenRouter, just remove or replace its entry here —
    no other code change is required.
    """

    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}
        self._initialize_models()

    # ------------------------------------------------------------------
    # Model catalogue — IDs mirror https://openrouter.ai/models
    # ------------------------------------------------------------------
    def _initialize_models(self):
        # ===== OpenAI =====
        self._add_model(ModelInfo(
            name="openai/gpt-6-astra",
            provider="openai",
            input_price=_price_usd_per_million(0.00001),
            output_price=_price_usd_per_million(0.00005),
            advantage="OpenAI flagship (GPT-6 Astra) — strongest general purpose, very long context",
            category=ModelCategory.TEXT_DIALOGUE,
            context_length=1_050_000,
            supports_reasoning=True,
            is_multimodal=True,
        ))
        self._add_model(ModelInfo(
            name="openai/gpt-6-astra-pro",
            provider="openai",
            input_price=_price_usd_per_million(0.00001),
            output_price=_price_usd_per_million(0.00005),
            advantage="GPT-6 Astra served with reasoning.mode=pro for higher quality on complex tasks",
            category=ModelCategory.DEEP_THINKING,
            context_length=1_050_000,
            supports_reasoning=True,
            is_multimodal=True,
        ))

        # ===== Anthropic (Claude Fable 5.x is the public OpenRouter name) =====
        self._add_model(ModelInfo(
            name="anthropic/claude-fable-5.1",
            provider="anthropic",
            input_price=_price_usd_per_million(0.00001),
            output_price=_price_usd_per_million(0.00005),
            advantage="Anthropic Claude Fable 5.1 — strongest agentic coding & long-horizon workflow",
            category=ModelCategory.TEXT_DIALOGUE,
            context_length=1_000_000,
            supports_reasoning=True,
            is_multimodal=True,
        ))
        # Same model re-categorised for writing/reasoning slots (OpenRouter pricing is identical).
        self._add_model(ModelInfo(
            name="anthropic/claude-fable-5.1",
            provider="anthropic",
            input_price=_price_usd_per_million(0.00001),
            output_price=_price_usd_per_million(0.00005),
            advantage="Claude Fable 5.1 — best-in-class long-form reasoning and academic writing",
            category=ModelCategory.DEEP_THINKING,
            context_length=1_000_000,
            supports_reasoning=True,
            is_multimodal=True,
        ))
        self._add_model(ModelInfo(
            name="anthropic/claude-fable-5.1",
            provider="anthropic",
            input_price=_price_usd_per_million(0.00001),
            output_price=_price_usd_per_million(0.00005),
            advantage="Claude Fable 5.1 — strongest academic writing quality on OpenRouter",
            category=ModelCategory.ACADEMIC_WRITING,
            context_length=1_000_000,
            supports_reasoning=True,
            is_multimodal=True,
        ))

        # ===== Google Gemini =====
        self._add_model(ModelInfo(
            name="google/gemini-3.8-flash",
            provider="google",
            input_price=_price_usd_per_million(0.00000075),
            output_price=_price_usd_per_million(0.00000375),
            advantage="Gemini 3.8 Flash — Google's most intelligent Flash model; fast & cheap",
            category=ModelCategory.TEXT_DIALOGUE,
            context_length=1_048_576,
            supports_reasoning=True,
            is_multimodal=True,
        ))
        self._add_model(ModelInfo(
            name="google/gemini-3.8-flash",
            provider="google",
            input_price=_price_usd_per_million(0.00000075),
            output_price=_price_usd_per_million(0.00000375),
            advantage="Gemini 3.8 Flash — strong reasoning at Flash-tier latency",
            category=ModelCategory.DEEP_THINKING,
            context_length=1_048_576,
            supports_reasoning=True,
            is_multimodal=True,
        ))
        self._add_model(ModelInfo(
            name="google/gemini-3.8-flash",
            provider="google",
            input_price=_price_usd_per_million(0.00000075),
            output_price=_price_usd_per_million(0.00000375),
            advantage="Gemini 3.8 Flash — fast code generation",
            category=ModelCategory.CODE_PROGRAMMING,
            context_length=1_048_576,
            supports_reasoning=True,
            is_multimodal=True,
        ))
        self._add_model(ModelInfo(
            name="google/gemini-3.8-flash",
            provider="google",
            input_price=_price_usd_per_million(0.00000075),
            output_price=_price_usd_per_million(0.00000375),
            advantage="Gemini 3.8 Flash — stable academic output, great cost-performance",
            category=ModelCategory.ACADEMIC_WRITING,
            context_length=1_048_576,
            supports_reasoning=True,
            is_multimodal=True,
        ))

        # ===== Meta (Muse Spark 1.x is OpenRouter's name for Llama 5 family) =====
        self._add_model(ModelInfo(
            name="meta/muse-spark-1.3",
            provider="meta",
            input_price=_price_usd_per_million(0.00000125),
            output_price=_price_usd_per_million(0.00000425),
            advantage="Meta Muse Spark 1.3 — multimodal reasoning, long-running agentic workflows",
            category=ModelCategory.TEXT_DIALOGUE,
            context_length=1_048_576,
            supports_reasoning=True,
            is_multimodal=True,
        ))
        self._add_model(ModelInfo(
            name="meta/muse-spark-1.3",
            provider="meta",
            input_price=_price_usd_per_million(0.00000125),
            output_price=_price_usd_per_million(0.00000425),
            advantage="Meta Muse Spark 1.3 — strong reasoning for agentic and coding tasks",
            category=ModelCategory.DEEP_THINKING,
            context_length=1_048_576,
            supports_reasoning=True,
            is_multimodal=True,
        ))
        self._add_model(ModelInfo(
            name="meta/muse-spark-1.3-contributor",
            provider="meta",
            input_price=_price_usd_per_million(0.0000001),
            output_price=_price_usd_per_million(0.0000002),
            advantage="Meta Muse Spark 1.3 Contributor — ultra-cheap contributor tier",
            category=ModelCategory.TEXT_DIALOGUE,
            context_length=1_048_576,
            supports_reasoning=True,
            is_multimodal=True,
        ))

        # ===== Qwen (Alibaba) =====
        self._add_model(ModelInfo(
            name="qwen/qwen3.8-max-0902",
            provider="qwen",
            input_price=_price_usd_per_million(0.000002),
            output_price=_price_usd_per_million(0.000006),
            advantage="Qwen 3.8 Max (0902 snapshot) — 2.4T MoE, top tier reasoning and code",
            category=ModelCategory.DEEP_THINKING,
            context_length=1_000_000,
            supports_reasoning=True,
            is_multimodal=True,
        ))
        self._add_model(ModelInfo(
            name="qwen/qwen3.8-flash",
            provider="qwen",
            input_price=_price_usd_per_million(0.00000015),
            output_price=_price_usd_per_million(0.00000047),
            advantage="Qwen 3.8 Flash — multimodal reasoning, very cheap",
            category=ModelCategory.TEXT_DIALOGUE,
            context_length=1_000_000,
            supports_reasoning=True,
            is_multimodal=True,
        ))
        self._add_model(ModelInfo(
            name="qwen/qwen3.8-flash",
            provider="qwen",
            input_price=_price_usd_per_million(0.00000015),
            output_price=_price_usd_per_million(0.00000047),
            advantage="Qwen 3.8 Flash — fast code generation at very low cost",
            category=ModelCategory.CODE_PROGRAMMING,
            context_length=1_000_000,
            supports_reasoning=True,
            is_multimodal=True,
        ))

        # ===== Z.ai (GLM) =====
        self._add_model(ModelInfo(
            name="z-ai/glm-5.3-flash",
            provider="z-ai",
            input_price=_price_usd_per_million(0.000000075),
            output_price=_price_usd_per_million(0.00000025),
            advantage="Z.ai GLM 5.3 Flash — efficient coding, long-horizon agent, hybrid attention",
            category=ModelCategory.TEXT_DIALOGUE,
            context_length=1_048_576,
            supports_reasoning=True,
            is_multimodal=True,
        ))
        self._add_model(ModelInfo(
            name="z-ai/glm-5.3-flash",
            provider="z-ai",
            input_price=_price_usd_per_million(0.000000075),
            output_price=_price_usd_per_million(0.00000025),
            advantage="Z.ai GLM 5.3 Flash — fast code generation at extremely low cost",
            category=ModelCategory.CODE_PROGRAMMING,
            context_length=1_048_576,
            supports_reasoning=True,
            is_multimodal=True,
        ))

        # ===== DeepSeek (all verified against OpenRouter API 2026-09) =====
        # V4 Flash — cheapest, multi-snapshot variants
        self._add_model(ModelInfo(
            name="~deepseek/deepseek-v4-flash-latest",
            provider="deepseek",
            input_price=_price_usd_per_million(0.00000005),
            output_price=_price_usd_per_million(0.00000016),
            advantage="DeepSeek V4 Flash Latest (alias) — always latest snapshot, extremely cheap",
            category=ModelCategory.TEXT_DIALOGUE,
            context_length=1_310_720,
            supports_reasoning=False,
            is_multimodal=False,
        ))
        self._add_model(ModelInfo(
            name="deepseek/deepseek-v4-flash-0731",
            provider="deepseek",
            input_price=_price_usd_per_million(0.00000014),
            output_price=_price_usd_per_million(0.00000028),
            advantage="DeepSeek V4 Flash 0731 — stable, very cheap general chat",
            category=ModelCategory.TEXT_DIALOGUE,
            context_length=1_310_720,
            supports_reasoning=False,
            is_multimodal=False,
        ))
        self._add_model(ModelInfo(
            name="deepseek/deepseek-v4-flash",
            provider="deepseek",
            input_price=_price_usd_per_million(0.000000088606),
            output_price=_price_usd_per_million(0.000000177212),
            advantage="DeepSeek V4 Flash (older) — legacy cheap model",
            category=ModelCategory.TEXT_DIALOGUE,
            context_length=1_048_576,
            supports_reasoning=False,
            is_multimodal=False,
        ))
        self._add_model(ModelInfo(
            name="deepseek/deepseek-v4-flash-0731",
            provider="deepseek",
            input_price=_price_usd_per_million(0.00000014),
            output_price=_price_usd_per_million(0.00000028),
            advantage="DeepSeek V4 Flash 0731 — fast code generation",
            category=ModelCategory.CODE_PROGRAMMING,
            context_length=1_310_720,
            supports_reasoning=False,
            is_multimodal=False,
        ))
        # V4 Pro — higher capability
        self._add_model(ModelInfo(
            name="deepseek/deepseek-v4-pro",
            provider="deepseek",
            input_price=_price_usd_per_million(0.00000095526),
            output_price=_price_usd_per_million(0.00000191052),
            advantage="DeepSeek V4 Pro — higher capability than Flash",
            category=ModelCategory.TEXT_DIALOGUE,
            context_length=1_048_576,
            supports_reasoning=False,
            is_multimodal=False,
        ))
        # V3 / V3.2 — stable chat models
        self._add_model(ModelInfo(
            name="deepseek/deepseek-chat",
            provider="deepseek",
            input_price=_price_usd_per_million(0.00000032),
            output_price=_price_usd_per_million(0.00000089),
            advantage="DeepSeek V3 (deepseek-chat) — proven stable chat model",
            category=ModelCategory.TEXT_DIALOGUE,
            context_length=1_048_576,
            supports_reasoning=False,
            is_multimodal=False,
        ))
        self._add_model(ModelInfo(
            name="deepseek/deepseek-v3.2",
            provider="deepseek",
            input_price=_price_usd_per_million(0.000000269),
            output_price=_price_usd_per_million(0.0000004),
            advantage="DeepSeek V3.2 — newer than V3, improved capability",
            category=ModelCategory.TEXT_DIALOGUE,
            context_length=1_048_576,
            supports_reasoning=False,
            is_multimodal=False,
        ))
        # R1 — OpenRouter's reasoning model (reasoning parameter is mandatory=True)
        self._add_model(ModelInfo(
            name="deepseek/deepseek-r1",
            provider="deepseek",
            input_price=_price_usd_per_million(0.0000007),
            output_price=_price_usd_per_million(0.0000025),
            advantage="DeepSeek R1 (deepseek-r1) — top open-source reasoning, chain-of-thought",
            category=ModelCategory.DEEP_THINKING,
            context_length=1_048_576,
            supports_reasoning=True,
            is_multimodal=False,
        ))
        self._add_model(ModelInfo(
            name="deepseek/deepseek-r1-0528",
            provider="deepseek",
            input_price=_price_usd_per_million(0.0000005),
            output_price=_price_usd_per_million(0.00000215),
            advantage="DeepSeek R1 0528 snapshot — slightly cheaper R1 variant",
            category=ModelCategory.DEEP_THINKING,
            context_length=1_048_576,
            supports_reasoning=True,
            is_multimodal=False,
        ))
        # V4 Pro for code
        self._add_model(ModelInfo(
            name="deepseek/deepseek-v4-pro",
            provider="deepseek",
            input_price=_price_usd_per_million(0.00000095526),
            output_price=_price_usd_per_million(0.00000191052),
            advantage="DeepSeek V4 Pro — strong code generation",
            category=ModelCategory.CODE_PROGRAMMING,
            context_length=1_048_576,
            supports_reasoning=False,
            is_multimodal=False,
        ))

        # ===== IBM Granite =====
        self._add_model(ModelInfo(
            name="ibm-granite/granite-4.2-8b",
            provider="ibm-granite",
            input_price=_price_usd_per_million(0.0000001),
            output_price=_price_usd_per_million(0.00000015),
            advantage="IBM Granite 4.2 8B — dense reasoning, math & code, very cheap",
            category=ModelCategory.TEXT_DIALOGUE,
            context_length=131_072,
            supports_reasoning=True,
            is_multimodal=False,
        ))

        # ===== Tencent =====
        self._add_model(ModelInfo(
            name="tencent/hy4-preview",
            provider="tencent",
            input_price=_price_usd_per_million(0.000000834),
            output_price=_price_usd_per_million(0.000002501),
            advantage="Tencent Hy4 preview — MoE coding agent, complex tool-use",
            category=ModelCategory.CODE_PROGRAMMING,
            context_length=1_048_576,
            supports_reasoning=True,
            is_multimodal=False,
        ))

        # ===== Inception (Mercury — diffusion LLM) =====
        self._add_model(ModelInfo(
            name="inception/mercury-2.5-preview",
            provider="inception",
            input_price=_price_usd_per_million(0.00000004),
            output_price=_price_usd_per_million(0.00000015),
            advantage="Inception Mercury 2.5 — diffusion LLM, fastest reasoning model",
            category=ModelCategory.DEEP_THINKING,
            context_length=260_000,
            supports_reasoning=True,
            is_multimodal=False,
        ))

    def _add_model(self, model: ModelInfo):
        # Same model may be registered under multiple categories; use
        # category+name as the unique key so all variants are kept.
        key = f"{model.category.value}:{model.name}"
        self.models[key] = model

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------
    def get_models_by_category(self, category: ModelCategory) -> List[ModelInfo]:
        """Return models in `category`, sorted cheapest-first, ties broken by name."""
        models = [m for m in self.models.values() if m.category == category]
        models.sort(key=lambda x: (x.input_price, x.name))
        return models

    def get_model_config(self, category: ModelCategory, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get model configuration.
        :param category: Model category
        :param model_name: Specific model id. If None, picks the cheapest in the category.
        :return: Model configuration dictionary
        """
        models = self.get_models_by_category(category)
        if not models:
            raise ValueError(f"No models available in category {category.value}")

        target_model_name = model_name or DEFAULT_MODEL_NAME

        if target_model_name:
            # Try exact match in the requested category first.
            for model in models:
                if model.name == target_model_name:
                    return self._model_to_config(model)
            # Fall back to any category with the same id (OpenRouter ids are global).
            for model in self.models.values():
                if model.name == target_model_name:
                    return self._model_to_config(model)
            logger.warning(
                f"Model {target_model_name} not found, using fallback {models[0].name} from category {category.value}"
            )

        return self._model_to_config(models[0])

    def _model_to_config(self, model: ModelInfo) -> Dict[str, Any]:
        """Convert a ModelInfo to a configuration dictionary."""
        from config import Config

        config: Dict[str, Any] = {
            "provider": model.provider,
            "model": model.name,
            "input_price": model.input_price,
            "output_price": model.output_price,
            "advantage": model.advantage,
            "category": model.category.value,
            "api_max_retries": Config.API_MAX_RETRIES,
            "context_length": model.context_length,
            "supports_reasoning": model.supports_reasoning,
            "is_multimodal": model.is_multimodal,
        }

        # Always point at OpenRouter unless the model explicitly overrides.
        config["base_url"] = model.base_url or OPENROUTER_BASE_URL
        if model.api_key:
            config["api_key"] = model.api_key
        return config

    def get_fallback_models(self, category: ModelCategory, current_model_name: str) -> List[Dict[str, Any]]:
        """Fallback model list excluding the current model."""
        models = self.get_models_by_category(category)
        fallback_models = [m for m in models if m.name != current_model_name]
        return [self._model_to_config(m) for m in fallback_models]

    def get_all_models_for_category(self, category: ModelCategory) -> List[Dict[str, Any]]:
        """All model configurations for the specified category (including the current one)."""
        models = self.get_models_by_category(category)
        return [self._model_to_config(m) for m in models]


# Global model library instance
_model_library: Optional[ModelLibrary] = None


def get_model_library() -> ModelLibrary:
    """Get the global model library instance (singleton pattern)."""
    global _model_library
    if _model_library is None:
        _model_library = ModelLibrary()
    return _model_library


def get_model_config_by_category(category: ModelCategory, model_name: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function: get model configuration by category."""
    library = get_model_library()
    return library.get_model_config(category, model_name)


def get_fallback_models(category: ModelCategory, current_model_name: str) -> List[Dict[str, Any]]:
    """Convenience function: get fallback model list."""
    library = get_model_library()
    return library.get_fallback_models(category, current_model_name)
