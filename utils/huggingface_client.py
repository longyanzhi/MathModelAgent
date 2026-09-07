#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hugging Face Client for Vision and Text Recognition
Supports various Hugging Face models for image understanding and OCR
"""
import base64
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import asyncio

from config import Config
from utils.logger import logger


@dataclass
class HuggingFaceVisionModel:
    """Hugging Face Vision Model configuration"""
    name: str
    description: str
    supports_ocr: bool
    supports_text_recognition: bool
    supports_image_captioning: bool
    supports_document_understanding: bool


# Predefined popular vision models on Hugging Face
HUGGINGFACE_VISION_MODELS = {
    # Image Captioning models
    "Salesforce/blip-image-captioning-large": HuggingFaceVisionModel(
        name="Salesforce/blip-image-captioning-large",
        description="BLIP large image captioning model, excellent at general image understanding",
        supports_ocr=False,
        supports_text_recognition=False,
        supports_image_captioning=True,
        supports_document_understanding=False,
    ),
    "Salesforce/blip-image-captioning-base": HuggingFaceVisionModel(
        name="Salesforce/blip-image-captioning-base",
        description="BLIP base image captioning model, lighter version",
        supports_ocr=False,
        supports_text_recognition=False,
        supports_image_captioning=True,
        supports_document_understanding=False,
    ),
    "nlpconnect/vit-gpt2-image-captioning": HuggingFaceVisionModel(
        name="nlpconnect/vit-gpt2-image-captioning",
        description="ViT-GPT2 image captioning model",
        supports_ocr=False,
        supports_text_recognition=False,
        supports_image_captioning=True,
        supports_document_understanding=False,
    ),
    
    # OCR and Text Recognition models
    "microsoft/trocr-base-handwritten": HuggingFaceVisionModel(
        name="microsoft/trocr-base-handwritten",
        description="TrOCR handwritten text recognition model",
        supports_ocr=True,
        supports_text_recognition=True,
        supports_image_captioning=False,
        supports_document_understanding=False,
    ),
    "microsoft/trocr-base-printed": HuggingFaceVisionModel(
        name="microsoft/trocr-base-printed",
        description="TrOCR printed text recognition model",
        supports_ocr=True,
        supports_text_recognition=True,
        supports_image_captioning=False,
        supports_document_understanding=False,
    ),
    "facebook/bart-large-cnn" : HuggingFaceVisionModel(
        name="facebook/bart-large-cnn",  # Not vision but for text summarization
        description="BART large for text summarization",
        supports_ocr=False,
        supports_text_recognition=False,
        supports_image_captioning=False,
        supports_document_understanding=False,
    ),
    
    # Document Understanding models (including MiniMax-like models)
    "Qwen/Qwen-VL-Max": HuggingFaceVisionModel(
        name="Qwen/Qwen-VL-Max",
        description="Qwen Vision-Language model, supports image understanding, OCR, and document analysis",
        supports_ocr=True,
        supports_text_recognition=True,
        supports_image_captioning=True,
        supports_document_understanding=True,
    ),
    "Qwen/Qwen-VL-Chat": HuggingFaceVisionModel(
        name="Qwen/Qwen-VL-Chat",
        description="Qwen Vision-Language chat model, excellent for multi-round image understanding",
        supports_ocr=True,
        supports_text_recognition=True,
        supports_image_captioning=True,
        supports_document_understanding=True,
    ),
    
    # MiniMax models (if available on Hugging Face)
    "MiniMaxAI/MiniMax-VL-01": HuggingFaceVisionModel(
        name="MiniMaxAI/MiniMax-VL-01",
        description="MiniMax Vision-Language model, supports image understanding and text recognition",
        supports_ocr=True,
        supports_text_recognition=True,
        supports_image_captioning=True,
        supports_document_understanding=True,
    ),
    
    # General vision-language models
    "llava-hf/llava-1.5-7b-hf": HuggingFaceVisionModel(
        name="llava-hf/llava-1.5-7b-hf",
        description="LLaVA 1.5 vision-language model, supports image understanding and conversation",
        supports_ocr=True,
        supports_text_recognition=True,
        supports_image_captioning=True,
        supports_document_understanding=True,
    ),
    "llava-hf/llava-1.6-mistral-7b-hf": HuggingFaceVisionModel(
        name="llava-hf/llava-1.6-mistral-7b-hf",
        description="LLaVA 1.6 with Mistral, higher performance vision-language model",
        supports_ocr=True,
        supports_text_recognition=True,
        supports_image_captioning=True,
        supports_document_understanding=True,
    ),
    "llava-hf/llava-vision-7b-hf": HuggingFaceVisionModel(
        name="llava-hf/llava-vision-7b-hf",
        description="LLaVA vision model",
        supports_ocr=True,
        supports_text_recognition=True,
        supports_image_captioning=True,
        supports_document_understanding=True,
    ),
    
    # Kosmos-2 and other advanced models
    "microsoft/kosmos-2-patch14-224": HuggingFaceVisionModel(
        name="microsoft/kosmos-2-patch14-224",
        description="KOSMOS-2 multimodal model, supports image grounding and understanding",
        supports_ocr=True,
        supports_text_recognition=True,
        supports_image_captioning=True,
        supports_document_understanding=True,
    ),
    
    # IDEFICS series (instruction-following vision models)
    "HuggingFaceM4/idefics2-8b": HuggingFaceVisionModel(
        name="HuggingFaceM4/idefics2-8b",
        description="IDEFICS2 vision-language model, supports complex image understanding",
        supports_ocr=True,
        supports_text_recognition=True,
        supports_image_captioning=True,
        supports_document_understanding=True,
    ),
    
    # Phi-3 Vision
    "microsoft/Phi-3-vision-128k-instruct": HuggingFaceVisionModel(
        name="microsoft/Phi-3-vision-128k-instruct",
        description="Phi-3 Vision instruction model, supports long context image understanding",
        supports_ocr=True,
        supports_text_recognition=True,
        supports_image_captioning=True,
        supports_document_understanding=True,
    ),
    
    # PaliGemma (Google's vision model)
    "google/paligemma-3b-mix-224": HuggingFaceVisionModel(
        name="google/paligemma-3b-mix-224",
        description="PaliGemma vision model, supports image understanding and captioning",
        supports_ocr=True,
        supports_text_recognition=True,
        supports_image_captioning=True,
        supports_document_understanding=True,
    ),
    
    # Florence-2 (Microsoft's unified vision model)
    "microsoft/Florence-2-large": HuggingFaceVisionModel(
        name="microsoft/Florence-2-large",
        description="Florence-2 large model, supports OCR, text recognition, and document understanding",
        supports_ocr=True,
        supports_text_recognition=True,
        supports_image_captioning=True,
        supports_document_understanding=True,
    ),
    "microsoft/Florence-2-base": HuggingFaceVisionModel(
        name="microsoft/Florence-2-base",
        description="Florence-2 base model, supports OCR, text recognition, and document understanding",
        supports_ocr=True,
        supports_text_recognition=True,
        supports_image_captioning=True,
        supports_document_understanding=True,
    ),
}


def get_available_vision_models() -> List[Dict[str, Any]]:
    """Get list of available Hugging Face vision models"""
    return [
        {
            "name": model.name,
            "description": model.description,
            "supports_ocr": model.supports_ocr,
            "supports_text_recognition": model.supports_text_recognition,
            "supports_image_captioning": model.supports_image_captioning,
            "supports_document_understanding": model.supports_document_understanding,
        }
        for model in HUGGINGFACE_VISION_MODELS.values()
    ]


def encode_file_as_base64(file_path: str) -> Optional[str]:
    """Encode file as base64 string"""
    try:
        path = Path(file_path)
        if not path.exists():
            return None
        return base64.b64encode(path.read_bytes()).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to encode file {file_path}: {e}")
        return None


class HuggingFaceClient:
    """
    Hugging Face Client for vision and text recognition.
    Supports both Inference API and local inference.
    """
    
    def __init__(self, api_key: str = None, model_name: str = None):
        """
        Initialize Hugging Face client
        
        Args:
            api_key: Hugging Face API key (if None, uses from Config)
            model_name: Default model name (if None, uses from Config)
        """
        self.api_key = api_key or Config.HF_API_KEY
        self.default_model = model_name or Config.HF_VISION_MODEL
        self.inference_endpoint = Config.HF_INFERENCE_ENDPOINT
        self._client = None
        self._initialized = False
    
    def _ensure_client(self):
        """Lazy initialization of Hugging Face client"""
        if self._initialized:
            return
        
        if not self.api_key:
            logger.warning("Hugging Face API key not configured")
            return
        
        try:
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(
                model=self.default_model,
                token=self.api_key,
                timeout=120,
            )
            self._initialized = True
            logger.info(f"HuggingFace client initialized with model: {self.default_model}")
        except ImportError:
            logger.error(
                "huggingface_hub not installed. Install with: pip install huggingface_hub"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace client: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if Hugging Face client is available"""
        if not self.api_key:
            return False
        try:
            self._ensure_client()
            return self._client is not None
        except Exception:
            return False
    
    async def recognize_image(
        self,
        image_path: str,
        model_name: str = None,
        prompt: str = None,
        max_new_tokens: int = 512,
    ) -> str:
        """
        Recognize image content using Hugging Face vision model
        
        Args:
            image_path: Path to image file
            model_name: Specific model to use (overrides default)
            prompt: Text prompt for the model
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Recognized text content
        """
        if not self.api_key:
            raise ValueError("Hugging Face API key not configured")
        
        self._ensure_client()
        
        model = model_name or self.default_model
        base64_image = encode_file_as_base64(image_path)
        
        if not base64_image:
            raise ValueError(f"Failed to read image file: {image_path}")
        
        default_prompt = prompt or (
            "Please carefully describe the content of this image. "
            "If there is text, formulas, or tables, transcribe them completely. "
            "If it's a mathematical problem, please transcribe all equations and data."
        )
        
        try:
            # Try using Inference API
            from huggingface_hub import InferenceClient
            
            client = InferenceClient(
                model=model,
                token=self.api_key,
                timeout=120,
            )
            
            # Determine the model type and call appropriate method
            if "qwen-vl" in model.lower():
                # Qwen VL models
                result = await self._call_qwen_vl_async(client, base64_image, default_prompt, max_new_tokens)
            elif "llava" in model.lower():
                # LLaVA models
                result = await self._call_llava_async(client, base64_image, default_prompt, max_new_tokens)
            elif "blip" in model.lower():
                # BLIP models (image captioning)
                result = await self._call_blip_async(client, image_path, default_prompt)
            elif "trocr" in model.lower():
                # TrOCR models (OCR)
                result = await self._call_trocr_async(client, image_path, default_prompt)
            elif "florence" in model.lower():
                # Florence models
                result = await self._call_florence_async(client, image_path, default_prompt)
            elif "minimax" in model.lower():
                # MiniMax models
                result = await self._call_minimax_async(client, base64_image, default_prompt, max_new_tokens)
            else:
                # Generic vision-language model
                result = await self._call_generic_vlm_async(client, base64_image, default_prompt, max_new_tokens)
            
            return result
            
        except Exception as e:
            logger.error(f"HuggingFace recognition failed: {e}", exc_info=True)
            raise
    
    async def _call_qwen_vl_async(
        self,
        client,
        base64_image: str,
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        """Call Qwen VL model via API"""
        try:
            from huggingface_hub import InferenceClient
            
            # Use synchronous call in async context
            def _sync_call():
                return client.chat.completions.messages.create(
                    model="Qwen/Qwen-VL-Max",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ],
                    max_tokens=max_new_tokens,
                )
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, _sync_call)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Qwen VL call failed: {e}")
            raise
    
    async def _call_llava_async(
        self,
        client,
        base64_image: str,
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        """Call LLaVA model via API"""
        try:
            from huggingface_hub import InferenceClient
            
            def _sync_call():
                return client.chat.completions.messages.create(
                    model="llava-hf/llava-1.5-7b-hf",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ],
                    max_tokens=max_new_tokens,
                )
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, _sync_call)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLaVA call failed: {e}")
            raise
    
    async def _call_blip_async(
        self,
        client,
        image_path: str,
        prompt: str,
    ) -> str:
        """Call BLIP model for image captioning"""
        try:
            def _sync_call():
                return client.image_to_text(
                    image=image_path,
                    model="Salesforce/blip-image-captioning-large",
                )
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _sync_call)
            return result
        except Exception as e:
            logger.error(f"BLIP call failed: {e}")
            raise
    
    async def _call_trocr_async(
        self,
        client,
        image_path: str,
        prompt: str,
    ) -> str:
        """Call TrOCR model for OCR"""
        try:
            def _sync_call():
                return client.image_to_text(
                    image=image_path,
                    model="microsoft/trocr-base-handwritten",
                )
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _sync_call)
            return result
        except Exception as e:
            logger.error(f"TrOCR call failed: {e}")
            raise
    
    async def _call_florence_async(
        self,
        client,
        image_path: str,
        prompt: str,
    ) -> str:
        """Call Florence model for document understanding"""
        try:
            def _sync_call():
                return client.document_question_answering(
                    image=image_path,
                    question=prompt,
                    model="microsoft/Florence-2-large",
                )
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _sync_call)
            # Florence returns dict with 'answer' key
            if isinstance(result, dict):
                return result.get('answer', str(result))
            return str(result)
        except Exception as e:
            logger.error(f"Florence call failed: {e}")
            raise
    
    async def _call_minimax_async(
        self,
        client,
        base64_image: str,
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        """Call MiniMax VL model via API"""
        try:
            def _sync_call():
                return client.chat.completions.messages.create(
                    model="MiniMaxAI/MiniMax-VL-01",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ],
                    max_tokens=max_new_tokens,
                )
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, _sync_call)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"MiniMax VL call failed: {e}")
            raise
    
    async def _call_generic_vlm_async(
        self,
        client,
        base64_image: str,
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        """Call generic vision-language model via API"""
        try:
            def _sync_call():
                return client.chat.completions.messages.create(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ],
                    max_tokens=max_new_tokens,
                )
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, _sync_call)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Generic VLM call failed: {e}")
            raise


# Global client instance
_hf_client: Optional[HuggingFaceClient] = None


def get_hf_client(api_key: str = None, model_name: str = None) -> HuggingFaceClient:
    """Get global HuggingFace client instance"""
    global _hf_client
    if _hf_client is None:
        _hf_client = HuggingFaceClient(api_key=api_key, model_name=model_name)
    return _hf_client


def is_hf_available() -> bool:
    """Check if Hugging Face client is available"""
    client = get_hf_client()
    return client.is_available()


async def recognize_with_huggingface(
    file_path: str,
    model_name: str = None,
    prompt: str = None,
) -> str:
    """
    Convenience function to recognize image with Hugging Face
    
    Args:
        file_path: Path to image file
        model_name: Hugging Face model name
        prompt: Text prompt
        
    Returns:
        Recognized text
    """
    client = get_hf_client(model_name=model_name)
    return await client.recognize_image(
        image_path=file_path,
        model_name=model_name,
        prompt=prompt,
    )


# ============================================================
# Alternative: Direct HTTP API calls (for models without SDK support)
# ============================================================

async def call_hf_inference_api(
    model_name: str,
    inputs: Dict[str, Any],
    task: str = "image-to-text",
    api_key: str = None,
) -> Any:
    """
    Call Hugging Face Inference API directly via HTTP
    
    Args:
        model_name: Hugging Face model name (e.g., "microsoft/trocr-base-handwritten")
        inputs: Model inputs (e.g., {"image": base64_image_str})
        task: Task type (image-to-text, visual-question-answering, etc.)
        api_key: Hugging Face API key
        
    Returns:
        API response
    """
    import aiohttp
    import json
    
    api_key = api_key or Config.HF_API_KEY
    if not api_key:
        raise ValueError("Hugging Face API key not configured")
    
    url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=inputs, timeout=aiohttp.ClientTimeout(total=120)) as response:
            if response.status == 200:
                return await response.json()
            elif response.status == 503:
                # Model is loading
                result = await response.json()
                raise RuntimeError(f"Model is loading: {result.get('error', 'Please wait and retry')}")
            else:
                error_text = await response.text()
                raise RuntimeError(f"HuggingFace API error {response.status}: {error_text}")


async def recognize_image_via_api(
    file_path: str,
    model_name: str = "microsoft/trocr-base-printed",
    prompt: str = None,
) -> str:
    """
    Recognize image via Hugging Face Inference API
    
    Args:
        file_path: Path to image file
        model_name: Hugging Face model name
        prompt: Optional prompt for VQA tasks
        
    Returns:
        Recognized text
    """
    import aiohttp
    import json
    
    api_key = Config.HF_API_KEY
    if not api_key:
        raise ValueError("Hugging Face API key not configured")
    
    # Encode image
    base64_image = encode_file_as_base64(file_path)
    if not base64_image:
        raise ValueError(f"Failed to read image: {file_path}")
    
    # Determine task type
    if "blip" in model_name.lower() or "captioning" in model_name.lower():
        task_type = "image-to-text"
        inputs = {"inputs": f"data:image/jpeg;base64,{base64_image}"}
    elif "trocr" in model_name.lower() or "ocr" in model_name.lower():
        task_type = "image-to-text"
        inputs = {"inputs": f"data:image/jpeg;base64,{base64_image}"}
    elif "vqa" in model_name.lower() or "visual-question" in model_name.lower():
        task_type = "visual-question-answering"
        inputs = {
            "inputs": {
                "image": f"data:image/jpeg;base64,{base64_image}",
                "question": prompt or "What is in this image?"
            }
        }
    elif "document" in model_name.lower() or "qwen" in model_name.lower() or "llava" in model_name.lower():
        # For document understanding models
        task_type = "document-question-answering"
        inputs = {
            "inputs": {
                "image": f"data:image/jpeg;base64,{base64_image}",
                "question": prompt or "Please describe the content of this image in detail, including any text, formulas, or tables."
            }
        }
    else:
        # Default to image-to-text
        task_type = "image-to-text"
        inputs = {"inputs": f"data:image/jpeg;base64,{base64_image}"}
    
    url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=inputs, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    # Parse result based on task type
                    if isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], dict):
                            return result[0].get('generated_text', str(result[0]))
                        return str(result[0])
                    elif isinstance(result, dict):
                        return result.get('generated_text', result.get('answer', str(result)))
                    return str(result)
                elif resp.status == 503:
                    result = await resp.json()
                    error_msg = result.get('error', 'Model is loading, please wait and retry')
                    raise RuntimeError(f"Model loading: {error_msg}. This may take a few minutes for first-time requests.")
                else:
                    error_text = await resp.text()
                    raise RuntimeError(f"HuggingFace API error {resp.status}: {error_text}")
    except aiohttp.ClientError as e:
        logger.error(f"HTTP request error: {e}")
        raise


def list_popular_vision_models() -> List[Dict[str, str]]:
    """
    Get list of popular vision models for user selection
    """
    return [
        {
            "id": "microsoft/trocr-base-printed",
            "name": "TrOCR (Printed)",
            "description": "Best for printed text recognition/OCR",
            "category": "ocr"
        },
        {
            "id": "microsoft/trocr-base-handwritten",
            "name": "TrOCR (Handwritten)",
            "description": "Best for handwritten text recognition",
            "category": "ocr"
        },
        {
            "id": "Salesforce/blip-image-captioning-large",
            "name": "BLIP Large",
            "description": "High-quality image captioning and understanding",
            "category": "captioning"
        },
        {
            "id": "Qwen/Qwen-VL-Max",
            "name": "Qwen VL Max",
            "description": "Advanced vision-language model, excellent for complex image understanding",
            "category": "vlm"
        },
        {
            "id": "Qwen/Qwen-VL-Chat",
            "name": "Qwen VL Chat",
            "description": "Conversational vision-language model",
            "category": "vlm"
        },
        {
            "id": "MiniMaxAI/MiniMax-VL-01",
            "name": "MiniMax VL",
            "description": "MiniMax vision-language model",
            "category": "vlm"
        },
        {
            "id": "llava-hf/llava-1.5-7b-hf",
            "name": "LLaVA 1.5 7B",
            "description": "Open-source vision-language model",
            "category": "vlm"
        },
        {
            "id": "microsoft/Florence-2-large",
            "name": "Florence 2 Large",
            "description": "Microsoft's unified vision model for OCR and document understanding",
            "category": "document"
        },
        {
            "id": "microsoft/Florence-2-base",
            "name": "Florence 2 Base",
            "description": "Lightweight Microsoft vision model for document understanding",
            "category": "document"
        },
        {
            "id": "microsoft/Phi-3-vision-128k-instruct",
            "name": "Phi-3 Vision",
            "description": "Long-context vision instruction model",
            "category": "vlm"
        },
        {
            "id": "google/paligemma-3b-mix-224",
            "name": "PaliGemma",
            "description": "Google's open vision model",
            "category": "vlm"
        },
        {
            "id": "HuggingFaceM4/idefics2-8b",
            "name": "IDEFICS2 8B",
            "description": "Instruction-following vision-language model",
            "category": "vlm"
        },
    ]
