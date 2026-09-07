#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision Handler - Process images and PDFs for multimodal recognition
Supports both OpenAI-compatible API and Hugging Face models
"""
import base64
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from openai import AsyncOpenAI
from utils.logger import logger


def encode_file_as_data_url(file_path: str) -> Optional[str]:
    """Encode file as base64 data URL for OpenAI-compatible vision API"""
    try:
        path = Path(file_path)
        if not path.exists():
            return None
        ext = path.suffix.lower().lstrip('.')
        mime = {
            'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
            'png': 'image/png', 'gif': 'image/gif',
            'webp': 'image/webp', 'bmp': 'image/bmp',
        }.get(ext, 'image/jpeg')

        data = base64.b64encode(path.read_bytes()).decode('utf-8')
        return f"data:{mime};base64,{data}"
    except Exception as e:
        logger.error(f"Failed to encode file {file_path}: {e}")
        return None


def extract_pdf_text_simple(file_path: str, max_chars: int = 8000) -> str:
    """
    Try to extract text from PDF using simple method.
    Note: For full PDF text + images, a real PDF parser is recommended,
    but we keep this minimal for simplicity.
    """
    try:
        # Lazy import to avoid hard dependency
        from PyPDF2 import PdfReader  # type: ignore
        reader = PdfReader(file_path)
        text_parts = []
        total = 0
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if not t.strip():
                continue
            if total + len(t) > max_chars:
                t = t[:max_chars - total]
            text_parts.append(t)
            total += len(t)
            if total >= max_chars:
                break
        return "\n\n".join(text_parts)
    except ImportError:
        return "(PDF text extraction requires PyPDF2, which is not installed. Only image-based vision will be performed.)"
    except Exception as e:
        logger.error(f"Failed to extract PDF text: {e}")
        return f"(Failed to extract PDF text: {e})"


async def recognize_files(
    file_paths: List[str],
    model_config: Dict[str, Any],
    prompt: Optional[str] = None,
    base_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Use vision-capable model to recognize image content (or PDF text).
    Returns {success, results: [{path, content}], error?}
    
    Supports:
    - OpenAI-compatible API (gpt-4o, etc.)
    - Hugging Face Inference API (BLIP, TrOCR, Qwen-VL, LLaVA, Florence, etc.)
    """
    if not file_paths:
        return {"success": False, "error": "No files provided"}

    # Determine which type of vision model to use
    provider = model_config.get("provider", "").lower()
    
    # Check if using Hugging Face
    if provider in ("huggingface", "hf") or model_config.get("use_huggingface"):
        return await _recognize_with_huggingface(file_paths, model_config, prompt)
    
    # Check model name for Hugging Face patterns
    model_name = model_config.get("model", "").lower()
    hf_patterns = ["blip", "trocr", "qwen-vl", "llava", "florence", "minimax", "idefics", "paligemma", "phi-3-vision"]
    if any(pattern in model_name for pattern in hf_patterns):
        return await _recognize_with_huggingface(file_paths, model_config, prompt)
    
    # Default to OpenAI-compatible API
    return await _recognize_with_openai(file_paths, model_config, prompt, base_config)


async def _recognize_with_openai(
    file_paths: List[str],
    model_config: Dict[str, Any],
    prompt: Optional[str] = None,
    base_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Use OpenAI-compatible API for vision recognition
    """
    base_url = model_config.get("base_url") or (base_config or {}).get("OPENAI_BASE_URL", "https://api.qnaigc.com/v1")
    api_key = model_config.get("api_key") or (base_config or {}).get("OPENAI_API_KEY", "")
    if not api_key:
        return {"success": False, "error": "API key not configured for vision model"}

    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=120)
    model_name = model_config.get("model", "gpt-4o")

    default_prompt = (
        "Please carefully recognize the content of the following image/PDF. "
        "Extract the key text, formulas, tables, and data. "
        "If it's a mathematical problem statement or dataset, please transcribe it completely in original language. "
        "Provide a structured summary at the end."
    )
    prompt = prompt or default_prompt

    results = []
    for fp in file_paths:
        try:
            ext = Path(fp).suffix.lower()
            if ext == ".pdf":
                # Use simple text extraction for PDFs
                text = extract_pdf_text_simple(fp)
                user_text = f"{prompt}\n\nPDF file: {Path(fp).name}\nExtracted text content:\n{text}"
                input_value = [
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": user_text}],
                    }
                ]
            else:
                # Image - use base64 inline data
                data_url = encode_file_as_data_url(fp)
                if not data_url:
                    results.append({"path": fp, "error": "Failed to encode"})
                    continue
                input_value = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": data_url},
                        ],
                    }
                ]

            from utils.api_client import call_api

            text_out = await call_api(
                messages=[{"role": "user", "content": prompt}],
                model_config=model_config,
                max_output_tokens=2000,
            )
            
            results.append({"path": fp, "name": Path(fp).name, "content": text_out or ""})
        except Exception as e:
            logger.error(f"Vision recognition failed for {fp}: {e}", exc_info=True)
            results.append({"path": fp, "error": str(e)})

    return {"success": True, "results": results}


async def _recognize_with_huggingface(
    file_paths: List[str],
    model_config: Dict[str, Any],
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Use Hugging Face Inference API for vision recognition
    """
    from utils.huggingface_client import (
        recognize_image_via_api,
        list_popular_vision_models,
        encode_file_as_base64,
    )
    from config import Config
    
    api_key = model_config.get("api_key") or Config.HF_API_KEY
    if not api_key:
        return {"success": False, "error": "Hugging Face API key not configured"}
    
    model_name = model_config.get("model") or Config.HF_VISION_MODEL
    
    default_prompt = (
        "Please carefully recognize the content of this image. "
        "Extract the key text, formulas, tables, and data. "
        "If it's a mathematical problem statement or dataset, please transcribe it completely in original language. "
        "Provide a structured summary at the end."
    )
    prompt = prompt or default_prompt

    results = []
    for fp in file_paths:
        try:
            ext = Path(fp).suffix.lower()
            if ext == ".pdf":
                # For PDFs, extract text first then use text recognition
                text = extract_pdf_text_simple(fp)
                if text and not text.startswith("(PDF"):
                    # Successfully extracted text
                    results.append({
                        "path": fp,
                        "name": Path(fp).name,
                        "content": f"PDF text content:\n{text}"
                    })
                else:
                    # Text extraction failed, try OCR model
                    logger.info(f"PDF text extraction limited, trying OCR for {fp}")
                    # Use OCR model for PDF
                    ocr_result = await _try_ocr_for_pdf(fp, model_name, prompt, api_key)
                    results.append({
                        "path": fp,
                        "name": Path(fp).name,
                        "content": ocr_result
                    })
            else:
                # Image - use Hugging Face API
                try:
                    content = await recognize_image_via_api(
                        file_path=fp,
                        model_name=model_name,
                        prompt=prompt,
                    )
                    results.append({"path": fp, "name": Path(fp).name, "content": content or ""})
                except RuntimeError as e:
                    error_msg = str(e)
                    if "loading" in error_msg.lower() or "please wait" in error_msg.lower():
                        # Model is loading, try alternative
                        logger.warning(f"Model {model_name} is loading, trying alternative model")
                        alt_result = await _try_alternative_model(fp, prompt, api_key)
                        results.append({"path": fp, "name": Path(fp).name, "content": alt_result})
                    else:
                        results.append({"path": fp, "error": error_msg})
                except Exception as e:
                    logger.error(f"HuggingFace recognition failed for {fp}: {e}", exc_info=True)
                    results.append({"path": fp, "error": str(e)})
        except Exception as e:
            logger.error(f"File processing failed for {fp}: {e}", exc_info=True)
            results.append({"path": fp, "error": str(e)})

    return {"success": True, "results": results}


async def _try_ocr_for_pdf(
    pdf_path: str,
    model_name: str,
    prompt: str,
    api_key: str,
) -> str:
    """Try to OCR a PDF page by page"""
    try:
        import subprocess
        import tempfile
        
        # Try to convert PDF to images using pdftoppm (from poppler)
        try:
            # Convert first page of PDF to image
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name
            
            result = subprocess.run(
                ['pdftoppm', '-png', '-f', '1', '-l', '1', '-r', '300', pdf_path, tmp_path.replace('.png', '')],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                # Find the generated PNG file
                png_files = list(Path(tmp_path).parent.glob(f"{Path(tmp_path).stem}*.png"))
                if png_files:
                    from utils.huggingface_client import recognize_image_via_api
                    content = await recognize_image_via_api(
                        file_path=str(png_files[0]),
                        model_name="microsoft/trocr-base-printed",
                        prompt=prompt,
                    )
                    # Clean up
                    for f in png_files:
                        f.unlink(missing_ok=True)
                    Path(tmp_path).unlink(missing_ok=True)
                    return content
            
            # Clean up on failure
            Path(tmp_path).unlink(missing_ok=True)
            
        except FileNotFoundError:
            logger.warning("pdftoppm not found, cannot convert PDF to image for OCR")
        except Exception as e:
            logger.warning(f"PDF to image conversion failed: {e}")
        
        return "(PDF requires OCR but conversion tool not available. Please use image files for better results.)"
        
    except Exception as e:
        logger.error(f"PDF OCR attempt failed: {e}")
        return f"(PDF OCR failed: {e})"


async def _try_alternative_model(
    image_path: str,
    prompt: str,
    api_key: str,
) -> str:
    """Try alternative vision models if primary model fails"""
    from utils.huggingface_client import recognize_image_via_api
    
    # Try models in order of preference
    alternative_models = [
        "microsoft/trocr-base-printed",  # Fast OCR
        "Salesforce/blip-image-captioning-large",  # Good captioning
        "microsoft/Florence-2-base",  # Document understanding
    ]
    
    last_error = None
    for model in alternative_models:
        try:
            logger.info(f"Trying alternative model: {model}")
            content = await recognize_image_via_api(
                file_path=image_path,
                model_name=model,
                prompt=prompt,
            )
            return f"[Used alternative model: {model}]\n\n{content}"
        except Exception as e:
            last_error = str(e)
            logger.warning(f"Alternative model {model} failed: {e}")
            continue
    
    return f"(All vision models failed. Last error: {last_error})"


def get_vision_provider_info() -> Dict[str, Any]:
    """
    Get information about available vision providers
    """
    from config import Config
    from utils.huggingface_client import list_popular_vision_models, is_hf_available
    
    info = {
        "openai_compatible": {
            "available": bool(Config.OPENAI_API_KEY),
            "models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1-mini", "claude-sonnet-4", "gemini-2.5-flash"],
        },
        "huggingface": {
            "available": is_hf_available(),
            "api_key_configured": bool(Config.HF_API_KEY),
            "default_model": Config.HF_VISION_MODEL,
            "popular_models": list_popular_vision_models(),
        }
    }
    
    return info


async def recognize_files_enhanced(
    file_paths: List[str],
    provider: str = "auto",
    model_name: str = None,
    prompt: str = None,
    openai_config: Dict[str, Any] = None,
    hf_config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Enhanced file recognition with explicit provider selection.
    
    Args:
        file_paths: List of file paths to recognize
        provider: "openai", "huggingface", or "auto" (default)
        model_name: Specific model to use (overrides default)
        prompt: Custom prompt for recognition
        openai_config: OpenAI API configuration
        hf_config: Hugging Face configuration
        
    Returns:
        Recognition results
    """
    if provider == "openai":
        config = {
            "provider": "openai",
            "model": model_name or "gpt-4o",
            "api_key": (openai_config or {}).get("api_key"),
            "base_url": (openai_config or {}).get("base_url"),
        }
        return await recognize_files(file_paths, config, prompt, openai_config)
    
    elif provider == "huggingface":
        from config import Config as ConfigClass
        config = {
            "provider": "huggingface",
            "model": model_name or ConfigClass.HF_VISION_MODEL,
            "api_key": (hf_config or {}).get("api_key") or ConfigClass.HF_API_KEY,
        }
        return await recognize_files(file_paths, config, prompt)
    
    else:  # auto
        # Try OpenAI first, then fall back to Hugging Face
        if openai_config and openai_config.get("api_key"):
            config = {
                "provider": "openai",
                "model": model_name or "gpt-4o",
                "api_key": openai_config.get("api_key"),
                "base_url": openai_config.get("base_url"),
            }
            result = await recognize_files(file_paths, config, prompt, openai_config)
            if result.get("success"):
                return result
        
        # Try Hugging Face as fallback
        from config import Config as ConfigClass
        if ConfigClass.HF_API_KEY:
            config = {
                "provider": "huggingface",
                "model": model_name or ConfigClass.HF_VISION_MODEL,
                "api_key": ConfigClass.HF_API_KEY,
            }
            return await recognize_files(file_paths, config, prompt)
        
        return {"success": False, "error": "No vision API configured"}
