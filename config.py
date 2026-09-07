#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration management module.
"""
import os
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()


class Config:
    """Application configuration class."""
    HF_ENDPOINT = os.getenv('HF_ENDPOINT', 'https://hf-mirror.com')
    os.environ.setdefault('HF_ENDPOINT', HF_ENDPOINT)

    # Hugging Face API配置
    HF_API_KEY = os.getenv('HF_API_KEY', '')
    HF_INFERENCE_ENDPOINT = os.getenv('HF_INFERENCE_ENDPOINT', '')
    HF_VISION_MODEL = os.getenv('HF_VISION_MODEL', 'Salesforce/blip-image-captioning-large')

    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-change-this-in-production')
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    # The debug reloader on Windows causes SentenceTransformer/httpx to raise
    # [WinError 10038] (WSAENOTSOCK) in subprocesses. We expose an explicit switch
    # that defaults to off.
    WERKZEUG_RELOAD = os.getenv('WERKZEUG_RELOAD', 'false').lower() == 'true'

    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))

    OPENAI_API_KEY = os.getenv('openai_api_key', '')
    OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://www.dmxapi.cn/v1')

    # Meeting turn bounds (used for client-side turn validation)
    MIN_TURNS = int(os.getenv('MIN_TURNS', 1))
    MAX_TURNS = int(os.getenv('MAX_TURNS', 20))
    DEFAULT_TURNS = int(os.getenv('DEFAULT_TURNS', 5))

    MAX_CONCURRENT_MEETINGS = int(os.getenv('MAX_CONCURRENT_MEETINGS', 10))

    API_TIMEOUT = int(os.getenv('API_TIMEOUT', 600))
    API_CONNECT_TIMEOUT = int(os.getenv('API_CONNECT_TIMEOUT', 30))
    API_READ_TIMEOUT = int(os.getenv('API_READ_TIMEOUT', 600))
    API_MAX_RETRIES = int(os.getenv('API_MAX_RETRIES', 8))
    API_RETRY_DELAY = float(os.getenv('API_RETRY_DELAY', 2.0))

    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/app.log')

    LOGS_DIR = os.getenv('LOGS_DIR', 'logs')
    CONTEXT_FILES = os.getenv('MEETING_CONTEXT_FILES', '')
    CONTEXT_FILES_BASE_DIR = os.getenv('MEETING_CONTEXT_BASE', '')

    MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', 100 * 1024 * 1024))
    MAX_FILES = int(os.getenv('MAX_FILES', 6))

    @classmethod
    def validate(cls) -> Dict[str, Any]:
        """Validate configuration."""
        errors = []
        warnings = []
        if not cls.OPENAI_API_KEY:
            errors.append('OPENAI_API_KEY is not set')
        if cls.SECRET_KEY == 'your-secret-key-change-this-in-production':
            warnings.append('SECRET_KEY is using the default value; please change it for production')
        if cls.DEBUG and cls.SECRET_KEY == 'your-secret-key-change-this-in-production':
            warnings.append('Default SECRET_KEY is being used while DEBUG is enabled')
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
        }