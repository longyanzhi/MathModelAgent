#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Logging Utility Module
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from config import Config


def setup_logger(name: str = 'meeting_system', log_file: str = None) -> logging.Logger:
    """
    Setup logger
    :param name: Logger name
    :param log_file: Log file path
    :return: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO))
    
    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Create default logger, use logger to write logs to terminal and log file
logger = setup_logger(log_file=Config.LOG_FILE)
