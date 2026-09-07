#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Retry Utility Module
"""

import asyncio
import time
from typing import Callable, Any, Optional
from functools import wraps
from config import Config
from utils.logger import logger


async def retry_async(
    func: Callable,
    max_retries: int = None,
    delay: float = None,
    backoff: float = 1.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable] = None
) -> Any:
    """
    Async function retry decorator
    :param func: Async function to retry
    :param max_retries: Maximum number of retries
    :param delay: Initial delay time (seconds)
    :param backoff: Backoff multiplier
    :param exceptions: Exception types to retry
    :param on_retry: Callback function on retry
    :return: Function execution result
    """
    if max_retries is None:
        max_retries = Config.API_MAX_RETRIES
    if delay is None:
        delay = Config.API_RETRY_DELAY
    
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except exceptions as e:
            last_exception = e
            
            if attempt < max_retries:
                wait_time = delay * (backoff ** attempt)
                logger.warning(
                    f'Function {func.__name__} execution failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}'
                    f', retrying in {wait_time:.2f} seconds'
                )
                
                if on_retry:
                    on_retry(attempt + 1, e)
                
                await asyncio.sleep(wait_time)
            else:
                logger.error(f'Function {func.__name__} failed after {max_retries} retries: {str(e)}')
                raise
    
    if last_exception:
        raise last_exception


def retry_sync(
    func: Callable,
    max_retries: int = None,
    delay: float = None,
    backoff: float = 1.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable] = None
) -> Any:
    """
    Sync function retry decorator
    :param func: Sync function to retry
    :param max_retries: Maximum number of retries
    :param delay: Initial delay time (seconds)
    :param backoff: Backoff multiplier
    :param exceptions: Exception types to retry
    :param on_retry: Callback function on retry
    :return: Function execution result
    """
    if max_retries is None:
        max_retries = Config.API_MAX_RETRIES
    if delay is None:
        delay = Config.API_RETRY_DELAY
    
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            
            if attempt < max_retries:
                wait_time = delay * (backoff ** attempt)
                logger.warning(
                    f'Function {func.__name__} execution failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}'
                    f', retrying in {wait_time:.2f} seconds'
                )
                
                if on_retry:
                    on_retry(attempt + 1, e)
                
                time.sleep(wait_time)
            else:
                logger.error(f'Function {func.__name__} failed after {max_retries} retries: {str(e)}')
                raise
    
    if last_exception:
        raise last_exception
