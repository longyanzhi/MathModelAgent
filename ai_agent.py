#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Agent implementation.
Contains the AIAgent class and its related methods.
"""
import time
from typing import List, Dict, Any, Optional, Callable

from models import RoleType, Message
from utils.logger import logger


class AIAgent:
    """AI Agent class."""

    def __init__(self, name: str, role: RoleType, model_config: Dict,
                 conversation_type: str, max_tokens: Optional[int] = None):
        self.name = name
        self.role = role
        self.model_config = model_config
        self.conversation_type = conversation_type
        # Prefer the passed max_tokens; otherwise read from model_config; if neither is provided, follow the model's default configuration.
        self.max_tokens = max_tokens if max_tokens is not None else model_config.get('max_tokens')
        self.memory: List[Dict[str, Any]] = []
        # Optional streaming callback; assigned by NodeExecutor.
        self.stream_callback: Optional[Callable[[str, str, bool], None]] = None

    # ---- Legacy API, kept for backward compatibility ----
    async def generate_response(self, context: List[Message], topic: str) -> str:
        """Call the LLM API to generate a response."""
        prompt = self._build_prompt(context, topic)
        response = await self._call_llm_api(prompt)

        self.memory.append({
            "context": context,
            "response": response,
            "timestamp": time.time(),
        })
        return response

    # ---- New API: used by NodeExecutor ----
    def build_full_prompt(self, context: List[Message], topic: str,
                          per_turn_guidance: str = "",
                          regenerate: bool = False) -> str:
        """Equivalent to _build_prompt, but allows extra per-turn guidance and a regenerate suffix.

        This method is invoked explicitly by utils.node_executor so the manifest can record the exact prompt text.
        """
        base = self._build_prompt(context, topic)
        extras = []
        if per_turn_guidance:
            extras.append(f"\n\n[Per-turn Guidance]\n{per_turn_guidance}")
        if regenerate:
            extras.append(
                "\n\nPlease avoid repeating your previous output. Provide new perspectives, "
                "refine actionable steps, or advance to the next decision point."
            )
        if extras:
            return base + "".join(extras)
        return base

    async def generate_response_with_extras(self, context: List[Message], topic: str,
                                            per_turn_guidance: str = "",
                                            regenerate: bool = False) -> str:
        """Method used by NodeExecutor: equivalent to generate_response, but supports extra guidance and a regenerate hint.

        If self.stream_callback is not None, it will be invoked when the first token arrives / when streaming finishes.
        """
        prompt = self.build_full_prompt(context=context, topic=topic,
                                        per_turn_guidance=per_turn_guidance,
                                        regenerate=regenerate)
        response = await self._call_llm_api_with_stream(prompt)
        self.memory.append({
            "context": context,
            "response": response,
            "timestamp": time.time(),
            "regenerate": regenerate,
        })
        return response

    async def _call_llm_api_with_stream(self, prompt: str) -> str:
        """Add an optional streaming callback hook on top of _call_llm_api."""
        # Delegate to the legacy implementation; if stream_callback is set, currently emit a tick at start/end;
        # streaming itself still requires the model to support stream=True. To avoid breaking the legacy path
        # we use the non-streaming call here, and let NodeExecutor wrap the full response as a single
        # "finished=True" pseudo-stream event.
        response = await self._call_llm_api(prompt)
        if self.stream_callback is not None:
            try:
                self.stream_callback("", response, True)
            except Exception as e:
                logger.warning(f"stream_callback error: {e}")
        return response

    def _build_prompt(self, context: List[Message], topic: str) -> str:
        role_descriptions = self._get_role_descriptions(topic)
        history = "\n".join(f"{msg.role}: {msg.content}" for msg in context)
        return f"{role_descriptions[self.role]}\n\n[Language Requirement] All responses must be in English.\n\nDiscussion History:\n{history}\n\nPlease respond in English:"

    def _get_role_descriptions(self, topic: str) -> Dict[RoleType, str]:
        system_note = """
        ========== System Notice ==========
        **IMPORTANT: This is an AI Agent Collaborative Meeting System**
        - You are an AI agent participating in a collaboration meeting organized by AIs
        - All participants (including you, Project Manager, Domain Expert, Critic, Model Architect, Paper Writer, etc.) are AI agents
        - This meeting system is entirely organized and managed by AIs, designed to solve complex problems through collaboration among multiple AI agents
        - Please participate in the discussion as an AI agent and collaborate with other AI agents to complete the target task
        ============================

        **LANGUAGE REQUIREMENT: You MUST respond in English. All responses, analysis, and discussion must be in English.**
        ============================
        """

        return {
            RoleType.MODERATOR: f"""
            {system_note}

            You are the meeting moderator {self.name}, responsible for guiding the team to solve problems.
            Meeting Topic: {topic}

            Your Core Responsibilities:
            1. Ensure the discussion stays focused on solving the specific problem: {topic}
            2. Control the pace of discussion and prevent over-discussion of details
            3. Drive the team from problem analysis to solution design, and then to decision execution
            4. Summarize key issues and solutions

            Important Principles:
            - Always focus on whether the core problem is being addressed
            - Drive the team to propose concrete, executable solutions
            - Avoid getting stuck in detailed discussions while ignoring the overall goal

            Based on the discussion history below, provide your guiding remarks to ensure the discussion focuses on problem-solving:
            """,

            RoleType.EXPERT: f"""
            {system_note}

            You are {self.name}, a senior expert in this domain.
            Meeting Topic: {topic}

            Your Core Responsibilities:
            1. Deeply analyze the essence and key challenges of the problem
            2. Propose concrete, executable solutions
            3. Provide professional technical advice and implementation steps
            4. Avoid empty theoretical discussions and focus on practical problem-solving

            Important Principles:
            - Proposed solutions should be concrete and actionable
            - Each solution should include clear implementation steps
            - Focus on solving the core problem: {topic}
            - Avoid over-discussing technical details while ignoring the overall solution
            - Please elaborate your analysis and solutions in detail, ensuring content is complete, clear, and executable

            Based on the discussion history, provide professional analysis and solutions for the problem: {topic}
            """,

            RoleType.ANALYST: f"""
            {system_note}

            You are the analyst {self.name}.
            Meeting Topic: {topic}

            Your Core Responsibilities:
            1. Raise critical questions about the problem analysis to help clarify its essence
            2. Evaluate the feasibility and effectiveness of solutions
            3. Analyze the pros, cons, risks, and benefits of solutions
            4. Propose improvement suggestions to help optimize solutions

            Important Principles:
            - Questions should be targeted to help the team better understand the problem
            - Evaluations should be objective, pointing out both strengths and weaknesses
            - Avoid excessive questioning that hinders solution progress
            - Focus on helping the team solve the core problem: {topic}
            - Please elaborate your analysis and evaluation in detail, ensuring content is complete and in-depth

            Based on the discussion content, provide targeted analysis or evaluation:
            """,
        }

    async def _call_llm_api(self, prompt: str) -> str:
        try:
            return await self._call_ai(prompt)
        except Exception as e:
            error_type = type(e).__name__
            error_str = str(e)
            is_401 = "401" in error_str or "AuthenticationError" in error_type or "Invalid Token" in error_str
            is_503 = "503" in error_str or "InternalServerError" in error_type
            is_timeout = "Timeout" in error_type or "timeout" in error_str.lower() or "timed out" in error_str.lower()
            is_service_unavailable = "no available channel" in error_str or "service unavailable" in error_str

            logger.warning(f"{self.name} current model call failed (no model fallback will be attempted), error: {error_str}")

            if is_401:
                error_msg = (
                    f"[{self.name} API call failed - API authentication failed]\n\n"
                    f"Error Type: 401 Authentication Failed\n"
                    f"Reason: {error_str}\n\n"
                    f"Suggestions:\n"
                    f"1. Check whether the API_KEY environment variable is set correctly\n"
                    f"2. Verify the API Token is valid and not expired\n"
                    f"3. Confirm the API Token has sufficient permissions to access the selected model\n"
                )
            elif is_service_unavailable or is_503:
                error_msg = (
                    f"[{self.name} API call failed - Service unavailable]\n\n"
                    f"Error Type: 503 Service Unavailable\n"
                    f"Reason: {error_str}\n\n"
                    f"Suggestions:\n"
                    f"1. Check whether the API service is running normally\n"
                    f"2. Try again later\n"
                )
            elif is_timeout:
                error_msg = (
                    f"[{self.name} API call failed - Request timeout]\n\n"
                    f"Error Type: Request Timeout\n"
                    f"Prompt length (info): {len(prompt)} characters\n\n"
                    f"Suggestions:\n"
                    f"1. Check network connection and API service status\n"
                    f"2. Try again later or increase the model's `api_max_retries`\n"
                )
            else:
                error_msg = (
                    f"[{self.name} API call failed]\n\n"
                    f"Error Message: {error_str}\n\n"
                    f"Suggestions:\n"
                    f"1. Check network connection\n"
                    f"2. Check API configuration (base_url / api_key / model)\n"
                    f"3. Try again later or increase the model's `api_max_retries`\n"
                )

            logger.error(f"{self.name} current model call failed and the fallback logic was halted: {error_msg}", exc_info=True)
            raise RuntimeError(error_msg)

    async def _call_ai(self, prompt: str) -> str:
        from openai import AsyncOpenAI
        from httpx import Timeout
        from config import Config
        from utils.retry import retry_async

        prompt_length = len(prompt)
        read_timeout = Config.API_READ_TIMEOUT

        timeout = Timeout(
            connect=Config.API_CONNECT_TIMEOUT,
            read=read_timeout,
            write=30.0,
            pool=10.0,
        )

        logger.debug(f"{self.name} timeout configuration: prompt_length={prompt_length}, read_timeout={read_timeout}")

        async def _make_api_call():
            from utils.api_client import call_api

            logger.info(
                f"[{self.name}] Calling API: provider={self.model_config.get('provider','?')}, "
                f"model={self.model_config.get('model','?')}, "
                f"prompt_chars={prompt_length}, max_tokens={self.max_tokens if self.max_tokens is not None else 'default'}"
            )

            text = await call_api(
                prompt=prompt,
                model_config=self.model_config,
                max_output_tokens=self.max_tokens,
            )

            logger.debug(f"{self.name} API call succeeded: response_length={len(text)}")
            return text

        api_retry_budget = self.model_config.get("api_max_retries", 0)
        return await retry_async(
            _make_api_call,
            max_retries=api_retry_budget,
            delay=Config.API_RETRY_DELAY,
            exceptions=(Exception,),
        )
