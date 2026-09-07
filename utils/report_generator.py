#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Report Generator
Handles final report generation
"""

import time
from typing import List, Dict, Any

from models import RoleType, Message
from ai_agent import AIAgent


class ReportGenerator:
    """Generate final report"""

    def __init__(self, topic: str, conversation: List[Message], solution_tracker,
                 context_files_content: str = ""):
        self.topic = topic
        self.conversation = conversation
        self.solution_tracker = solution_tracker
        self.context_files_content = context_files_content

    async def generate_final_report(self) -> str:
        """Generate final report"""
        report_prompt = self._build_report_prompt()
        report_agent = AIAgent(
            "Report Organizer",
            RoleType.EXPERT,
            {"provider": "gpt", "model": "gpt-oss-20b"},
            conversation_type='decision',
            max_tokens=8000  # Word limit removed, allowing complete report output
        )
        final_report = await report_agent.generate_response(
            [Message("system", report_prompt, time.time(), 0)],
            self.topic
        )
        return final_report

    def _build_report_prompt(self) -> str:
        """Build report generation prompt"""
        # Add pre-meeting document file content (ensure report generation can also access files)
        context_files_section = ""
        if self.context_files_content:
            context_files_section = f"\n\n{self.context_files_content}\n\n**Important**: Please carefully read the above pre-meeting document file content. These files contain important background data and information, and should be used as important reference for report generation."

        solution_summary = ""
        if self.solution_tracker:
            solution_summary = self.solution_tracker.get_solutions_summary()

        return f"""
        Please generate a problem-solving-oriented final report based on the following complete meeting discussion:

        Meeting Topic: {self.topic}

        Discussion Content:
        {self._format_conversation_for_report()}

        Solution Tracking:
        {solution_summary}
        {context_files_section}

        Please organize the report according to the following structure:
        1. Problem Overview
        2. Solution
        3. Action Plan
        4. Follow-up

        Report Requirements:
        - Focus on problem-solving, highlight executable solutions and actions
        - Professional, clear, and actionable
        - Ensure the report helps to actually solve the problem: {self.topic}
        - Please elaborate in detail, ensuring content is complete and comprehensive, no word limit
        """

    def _format_conversation_for_report(self) -> str:
        """Format conversation content for report"""
        formatted = []
        for msg in self.conversation:
            formatted.append(f"{msg.role} (Round {msg.turn + 1}): {msg.content}")
        return "\n".join(formatted)
