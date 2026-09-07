#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution Tracker
Used to record and summarize solutions proposed during meetings
"""

from typing import List, Dict, Any


class SolutionTracker:
    """Solution tracker"""

    def __init__(self, topic: str):
        self.topic = topic
        self.solutions: List[Dict[str, Any]] = []
        self.problems_identified: List[str] = []
        self.action_items: List[str] = []

    def add_solution(self, solution: str, proposer: str, turn: int):
        """Add a solution"""
        self.solutions.append({
            'solution': solution,
            'proposer': proposer,
            'turn': turn,
            'evaluated': False,
        })

    def add_problem(self, problem: str):
        """Add an identified problem"""
        if problem not in self.problems_identified:
            self.problems_identified.append(problem)

    def add_action_item(self, action: str):
        """Add an action item"""
        if action not in self.action_items:
            self.action_items.append(action)

    def get_solutions_summary(self) -> str:
        """Get solution summary"""
        if not self.solutions:
            return "No specific solutions proposed yet"

        summary = "Proposed solutions:\n"
        for i, sol in enumerate(self.solutions, 1):
            summary += f"{i}. {sol['solution'][:100]}... (Proposer: {sol['proposer']})\n"
        return summary

    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        return {
            'solutions_count': len(self.solutions),
            'problems_count': len(self.problems_identified),
            'action_items_count': len(self.action_items),
            'has_solutions': len(self.solutions) > 0,
            'solutions': self.solutions,
        }
