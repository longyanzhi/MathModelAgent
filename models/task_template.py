#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structured task template system.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List
from enum import Enum


class TaskType(Enum):
    """Task type enum."""
    PROBLEM_SOLVING = "Problem Solving"
    ANALYSIS = "Analysis"
    DESIGN = "Design"
    EVALUATION = "Evaluation"
    DECISION = "Decision"


@dataclass
class TaskTemplate:
    """Task template."""
    task_type: TaskType
    background: str = ""
    goal: str = ""
    constraints: List[str] = field(default_factory=list)
    output_format: str = ""
    requirements: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        return {
            'task_type': self.task_type.value,
            'background': self.background,
            'goal': self.goal,
            'constraints': self.constraints,
            'output_format': self.output_format,
            'requirements': self.requirements,
            'context': self.context,
        }
