#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt Builder Module
Responsible for building and optimizing prompts for each phase and role

v2:
- Support reloading strategy from templates/prompts/*.json, convenient for future research / evaluation.
- Introduce Path-style positioning, no longer dependent on hardcoded strings, Builder pattern easier to extend.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional


PROMPTS_DIR = Path(os.getcwd()) / "templates" / "prompts"


class PromptBuilder:
    """Prompt builder, responsible for generating structured and optimized prompts"""

    # ---------------------------------------------------------------
    # v2 Reload: Read strategy from JSON file
    # ---------------------------------------------------------------
    _cached_roles: Optional[Dict] = None
    _cached_format: Optional[Dict] = None

    @classmethod
    def reload(cls) -> "PromptBuilder":
        """Reload prompt strategy from disk, decoupled from running agent_roles."""
        cls._cached_roles = cls._read_json(PROMPTS_DIR / "roles.json")
        cls._cached_format = cls._read_json(PROMPTS_DIR / "format_requirements.json")
        return cls

    @staticmethod
    def _read_json(p: Path) -> Dict:
        try:
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    @classmethod
    def roles(cls) -> Dict:
        if cls._cached_roles is None:
            cls._cached_roles = cls._read_json(PROMPTS_DIR / "roles.json")
        return cls._cached_roles

    @classmethod
    def format_requirements(cls) -> Dict:
        if cls._cached_format is None:
            cls._cached_format = cls._read_json(PROMPTS_DIR / "format_requirements.json")
        return cls._cached_format

    @classmethod
    def get_role_phase_guidance(cls, role_name: str, phase: str) -> str:
        """External interface: Get specific behavior guidance for a role in a phase."""
        roles = cls.roles()
        rec = roles.get("roles", {}).get(role_name) or {}
        per_phase = rec.get("phase_specific_guidance", {}) or {}
        g = per_phase.get(phase)
        if not g:
            return ""
        lines = ["# Your Tasks in Current Phase (" + phase + "):"]
        for line in (g if isinstance(g, list) else [g]):
            lines.append(f"- {line}")
        return "\n".join(lines)

    @classmethod
    def get_role_responsibilities(cls, role_name: str) -> List[str]:
        rec = cls.roles().get("roles", {}).get(role_name) or {}
        return list(rec.get("responsibilities") or [])

    # ---------------------------------------------------------------
    # Public components
    # ---------------------------------------------------------------
    @staticmethod
    def build_context_section(context_files_content: str = "") -> str:
        """Build context files section"""
        if not context_files_content:
            return ""
        return f"""
|**Pre-meeting Document Files** (Important Reference):
{context_files_content}
"""

    # ========== LANGUAGE REQUIREMENT ==========
    # This is added to all prompts to ensure English responses
    LANGUAGE_REQUIREMENT = """
# ========== LANGUAGE REQUIREMENT ==========
# **IMPORTANT: You MUST respond in English. All responses, analysis, suggestions, and discussion must be in English.**
# **This is a strict requirement. Do not use any other language in your response.**
# ==========
"""

    @staticmethod
    def build_architect_prompt(
        workflow_info: str,
        phase: str,
        expert_requirements: str,
        context_files_content: str,
        team_boundaries: str,
    ) -> str:
        """Build prompt for Model Architect"""
        context_section = PromptBuilder.build_context_section(context_files_content)

        phase_specific_content = {
            "Problem Analysis": """
Please abstract the problem from the perspective of mathematical modeling competition:
- Define variables/parameters/sets, inputs/outputs, constraint boundaries and objectives
- Clarify key assumptions and data requirements
- Identify sources of uncertainty and quantifiable metrics

|**Key Constraints**:
1. **Must wait for Domain Expert to clearly state modeling requirements before starting architecture work**
2. **Can only architect model according to Domain Expert's requirements, strictly follow Domain Expert's modeling requirements and constraints**
3. **If Domain Expert has not yet provided clear modeling requirements, should proactively ask or wait**
4. **Must return in Markdown format; provide formulas (inline/block) and symbol table where necessary; clear structure and well-organized**
5. **Please carefully read the pre-meeting document files, these files contain important background data and information, and should be used as important reference for modeling**
6. **Think from the perspective of mathematical modeling competition: Is the problem abstraction accurate in reflecting the actual problem? Can it provide clear direction for subsequent modeling? Does it meet the competition's requirements for model reasonableness?**
""",
            "Model Design": """
Please provide an executable mathematical model solution:
- Choose model paradigm (e.g., optimization, differential equations, probabilistic graphical models, game theory, simulation, reinforcement learning, etc.) and explain the rationale
- Provide objective function/constraints/state equations/transition and observation models
- Design solution algorithm and complexity evaluation
- Propose parameter calibration and cross-validation/sensitivity analysis process

|**Key Constraints**:
1. **Must wait for Domain Expert to clearly state modeling requirements before starting architecture work**
2. **Can only architect model according to Domain Expert's requirements, strictly follow Domain Expert's modeling requirements and constraints**
3. **If Domain Expert has not yet provided clear modeling requirements, should proactively ask or wait**
4. **Clearly list model and algorithm key points, provide pseudocode or small-scale numerical examples when necessary**
5. **Please carefully read the pre-meeting document files, these files contain important background data and information, and should be used as important reference for model design**
6. **Think from the perspective of mathematical modeling competition: Is the chosen modeling paradigm suitable for the problem characteristics? Is the method innovative? Is the model reasonable and solvable? Does it meet the competition's evaluation criteria?**
""",
            "Model Building": """
Please evaluate the model solution from dimensions of solvability, robustness, generalization, interpretability, data dependency, engineering implementation cost, etc., provide comparison table and improvement suggestions; point out possible degradation cases and mitigation strategies. Also output implementation roadmap:
- Data preparation and feature engineering
- Parameter calibration
- Solver and hyperparameters
- Experiment stratification
- Visualization and report template
- Online monitoring and feedback loop
Also provide milestones and person-hour estimates.

|**Important Constraints**:
1. **Provide evaluation metrics and acceptance criteria, design ablation/sensitivity experiments when necessary**
2. **Provide clear execution checklist and acceptance criteria for direct implementation**
3. **Please carefully read the pre-meeting document files, these files contain important background data and information, and should be used as important reference for model building**
4. **Think from the perspective of mathematical modeling competition: Does the model accurately reflect the actual problem (model reasonableness)? Is the method innovative? Are the model results accurate (result accuracy)? Is the model reproducible (reproducibility)? Does it provide complete model foundation for paper writing?**
""",
        }

        phase_content = phase_specific_content.get(
            phase,
            "Please participate in discussion with mathematical modeling as the core, focusing on model abstraction, solving, and verification.\n\n**Important Constraints: Use Markdown and formulas, clear structure and executable.**",
        )

        return f"""# ========== LANGUAGE REQUIREMENT ==========
# **IMPORTANT: You MUST respond in English. All responses, analysis, suggestions, and discussion must be in English.**
# **This is a strict requirement. Do not use any other language in your response.**
# ==========

{workflow_info}

{phase_content}

{context_section}

|**Domain Expert's Modeling Requirements**:
{expert_requirements}

{team_boundaries}
"""

    @staticmethod
    def build_critic_prompt(
        workflow_info: str,
        phase: str,
        architect_design: str = "",
        paper_content: str = "",
        team_boundaries: str = "",
    ) -> str:
        """Build prompt for Critic"""
        phase_specific_content = {
            "Model Design": f"""
Please evaluate the model solution proposed by Model Architect from the perspective of evaluation and critique.

|**Model Architect's Model Design**:
{architect_design}

|**Your Responsibilities**:
- Raise critical questions about the solution proposed by Model Architect
- Identify potential risks and problems in the model solution
- Evaluate pros and cons of the model solution (solvability, robustness, generalization, interpretability, data dependency, engineering implementation cost, etc.)
- Point out possible degradation cases and mitigation strategies
- Provide improvement suggestions

|**Absolutely Forbidden**:
- Cannot design new model architecture
- Cannot derive mathematical formulas
- Cannot establish objective function or constraints
- Cannot perform any modeling work
- If Model Architect has not yet proposed a model design, you should wait or ask, not do modeling yourself
""",
            "Model Building": f"""
Please evaluate the model built by Model Architect from the perspective of evaluation and critique.

|**Model Architect's Model Building**:
{architect_design}

|**Your Responsibilities**:
- Evaluate model solution from dimensions of solvability, robustness, generalization, interpretability, data dependency, engineering implementation cost
- Raise critical questions about the model solution
- Identify potential risks and problems in the model solution
- Evaluate pros and cons of the model solution
- Point out possible degradation cases and mitigation strategies
- Provide improvement suggestions
- Evaluate feasibility of implementation roadmap (data preparation, parameter calibration, experiment design, etc.)

|**Absolutely Forbidden**:
- Cannot design new model architecture
- Cannot derive mathematical formulas
- Cannot establish objective function or constraints
- Cannot perform any modeling work
- Cannot output implementation roadmap or execution checklist on behalf of Model Architect
- If Model Architect has not yet built the model, you should wait or ask, not do modeling yourself
""",
            "Paper Writing": f"""
Please evaluate the paper written by Paper Writer from the perspective of evaluation and critique.

|**Paper Writer's Latest Paper Content**:
{paper_content}

|**Your Responsibilities**:
- Raise critical questions and evaluate the paper written by Paper Writer
- Evaluate paper completeness, logic, accuracy, and compliance with standards
- Check if the paper meets mathematical modeling competition paper requirements (complete structure, clear logic, accurate expression, result visualization, standard format)
- Check if the paper meets competition evaluation criteria (model reasonableness, method innovation, result accuracy, paper completeness, reproducibility)
- Identify potential problems and deficiencies in the paper (incomplete structure, unclear logic, inaccurate expression, missing figures/tables, non-standard format, etc.)
- Evaluate whether the paper meets the requirement of complete 25-page paper with substantial content
- Check if the paper's LaTeX format is correct and compilable
- Evaluate academic compliance (citation format, figure/table captions, formula numbering, etc.)
- Provide specific improvement suggestions to help Paper Writer improve the paper

|**Absolutely Forbidden**:
- Cannot write paper
- Cannot modify paper content
- Cannot implement paper writing functionality
- Cannot complete paper writing on behalf of Paper Writer
- If Paper Writer has not yet written the paper, you should wait or ask, not write the paper yourself

|**Evaluation Focus**:
- Is the paper structure complete (contains all required sections)?
- **Does the Abstract clearly specify which model is used for each problem?** (Does it clearly state what model Problem 1 uses, what model Problem 2 uses, what model Problem 3 uses?)
- Is the paper logic clear (forms a complete logical chain from problem analysis to model building to solution verification)?
- Is the paper expression accurate (mathematical language and symbols are accurate, formula derivation is rigorous)?
- **Are mathematical formulas used where they should be?** (Are there corresponding mathematical formula expressions in model building, solution algorithms, result analysis, etc. involving mathematical relationships? Are mathematical relationships described in pure text without formulas?)
- Does the paper showcase results through figures/tables (enhancing readability)?
- Is the paper format standard (LaTeX format is correct, complies with academic standards)?
- Does the paper meet the requirement of complete 25-page paper with substantial content?
- Does the paper meet mathematical modeling competition evaluation criteria?
""",
        }

        phase_content = phase_specific_content.get(
            phase,
            "Please participate in discussion from the perspective of evaluation and critique, focusing on identifying risks, evaluating pros and cons, providing improvement suggestions.\n\n**Important Constraints: Can only evaluate and critique, absolutely cannot do work on behalf of other roles.**",
        )

        return f"""# ========== LANGUAGE REQUIREMENT ==========
# **IMPORTANT: You MUST respond in English. All responses, analysis, suggestions, and discussion must be in English.**
# **This is a strict requirement. Do not use any other language in your response.**
# ==========

{workflow_info}

{phase_content}

{team_boundaries}
"""

    @staticmethod
    def build_paper_writer_prompt(
        workflow_info: str,
        phase: str,
        topic: str,
        discussion_summary: str,
        solution_summary: str,
        team_boundaries: str,
    ) -> str:
        """Build prompt for Paper Writer"""
        latex_template = """
|**The paper must strictly follow the MCM/ICM standard paper template structure, no section can be missing. All content must be written in English**:

1. Cover Page (Abstract + Keywords)
   - Abstract: **IMPORTANT: The abstract must clearly specify which model is used for each problem (Problem 1, Problem 2, Problem 3).** For example: "For Problem 1, we use [model name/type]; for Problem 2, we use [model name/type]; for Problem 3, we use [model name/type]." Summarize the problem, methodology (including model specifications for each problem), and main results
   - Keywords: Relevant keywords for the mathematical modeling competition

2. Table of Contents
   - Complete table of contents with all sections and subsections

3. Introduction (Background + Tasks + Methodology Overview + Flowchart)
   - Background: Context and significance of the problem
   - Tasks: Problem description and objectives
   - Methodology Overview: Brief overview of the approach
   - Flowchart: Visual representation of the overall methodology

4. Modeling Preparation (Assumptions + Notation + Data Preprocessing)
   - Assumptions: Model assumptions
   - Notation: Symbol definitions and notation
   - Data Preprocessing: Data preparation steps

5. Problem 1 Modeling (Method + Model + Results + Figures/Tables)
   - Method: Methodology for Problem 1
   - Model: Mathematical model for Problem 1
   - Results: Results and analysis
   - Figures/Tables: Supporting visualizations

6. Problem 2 Modeling (Method + Model + Results + Figures/Tables)
   - Method: Methodology for Problem 2
   - Model: Mathematical model for Problem 2
   - Results: Results and analysis
   - Figures/Tables: Supporting visualizations

7. Problem 3 Modeling / New Insights (Originality Analysis + Recommendations)
   - Originality Analysis: Discussion of novel contributions
   - Recommendations: Practical recommendations

8. Sensitivity / Robustness Analysis
   - Analysis of how model results change with parameter variations

9. Model Evaluation (Advantages + Disadvantages)
   - Advantages: Strengths of the model
   - Disadvantages: Limitations and weaknesses

10. Memo (Letter to Decision Makers)
    - Executive summary for decision makers

11. References
    - All citations formatted according to academic standards

12. AI Usage Report (if applicable)
    - Documentation of AI tool usage (if any)
"""

        constraints = [
            "**Must use LaTeX format to return complete paper content, complying with MCM/ICM format requirements**",
            "**All content must be written in English**",
            "**【Important】Must output LaTeX code using Markdown code block format (wrapped with ```latex or ```)**",
            (
                "**Output format example:**\n"
                "   ```latex\n"
                "   \\documentclass{article}\n"
                "   \\begin{document}\n"
                "   ...\n"
                "   \\end{document}\n"
                "   ```"
            ),
            "Use standard LaTeX document structure (\\documentclass{{article}}, \\begin{{document}}, \\end{{document}}, etc.)",
            "Use LaTeX commands and syntax, including sections (\\section{{}}, \\subsection{{}}), formulas ($...$, \\[\\]), tables (\\begin{{table}}), figures (\\begin{{figure}}), etc.",
            "Ensure LaTeX code is complete and compilable",
            "**The paper must strictly follow the 12 sections of MCM/ICM standard template, no section can be missing**",
            "**Paper content should be complete, logically clear, and comply with academic standards. Paper logic should form a complete chain from problem analysis to model building to solution verification**",
            "**Use accurate mathematical language and symbols, formula derivation should be rigorous, showcase model results through figures/tables to enhance paper readability**",
            (
                "**【Important】Mathematical formulas must be used where mathematical explanations are needed:**\n"
                "   - Model building process must be expressed with mathematical formulas (objective function, constraints, state equations, transition equations, etc.)\n"
                "   - Solution algorithms and steps must be explained with mathematical formulas (iteration formulas, optimization process, numerical methods, etc.)\n"
                "   - Mathematical relationships in result analysis and discussion must be expressed with formulas (correlation, trends, relationships, etc.)\n"
                "   - Do not describe mathematical relationships in pure text, must use LaTeX mathematical formulas (inline formula $...$ or display formula \\[...\\])\n"
                "   - Ensure all mathematical concepts, models, and algorithms have corresponding mathematical formula expressions"
            ),
            "**Goal: Generate complete 25-page paper. Ensure each section has substantial content, including detailed model analysis, solving process, result presentation, and discussion. Continuously improve the paper through multiple iterations**",
            "**【Important】Problem modeling sections (Problem 1, Problem 2, Problem 3) must account for the largest portion of the paper, this is the core content. In problem modeling sections, content allocation must follow: Problem 1 modeling has the most content (most detailed, most comprehensive), Problem 2 modeling has less content, Problem 3 modeling/new insights has the least content. Ensure Problem 1 modeling section contains the most detailed model building, solving process, result analysis, and figure/table presentation**",
            "**Each round of output must add completely new content on the basis of the previous round. Strictly prohibit copy-paste or repetition of already written content; explicitly list the sections and key points added or modified in this round**",
            "**Think from the perspective of mathematical modeling competition (MCM/ICM): Is the paper structure complete (12 sections)? Is the logic clear? Is the expression accurate? Are results showcased through figures/tables? Is the format standard? Does it meet competition evaluation criteria? Does it meet the requirement of complete 25-page paper? Is all content in English? Do problem modeling sections account for the largest portion? Is content for Problem 1, 2, 3 allocated in decreasing order?**",
        ]
        constraints_text = "\n".join(
            f"{idx + 1}. {c}" for idx, c in enumerate(constraints, 1)
        )
        constraints_section = (
            f"\n|**Important Constraints**：\n{constraints_text}\n"
        )

        iteration_guidance = """
|**Multi-round Iteration Instructions**:
- Paper writing phase will go through multiple iterations (at least 12 rounds, corresponding to 12 sections of MCM template) to gradually improve the paper and meet the 25-page requirement
- Each round corresponds to one section of MCM template, completed in order: Cover Page → Table of Contents → Introduction → Modeling Preparation → Problem 1 Modeling → Problem 2 Modeling → Problem 3 Modeling/New Insights → Sensitivity Analysis → Model Evaluation → Memo → References → AI Usage Report
- **【Important】When writing Cover Page (Abstract + Keywords), the Abstract must clearly specify which model is used for each problem:**
  - Must clearly state: what model Problem 1 uses, what model Problem 2 uses, what model Problem 3 uses
  - Example: "For Problem 1, we use [model name/type]; for Problem 2, we use [model name/type]; for Problem 3, we use [model name/type]."
  - Abstract should summarize problem, methodology (including model specifications for each problem), and main results
- **【Important】Problem modeling sections (Problem 1, Problem 2, Problem 3) are the core of the paper and must account for the largest portion. When writing these three sections, the following content allocation principles must be followed:**
  - **Problem 1 Modeling**: Most content, most detailed, most comprehensive. Should include most detailed model building process, complete solving steps, in-depth result analysis, rich figure/table presentation (at least 3-5 figures/tables), detailed discussion and explanation. Recommended length: 6-8 pages
  - **Problem 2 Modeling**: Less content than Problem 1 but still detailed. Should include complete model building, solving process, result analysis, and figure/table presentation (at least 2-3 figures/tables). Recommended length: 4-5 pages
  - **Problem 3 Modeling/New Insights**: Least content, but should include originality analysis and recommendations. Should include model building, result presentation (at least 1-2 figures/tables), and originality analysis. Recommended length: 2-3 pages
- In the first few rounds, you can first write the main framework and core content of the paper
- In subsequent rounds, you should improve the paper based on Critic's feedback, supplement details, expand content, ensure the paper meets the 25-page requirement
- Each round should output complete LaTeX format paper, even during the improvement process
- **All content must be written in English**
- Pay attention to Critic's feedback, continuously improve paper quality based on feedback
- Each round must clearly state the "New Content in This Round" section, summarizing new or improved parts compared to the previous round
"""

        return f"""# ========== LANGUAGE REQUIREMENT ==========
# **IMPORTANT: You MUST respond in English. All responses, analysis, suggestions, and discussion must be in English.**
# **This is a strict requirement. Do not use any other language in your response.**
# ==========

{workflow_info}

**Important Goal**: The goal of this paper writing is to generate a complete 25-page paper, strictly following the 12 sections of MCM/ICM standard template. You need to gradually improve the paper through multiple iterations, continuously improve based on Critic's feedback, ensure the paper has substantial content, complete structure, clear logic, and finally meets the requirement of complete 25-page paper. **All content must be written in English**.

|**【Core Requirement】Problem Modeling Section Proportion Requirements**:
- Problem modeling sections (Problem 1, Problem 2, Problem 3) must account for the largest portion of the paper, this is the core content
- Problem 1 Modeling: Most content, most detailed (recommended 6-8 pages), including most detailed model building, solving process, result analysis, and rich figures/tables
- Problem 2 Modeling: Less content (recommended 4-5 pages), including complete model building, solving process, and result analysis
- Problem 3 Modeling/New Insights: Least content (recommended 2-3 pages), including model building, result presentation, and originality analysis
- Ensure Problem 1 modeling section contains the most detailed model analysis, solving process, result presentation, and discussion

|**Meeting Topic**: {topic}

|**Discussion Content Summary**:
{discussion_summary}

|**Solution Summary**:
{solution_summary}

{team_boundaries}

{latex_template}

{constraints_section}

{iteration_guidance}

|**Note**: Please directly output complete LaTeX format paper without additional explanation. All content must be written in English.
"""
