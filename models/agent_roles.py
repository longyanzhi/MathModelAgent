#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI role definitions.
"""

from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


class AgentRoleType(Enum):
    """AI role types."""
    PROJECT_MANAGER = "Project Manager"
    DOMAIN_EXPERT_1 = "Domain Expert"
    CRITIC = "Critic"
    ARCHITECT = "Model Architect"
    PAPER_WRITER = "Paper Writer"


@dataclass
class AgentRole:
    """AI role."""
    role_type: AgentRoleType
    name: str
    description: str
    responsibilities: List[str]
    model_config: Dict[str, Any]
    max_tokens: Optional[int] = None
    weight: float = 1.0

    def get_prompt_context(self, task_template, phase: str) -> str:
        """Get the prompt context for this role in the specified phase."""
        markdown_requirement = ""
        if self.name == "Model Architect":
            markdown_requirement = "\n# Format Requirements:\n# - **Must return text in Markdown format**\n# - Use headings (#, ##, ###) to organize content structure\n# - Use lists (-, *, 1.) to display key points\n# - Use code blocks (```) to show code or configuration\n# - Use tables to show comparisons\n# - Use bold (**) to emphasize important content\n# - Ensure output is structured, clear, and well-organized\n"

        latex_requirement = ""
        if self.name == "Paper Writer":
            latex_requirement = """
# Format Requirements:
# - **Must return paper content in LaTeX format**
# - **All content must be written in English**
# - **IMPORTANT: Must use Markdown code block format to output LaTeX code (wrap with ```latex or ```)**
# - **Output Example:**
#   ```latex
#   \\documentclass{article}
#   \\begin{document}
#   ...
#   \\end{document}
#   ```
# - Paper must strictly follow MCM/ICM standard template structure (12 sections):
#   1. Cover Page (Abstract + Keywords)
#   2. Table of Contents
#   3. Introduction (Background + Tasks + Methodology Overview + Flowchart)
#   4. Modeling Preparation (Assumptions + Notation + Data Preprocessing)
#   5. Problem 1 Modeling (Method + Model + Results + Figures/Tables)
#   6. Problem 2 Modeling (Method + Model + Results + Figures/Tables)
#   7. Problem 3 Modeling / New Insights (Originality Analysis + Recommendations)
#   8. Sensitivity / Robustness Analysis
#   9. Model Evaluation (Advantages + Disadvantages)
#   10. Memo (Letter to Decision Makers)
#   11. References
#   12. AI Usage Report (if applicable)
# - Use LaTeX commands and syntax, including sections (\\section{}, \\subsection{}), formulas ($...$, \\[\\]), tables (\\begin{table}), figures (\\begin{figure}), etc.
# - Ensure LaTeX code is complete and compilable
# - Use standard LaTeX document structure (\\documentclass, \\begin{document}, etc.)
"""

        meeting_flow_info = self._get_meeting_flow_info(phase)
        phase_specific_guidance = self._get_phase_specific_guidance(phase)

        base_prompt = f"""
# ========== System Notice ==========
# **IMPORTANT: This is an AI Agent Collaborative Meeting System**
# - You are an AI agent participating in a collaboration meeting organized by AIs
# - All participants (including you, Project Manager, Domain Expert, Critic, Model Architect, Paper Writer, etc.) are AI agents
# - This meeting system is entirely organized and managed by AIs, designed to solve complex problems through collaboration among multiple AI agents
# - Please participate in the discussion as an AI agent and collaborate with other AI agents to complete the target task
# ============================

# ========== LANGUAGE REQUIREMENT ==========
# **IMPORTANT: You MUST respond in English. All responses, analysis, suggestions, and discussion must be in English.**
# **This is a strict requirement. Do not use any other language in your response.**
# ============================

# ========== Mathematical Modeling Competition Context ==========
# **This task is conducted in the context of a Mathematical Modeling Competition**
# - Workflow: Problem Analysis -> Model Design -> Model Building -> Code Implementation -> Paper Writing
# - Requirements: Reasonable models, innovative methods, accurate results, complete paper (LaTeX format), reproducible code
# - Modeling paradigms: Optimization / Differential Equations / Probability & Statistics / Graph Theory / Machine Learning / Simulation, etc.
# ============================

# ========== Meeting Flow Description ==========
{meeting_flow_info}
# ============================

# Role: {self.name}
# Role Description: {self.description}

# Responsibilities:
{chr(10).join(f'- {r}' for r in self.responsibilities)}

# Current Phase: {phase}
{phase_specific_guidance}
# Task Background: {task_template.background}
# Task Goal: {task_template.goal}

# Output Requirements:
# - Please elaborate your analysis and suggestions in detail, ensuring content is complete, clear, and executable
# - No word limit, please fully express your views and analysis
# - Prioritize core points, but you may elaborate in detail
# - Use clear language, ensuring content is complete and comprehensive
# - If content is long, use lists or bullet points to improve information density
{markdown_requirement}{latex_requirement}
"""
        return base_prompt.strip()

    def _get_meeting_flow_info(self, current_phase: str) -> str:
        """Get meeting flow information."""
        phase_order = ["Problem Analysis", "Model Design", "Model Building", "Paper Writing"]
        current_index = phase_order.index(current_phase) if current_phase in phase_order else -1

        flow_info = f"""# **Meeting Flow (4 phases, completed in order):**
# 1. Problem Analysis  2. Model Design  3. Model Building  4. Paper Writing
# **Current: Phase {current_index + 1} - {current_phase}**
#
# **Current Phase Details:**
# - Goal: {self._get_phase_goal(current_phase)}
# - Participants: {self._get_phase_participants(current_phase)}
# - Minimum Rounds: {self._get_phase_min_turns(current_phase)} rounds
# - Speaking Order: {self._get_phase_speech_order(current_phase)}
# - Each round tasks (note: "each round", not "each day"):
{self._get_phase_turn_tasks(current_phase)}
# - Completion Criteria: {self._get_phase_completion_criteria(current_phase)}
#
# **Phase Advancement:** Project Manager judges when completion criteria are met to advance to the next phase
# """
        return flow_info

    @staticmethod
    def _get_phase_goal(phase: str) -> str:
        goals = {
            "Problem Analysis": "Deeply understand the task and abstract the problem from a mathematical modeling perspective.",
            "Model Design": "Design executable mathematical model solutions and select modeling paradigms.",
            "Model Building": "Build complete mathematical models, output implementation roadmap, provide input for paper writing.",
            "Paper Writing": "Write a complete academic paper using LaTeX format.",
        }
        return goals.get(phase, "")

    @staticmethod
    def _get_phase_participants(phase: str) -> str:
        participants = {
            "Problem Analysis": "Domain Expert, Critic, Project Manager",
            "Model Design": "Project Manager, Domain Expert, Critic",
            "Model Building": "Project Manager, Model Architect, Critic",
            "Paper Writing": "Project Manager, Paper Writer, Critic",
        }
        return participants.get(phase, "")

    @staticmethod
    def _get_phase_min_turns(phase: str) -> int:
        turns = {
            "Problem Analysis": 1,
            "Model Design": 2,
            "Model Building": 2,
            "Paper Writing": 6,
        }
        return turns.get(phase, 1)

    @staticmethod
    def _get_phase_speech_order(phase: str) -> str:
        orders = {
            "Problem Analysis": "Domain Expert abstracts -> Critic questions -> Project Manager advances",
            "Model Design": "Domain Expert designs -> Critic evaluates -> Project Manager advances",
            "Model Building": "Model Architect builds -> Critic evaluates -> Project Manager advances",
            "Paper Writing": "Paper Writer writes -> Critic evaluates -> Project Manager concludes",
        }
        return orders.get(phase, "")

    @staticmethod
    def _get_phase_turn_tasks(phase: str) -> str:
        tasks = {
            "Problem Analysis": "#   * Round 1: Domain Expert abstracts the problem (variables / parameters / sets, inputs / outputs, constraints and objectives); Critic evaluates and questions; Project Manager summarizes and decides to advance.",
            "Model Design": "#   * Round 1: Domain Expert proposes design (model paradigm, objective functions / constraints / state equations); Critic evaluates and questions; Project Manager coordinates.\n#   * Round 2: Domain Expert refines the design (parameter calibration, validation process); Critic further evaluates; Project Manager decides to advance.",
            "Model Building": "#   * Round 1: Model Architect follows the Domain Expert's requirements for mathematical modeling; Critic evaluates (solvability / robustness / generalizability / interpretability); Project Manager coordinates.\n#   * Round 2: Model Architect refines the model and outputs the implementation roadmap; Critic further evaluates; Project Manager decides to advance (meet conditions for entering the Paper Writing phase).",
            "Paper Writing": "#   * Round 1: Paper Writer writes the draft (problem restatement / model assumptions / notation / problem analysis); Critic evaluates; Project Manager coordinates.\n#   * Round 2: Refine the model building section; Critic evaluates; Project Manager coordinates.\n#   * Round 3: Refine the model solution section; Critic evaluates; Project Manager coordinates.\n#   * Round 4: Refine the model verification and evaluation section; Critic evaluates; Project Manager coordinates.\n#   * Round 5: Refine references / figures / appendix; Critic evaluates; Project Manager coordinates.\n#   * Round 6: Final refinement (ensure the 25-page requirement); Critic final evaluation; Project Manager concludes.",
        }
        return tasks.get(phase, "")

    @staticmethod
    def _get_phase_completion_criteria(phase: str) -> str:
        criteria = {
            "Problem Analysis": "Domain Expert completes problem abstraction, Critic raises questions and supplements, Project Manager confirms it is ready to enter the Model Design phase.",
            "Model Design": "Domain Expert provides a clear design and modeling requirements, Critic has evaluated and questioned, the design is clear and complete to provide input for Model Building.",
            "Model Building": "Model Architect follows the Domain Expert's requirements to complete mathematical modeling, Critic has evaluated and questioned, the model is complete and executable supporting paper writing.",
            "Paper Writing": "Paper Writer completes paper writing, Critic has evaluated and given feedback, the paper uses LaTeX format with complete structure and clear logic, Project Manager confirms it is ready to end.",
        }
        return criteria.get(phase, "")

    def _get_phase_specific_guidance(self, phase: str) -> str:
        """Get specific task guidance for the current role in the current phase."""
        if phase == "Problem Analysis":
            if self.name == "Project Manager":
                return """# **Your Tasks in Current Phase (Problem Analysis):**
# - This is the first phase of the meeting. The Domain Expert first performs problem abstraction, then the Critic questions and supplements.
# - Listen carefully to both roles' analysis. Record variables, constraints, assumptions, data requirements, and risk points.
# - After they speak, summarize the problem definition, key challenges, and input gaps in bullet points.
# - Determine whether information is sufficient and whether supplementary data or clarification is needed before deciding whether to advance to the "Model Design" phase.
# - If unable to advance temporarily, clarify what supplementary information is needed and the next steps.
"""
            if self.name == "Domain Expert":
                return """# **Your Tasks in Current Phase (Problem Analysis):**
# - Speak proactively, complete problem abstraction based on the task background (variables / parameters / sets, inputs / outputs, objectives, constraints).
# - Clarify key assumptions, data requirements, quantitative indicators, and observables.
# - Break down the problem into discussable sub-problems to provide executable input for subsequent model design.
# - Point out potential uncertainties and data sources that need verification.
"""
            if self.name == "Critic":
                return """# **Your Tasks in Current Phase (Problem Analysis):**
# - After the Domain Expert speaks, evaluate the completeness and rationality of their problem abstraction.
# - Question key assumptions, identify potential risks and blind spots, propose boundaries or data that need clarification.
# - Pose the most urgent key questions to ensure the problem definition is rigorous enough.
# - Do not model on behalf of other roles. Only raise evaluations, questions, and improvement suggestions.
"""
            return """# **Your Tasks in Current Phase (Problem Analysis):**
# - This phase is led by the Domain Expert and Critic. You do not need to speak at this time.
# - Focus on their definitions of variables, constraints, and data requirements. Record key information useful for your subsequent stages.
# - Prepare for the next phase (Model Design) and understand the core contradictions and success metrics of the problem.
"""

        if phase == "Model Design":
            if self.name == "Project Manager":
                return """# **Your Tasks in Current Phase (Model Design):**
# - After the Domain Expert and Critic speak, summarize the key points of this phase's discussion.
# - Evaluate whether the design is clear and complete enough to decide whether to enter the "Model Building" phase.
# - Ensure the Domain Expert provides clear modeling requirements and constraints to the Model Architect.
"""
            if self.name.startswith("Domain Expert"):
                return """# **Your Tasks in Current Phase (Model Design):**
# - This is your main work phase. You need to provide professional design solutions.
# - Analyze the problem from your professional domain perspective and propose the overall architecture of the solution.
# - **IMPORTANT: Must provide clear modeling requirements and constraints to the Model Architect.**
# - The first round of speech must provide solid data or theoretical support to establish the discussion baseline.
# - Evaluate the feasibility of the design and provide implementation suggestions.
# - Discuss the pros and cons of the design with the Critic and refine the design solution.
"""
            if self.name == "Critic":
                return """# **Your Tasks in Current Phase (Model Design):**
# - Raise key questions about the design proposed by the Domain Expert.
# - Identify potential risks and problems.
# - Evaluate the pros and cons of the design.
# - In the middle of the discussion, you may propose disruptive alternative solutions to challenge the established path.
# - Provide improvement suggestions to ensure the quality and feasibility of the design.
# - **IMPORTANT CONSTRAINT: Only evaluate and question the design. Do not design on behalf of the Domain Expert.**
"""
            return """# **Your Tasks in Current Phase (Model Design):**
# - This phase mainly involves the Project Manager, Domain Expert, and Critic.
# - You do not need to speak at this time, but should focus on the discussion content and understand the design solution.
# - Prepare for subsequent phases.
"""

        if phase == "Model Building":
            if self.name == "Project Manager":
                return """# **Your Tasks in Current Phase (Model Building):**
# - After the Model Architect and Critic speak, summarize the key points of this phase's modeling.
# - Ensure the Model Architect strictly follows the modeling requirements and constraints proposed by the Domain Expert in the "Model Design" phase.
# - Evaluate whether the model is complete and executable, and decide whether to directly enter the "Paper Writing" phase.
"""
            if self.name == "Model Architect":
                return """# **Your Tasks in Current Phase (Model Building):**
# - This is your main work phase. You need to perform mathematical modeling.
# - **IMPORTANT: Must strictly follow the modeling requirements and constraints proposed by the Domain Expert in the "Model Design" phase.**
# - Abstract the problem and define variables / parameters from a mathematical modeling perspective.
# - Propose model assumptions, establish constraints and objective functions.
# - Select and derive appropriate mathematical models and algorithms (analytical / numerical).
# - Provide parameter calibration and sensitivity analysis plans.
# - Design solution processes, complexity analysis, and robustness verification.
# - Output implementable numerical experiments and evaluation metrics.
# - **Must return text in Markdown format.**
"""
            if self.name == "Critic":
                return """# **Your Tasks in Current Phase (Model Building):**
# - Raise key questions about the mathematical model built by the Model Architect.
# - Identify potential risks and problems with the model.
# - Evaluate the model's pros, cons, feasibility, and robustness.
# - Provide improvement suggestions to ensure model quality.
# - **IMPORTANT CONSTRAINT: Only evaluate and question the model. NEVER model on behalf of the Model Architect.**
# - **STRICTLY PROHIBITED: No mathematical modeling, no model architecture design, no formula derivation, no objective function or constraint establishment.**
"""
            return """# **Your Tasks in Current Phase (Model Building):**
# - This phase mainly involves the Project Manager, Model Architect, and Critic.
# - You do not need to speak at this time, but should focus on the discussion content and understand the model design.
# - Prepare for subsequent phases.
"""

        if phase == "Paper Writing":
            if self.name == "Project Manager":
                return """# **Your Tasks in Current Phase (Paper Writing):**
# - Coordinate the collaboration between the Paper Writer and Critic to ensure paper quality.
# - After the Paper Writer completes the paper, summarize this phase's achievements and evaluate whether the paper is complete and meets the requirements.
# - This is the last phase. The meeting will end after completion.
"""
            if self.name == "Paper Writer":
                return """# **Your Tasks in Current Phase (Paper Writing):**
# - This is your main work phase, and also the last phase of the meeting.
# - Write a complete academic paper based on all discussion content, targeting a complete 25-page paper.
# - **Must output paper content in LaTeX format.**
# - Strictly organize content according to the paper template structure: Problem Restatement, Model Assumptions, Notation, Problem Analysis, Model Building, Model Solution, Model Verification, Model Evaluation, References.
# - Improve the paper through multiple rounds of iteration, continuously improving based on the Critic's feedback.
# - Ensure the paper content is complete, the logic is clear, and it meets academic standards, ultimately achieving the 25-page complete paper requirement.
# - Use standard LaTeX syntax and commands to ensure the output LaTeX code can be compiled.
"""
            if self.name == "Critic":
                return """# **Your Tasks in Current Phase (Paper Writing):**
# - Raise key questions and evaluations on the paper written by the Paper Writer.
# - Evaluate the paper's completeness, logic, accuracy, and standardization.
# - Check whether the paper meets the requirements of a mathematical modeling competition paper (complete structure, clear logic, accurate expression, visualized results, standardized format).
# - Check whether the paper meets the competition's evaluation criteria (model reasonableness, method innovation, result accuracy, paper completeness, reproducibility).
# - Identify potential problems and shortcomings in the paper (e.g., incomplete structure, unclear logic, inaccurate expression, missing figures / tables, non-standard format).
# - Evaluate whether the paper meets the 25-page complete paper requirement and whether the content is substantial.
# - Provide specific improvement suggestions to help the Paper Writer refine the paper.
# - **IMPORTANT CONSTRAINT: Only evaluate and question the paper. NEVER write the paper on behalf of the Paper Writer.**
# - **STRICTLY PROHIBITED: Do not write the paper, do not modify paper content, do not perform paper writing functions. Only evaluate and make suggestions.**
"""
            return """# **Your Tasks in Current Phase (Paper Writing):**
# - This phase mainly involves the Project Manager, Paper Writer, and Critic.
# - This is the last phase of the meeting.
# - You do not need to speak at this time, but should monitor the paper writing progress.
"""

        return ""


class AgentRoleFactory:
    """AI role factory."""

    @staticmethod
    def create_project_manager(model_config: Dict[str, Any] = None) -> AgentRole:
        """Create a Project Manager role."""
        if model_config is None:
            from model_library import get_model_config_by_category, ModelCategory
            model_config = get_model_config_by_category(ModelCategory.TEXT_DIALOGUE, "gemini-3-pro-preview")

        return AgentRole(
            role_type=AgentRoleType.PROJECT_MANAGER,
            name="Project Manager",
            description="Responsible for coordinating and controlling the entire mathematical modeling competition process, focusing on guidance and decision confirmation, ensuring each phase advances in an orderly manner and implementing consensus. Understands the characteristics, processes, and requirements of mathematical modeling competitions, ensuring the team's work meets competition standards.",
            responsibilities=[
                "Understand the complete process of mathematical modeling competitions: Problem Analysis -> Model Design -> Model Building -> Code Implementation -> Paper Writing.",
                "Parse tasks and route them to appropriate phases, ensuring compliance with the standard process of mathematical modeling competitions.",
                "Coordinate the work of various AI roles and maintain clear division of labor, ensuring efficient team collaboration (meeting competition time-limit requirements).",
                "Control the meeting pace and process, keep discussions focused on task goals, ensure completion of competition tasks within limited time.",
                "Summarize the core conclusions of each phase and propose next action suggestions, ensuring each phase meets the completion standards required by the competition.",
                "Confirm whether to enter the next phase after phase completion, ensuring current phase achievements meet the evaluation standards of mathematical modeling competitions.",
                "Exercise mandatory advancement rights after team consensus is reached, locking in action plans.",
                "Do not write deliverables on behalf of other roles; only responsible for guidance and decision confirmation.",
                "**IMPORTANT: Project Manager's requirements must be responded to and complied with by all roles**",
                "**Mathematical Modeling Competition Awareness: Understand the competition's evaluation criteria (model reasonableness, method innovation, result accuracy, paper completeness, reproducibility), ensuring the team's work meets these standards.**",
            ],
            model_config=model_config,
            weight=10.0,
        )

    @staticmethod
    def create_domain_expert(name: str, domain: str, model_config: Dict[str, Any] = None) -> AgentRole:
        """Create a Domain Expert role."""
        if model_config is None:
            from model_library import get_model_config_by_category, ModelCategory
            model_config = get_model_config_by_category(ModelCategory.DEEP_THINKING, "gemini-3-pro-preview")

        return AgentRole(
            role_type=AgentRoleType.DOMAIN_EXPERT_1,
            name=name,
            description=f"Senior expert in the {domain} domain, providing professional knowledge and suggestions in mathematical modeling competitions. Understands the characteristics and requirements of mathematical modeling competitions, and can guide modeling work from a domain knowledge perspective.",
            responsibilities=[
                f"Analyze mathematical modeling competition problems from the {domain} perspective, understand the essence of practical problems and modeling needs.",
                "The first round of speech must provide solid data or theoretical support to establish the discussion baseline, providing professional guidance for subsequent modeling work.",
                "Provide professional solutions in the model design phase, selecting suitable modeling paradigms (optimization models, differential equations, probability & statistics, graph theory, machine learning, simulation, etc.).",
                "Evaluate the feasibility of solutions, ensuring the design meets the innovation requirements and reasonableness standards of mathematical modeling competitions.",
                "Provide implementation suggestions, ensuring modeling solutions can be completed within limited time (meeting competition time limits).",
                "**IMPORTANT: Provide clear modeling requirements and constraints to the Model Architect, ensuring the model design meets the evaluation standards of mathematical modeling competitions.**",
                "**Mathematical Modeling Competition Awareness: Understand that the competition encourages innovative modeling ideas and solution methods, not limited to traditional methods, while ensuring the model's reasonableness and solvability.**",
            ],
            model_config=model_config,
            weight=8.0,
        )

    @staticmethod
    def create_critic(model_config: Dict[str, Any] = None) -> AgentRole:
        """Create a Critic role."""
        if model_config is None:
            from model_library import get_model_config_by_category, ModelCategory
            model_config = get_model_config_by_category(ModelCategory.DEEP_THINKING, "gemini-3-pro-preview")

        return AgentRole(
            role_type=AgentRoleType.CRITIC,
            name="Critic",
            description="Responsible for questioning and evaluating solutions in mathematical modeling competitions, identifying risks and problems, and proposing disruptive alternative solutions in the middle phase to test team consensus. Understands the evaluation standards of mathematical modeling competitions and evaluates from dimensions such as model reasonableness, method innovation, and result accuracy.",
            responsibilities=[
                "Raise key questions about solutions from the perspective of mathematical modeling competition evaluation criteria (model reasonableness, method innovation, result accuracy, etc.).",
                "Identify potential risks and problems, ensuring model solutions meet competition requirements (solvability, robustness, generalizability, interpretability, etc.).",
                "Evaluate the pros and cons of solutions, conducting evaluation against mathematical modeling competition evaluation standards.",
                "Propose disruptive alternative solutions in the middle of discussion, challenging established paths, and testing team consensus, reflecting the innovation requirements of the competition.",
                "Provide improvement suggestions to ensure solution and code quality meet the standards of mathematical modeling competitions.",
                "Ensure the quality and feasibility of solutions, evaluating whether the solution can be completed within limited time (meeting competition time limits).",
                "**IMPORTANT CONSTRAINT: Only evaluate and question model solutions and code. NEVER model on behalf of the Model Architect.**",
                "**STRICTLY PROHIBITED: No mathematical modeling, no model architecture design, no formula derivation, no objective function or constraint establishment.**",
                "**Responsibility Boundary: Can only raise questions, identify risks, and evaluate pros and cons from an evaluation perspective. Cannot perform any modeling or coding work.**",
                "**Mathematical Modeling Competition Awareness: Understand the competition's evaluation criteria, evaluating and questioning from dimensions such as model reasonableness, method innovation, result accuracy, paper completeness, and reproducibility.**",
            ],
            model_config=model_config,
            weight=5.0,
        )

    @staticmethod
    def create_architect(model_config: Dict[str, Any] = None) -> AgentRole:
        """Create a Model Architect role (Mathematical Modeling)."""
        if model_config is None:
            from model_library import get_model_config_by_category, ModelCategory
            model_config = get_model_config_by_category(ModelCategory.TEXT_DIALOGUE, "gemini-3-pro-preview")

        return AgentRole(
            role_type=AgentRoleType.ARCHITECT,
            name="Model Architect",
            description="Responsible for the establishment, solution, and verification of mathematical models in mathematical modeling competitions. From a modeling perspective, abstracts problems, selects methods, and produces derivations and numerical implementations. Outputs reusable models and parameter calibration plans. Deeply understands the modeling paradigms, evaluation standards, and paper requirements of mathematical modeling competitions.",
            responsibilities=[
                "Understand common modeling paradigms of mathematical modeling competitions (optimization models, differential equations, probability & statistics, graph theory, machine learning, simulation, etc.), and select appropriate modeling paradigms based on problem characteristics.",
                "Abstract the problem and define variables / parameters from a mathematical modeling perspective, ensuring the model accurately reflects the actual problem (meeting the competition's model reasonableness requirement).",
                "Propose model assumptions, establish constraints and objective functions, ensuring assumptions are reasonable and the model is solvable (meeting the competition's model reasonableness requirement).",
                "Select and derive appropriate mathematical models and algorithms (analytical / numerical), reflecting the innovation of methods (meeting the competition's innovation requirement).",
                "Provide parameter calibration and sensitivity analysis plans, ensuring model results are accurate (meeting the competition's result accuracy requirement).",
                "Design solution processes, complexity analysis, and robustness verification, ensuring the model is reproducible (meeting the competition's reproducibility requirement).",
                "Output implementable numerical experiments and evaluation metrics, supporting paper writing (meeting the competition's paper completeness requirement).",
                "**IMPORTANT CONSTRAINT: Can only architect models according to the Domain Expert's requirements. Must strictly follow the modeling requirements and constraints proposed by the Domain Expert.**",
                "**Must wait for the Domain Expert to clarify modeling requirements before starting architecture work.**",
                "**Mathematical Modeling Competition Awareness: Deeply understand the evaluation standards of mathematical modeling competitions, ensuring the model design meets the requirements of model reasonableness, method innovation, result accuracy, and reproducibility, providing a complete model foundation for subsequent paper writing.**",
            ],
            model_config=model_config,
            weight=6.0,
        )

    @staticmethod
    def create_paper_writer(model_config: Dict[str, Any] = None) -> AgentRole:
        """Create a Paper Writer role."""
        if model_config is None:
            from model_library import get_model_config_by_category, ModelCategory
            model_config = get_model_config_by_category(ModelCategory.ACADEMIC_WRITING, "claude-sonnet-5")
            model_config['model'] = "claude-sonnet-5"

        return AgentRole(
            role_type=AgentRoleType.PAPER_WRITER,
            name="Paper Writer",
            description="Responsible for writing comprehensive analytical reports on real-world mathematical modeling problems, outputting in LaTeX format. Adapts the report structure to the problem type and the team template in use.",
            responsibilities=[
                "Write complete analytical reports based on all discussion content, integrating the achievements of all phases including problem analysis, model design, model building, and verification.",
                "Output report content in LaTeX format (```latex ... ``` block), meeting the format specification requirements of the team template.",
                "Adapt the report structure to the team template in use (competition / scientific_research / industry / custom). For competition template use MCM/ICM-style sections; for research/industry templates use the generic 12-section structure.",
                "Ensure the report content is complete, the logic is clear, and it meets academic standards. The report logic forms a complete chain from problem analysis to model building to solution verification.",
                "Use accurate mathematical language and symbols. Formula derivations must be rigorous. Display model results through figures and tables to enhance report readability.",
                "Use standard LaTeX syntax and commands to ensure the output LaTeX code can be compiled.",
            ],
            model_config=model_config,
            weight=2.0,
        )

    # -----------------------------------------------------------------
    # Unified slot-based factory (used by team templates)
    # -----------------------------------------------------------------
    @staticmethod
    def create_from_slot(slot, model_name: str):
        """Create an AgentRole from a TeamTemplate slot + user-chosen model.

        Maps role_name keywords to the existing create_* methods:
          - "Project Manager" / "Engagement Lead" / "PI" → create_project_manager
          - "Domain Expert" / "Subject Matter Expert" / "Senior Researcher" → create_domain_expert
          - "Critic" / "Risk Reviewer" / "Methodologist" → create_critic
          - "Model Architect" / "Solution Architect" → create_architect
          - "Paper Writer" / "Report Writer" → create_paper_writer

        For domain-expert variants (SME / Senior Researcher), the original
        "Domain Expert" name and a domain label are passed so existing phase
        guidance keeps working.
        """
        from utils.model_resolver import resolve_model_config_for_slot

        model_config = resolve_model_config_for_slot(model_name)

        name = (slot.role_name or "").strip()
        lower = name.lower()

        # PM family
        if lower in ("project manager", "engagement lead", "pi"):
            # Preserve the template's display name (PI / Engagement Lead / Project Manager)
            role = AgentRoleFactory.create_project_manager(model_config)
            role.name = name
            role.description = AgentRoleFactory._pm_description_for_name(name)
            return role

        # Domain Expert family — pass through the actual template name
        if lower in ("domain expert", "subject matter expert", "senior researcher"):
            domain = "Technical"
            if "subject" in lower or "industry" in lower or "consulting" in lower:
                domain = "Industry / Policy"
            elif "senior" in lower or "research" in lower:
                domain = "Scientific Research"
            role = AgentRoleFactory.create_domain_expert(name, domain, model_config)
            return role

        # Critic family
        if lower in ("critic", "risk reviewer", "methodologist", "reproducibility reviewer"):
            role = AgentRoleFactory.create_critic(model_config)
            role.name = name
            role.description = AgentRoleFactory._analyst_description_for_name(name)
            return role

        # Architect family
        if lower in ("model architect", "solution architect"):
            role = AgentRoleFactory.create_architect(model_config)
            role.name = name
            role.description = AgentRoleFactory._architect_description_for_name(name)
            return role

        # Writer family
        if lower in ("paper writer", "report writer") or lower.endswith("writer"):
            role = AgentRoleFactory.create_paper_writer(model_config)
            role.name = name
            role.description = AgentRoleFactory._writer_description_for_name(name)
            return role

        # Unknown role - default to domain expert so the meeting can still proceed
        return AgentRoleFactory.create_domain_expert(name or "Domain Expert", "General", model_config)

    # -----------------------------------------------------------------
    # Family-specific description overrides (used when a non-canonical name is
    # used in a template, e.g. "PI" or "Engagement Lead"). The base prompt body
    # still comes from the canonical factory for stability.
    # -----------------------------------------------------------------
    @staticmethod
    def _pm_description_for_name(name: str) -> str:
        if name == "PI":
            return ("Principal Investigator for a scientific research team. Responsible for setting the research direction, "
                    "coordinating collaborators, and ensuring methodological rigor and reproducibility.")
        if name == "Engagement Lead":
            return ("Engagement lead for an industry / consulting team. Responsible for stakeholder alignment, scoping "
                    "deliverables, and ensuring risk-aware decision support throughout the project.")
        return ("Responsible for coordinating and controlling the entire mathematical modeling competition process, ensuring "
                "each phase advances in an orderly manner and that team consensus is implemented.")

    @staticmethod
    def _analyst_description_for_name(name: str) -> str:
        if name == "Methodologist":
            return ("Methodologist focused on research design, statistical validity, reproducibility, and methodological critique. "
                    "Strictly evaluation-only: never performs modeling or paper-writing work itself.")
        if name == "Risk Reviewer":
            return ("Risk reviewer focused on identifying business, technical and operational risks. Evaluates decision "
                    "alternatives, raises objections, and recommends risk mitigation strategies.")
        return ("Evaluates and questions work produced by other agents. Strictly evaluation-only: never performs modeling "
                "or paper-writing work itself.")

    @staticmethod
    def _architect_description_for_name(name: str) -> str:
        if name == "Solution Architect":
            return ("Solution architect responsible for translating problem requirements into an executable technical design — "
                    "covering data preparation, algorithm choice, scalability, and operational considerations.")
        return ("Model architect responsible for translating modeling requirements into a complete mathematical model with "
                "clear interfaces, algorithm choices, and evaluation hooks.")

    @staticmethod
    def _writer_description_for_name(name: str) -> str:
        if name == "Report Writer":
            return ("Report writer for a scientific research / industry consulting team. Produces the final document using "
                    "the template's report structure (different from the MCM/ICM 12-section paper).")
        return ("Paper writer responsible for producing a complete academic paper in LaTeX, following the team template's "
                "section structure and formatting requirements.")
