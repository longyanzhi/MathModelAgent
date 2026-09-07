#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meeting Manager
Features:
- Phased workflow: Problem Analysis -> Model Design -> Model Building -> Paper Writing
- AI agent collaboration meeting system
- Model library for automatic model selection
- Knowledge base management
- Context file management for pre-meeting materials
- Phase progress tracker
- Phase controller
- Phase confirmation handler
- Phase advisor
- Phase summarizer
"""

import asyncio
import json
import time
import os
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import difflib

from config import Config
from models import RoleType, Message
from ai_agent import AIAgent
from vector_memory import VectorMemory
from utils.topic_focus import SolutionTracker
from utils.logger import logger
from models.task_template import TaskTemplate
from models.agent_roles import AgentRoleFactory, AgentRole
from utils.phase_tracker import PhaseProgressTracker
from utils.context_file_manager import ContextFileManager
from utils.report_generator import ReportGenerator
from utils.prompt_builder import PromptBuilder
from utils.node_executor import NodeExecutionResult

load_dotenv()


class MeetingManagerV2:
    THINKING_PAUSE_SECONDS = 0.5
    CONTEXT_FILE_MAX_CHARS = 8000
    MAX_CONTEXT_FILES = 6
    PAPER_WRITER_MAX_HISTORY_MESSAGES = 60
    PAPER_WRITER_MAX_HISTORY_CHARS = 50000

    def __init__(self, topic: str, max_turns: int = None, message_callback=None,
                 session_id: str = None, socketio=None, selected_models=None,
                 # -------- v2: Optional dependency injection --------
                 manifest_writer=None,
                 intervention_policy=None,
                 node_executor=None,
                 graph_runner=None):
        self.topic = topic
        self.problem_segments = self._extract_problem_segments(topic)
        self.max_turns = max_turns
        self.agents: List[AIAgent] = []
        self.conversation: List[Message] = []
        self.current_turn = 0
        self.message_callback = message_callback

        # New: streaming & human-AI interaction support
        self.session_id = session_id
        self.socketio = socketio
        self.selected_models = selected_models  # user-selected models dict
        self._user_msg_provider = None  # Callable returning list of pending user messages
        self._external_context: List[str] = []  # appended by vision handler

        # -------- v2: Injected new modules, auto-configured if not provided --------
        from utils.manifest import ManifestWriter
        from utils.manifest import AgentSpec
        from utils.intervention import UserInterventionPolicy
        from utils.node_executor import NodeExecutor

        self.manifest: Optional[ManifestWriter] = manifest_writer
        if self.manifest is None:
            self.manifest = ManifestWriter(
                topic=topic,
                max_turns=max_turns,
                phase_turns={
                    "Problem Analysis": 20,
                    "Model Design": 20,
                    "Model Building": 20,
                    "Paper Writing": 30,
                },
                selected_models=selected_models,
            )
        self.intervention: UserInterventionPolicy = intervention_policy or UserInterventionPolicy(manifest=self.manifest)
        self.node_executor: NodeExecutor = node_executor or NodeExecutor(manifest=self.manifest)

        from utils.graph_runner import GraphRunner
        self.graph_runner: GraphRunner = graph_runner or GraphRunner()

        self.vector_memory = VectorMemory()
        self.solution_tracker = SolutionTracker(topic)
        self.progress_tracker = PhaseProgressTracker()
        self.meeting_phase = "Problem Analysis"
        self.agent_roles: Dict[str, AgentRole] = {}  # Store role definitions
        # Deduplication: record each role's last output
        self._last_agent_output: Dict[str, str] = {}
        # Dynamic flow control state (mandatory voting and silence mode removed)
        # Pre-meeting materials state
        self.workspace_root = os.getcwd()
        # Use ContextFileManager for pre-meeting materials
        self.context_file_manager = ContextFileManager(self.workspace_root, self.vector_memory)
        # Phase advancement control
        self.phase_order = ["Problem Analysis", "Model Design", "Model Building", "Paper Writing"]
        self.awaiting_external_confirmation: bool = False
        self.pending_phase_transition: Optional[str] = None
        # Phase rounds configuration (default values)
        self.phase_turns = {
            "Problem Analysis": 20,
            "Model Design": 20,
            "Model Building": 20,
            "Paper Writing": 30
        }
        self._phase_advance_signal: bool = False
        self._phase_hold_signal: bool = False
        self._phase_confirmation_event: Optional[asyncio.Event] = None
        self._phase_confirmation_result: Optional[bool] = None
        self.context_file_manager.load_default_context_files()

    # -----------------------------------------------------------------
    # Manifest convenience methods
    # -----------------------------------------------------------------
    def register_agent_spec(self, agent_name: str, model_config: Optional[Dict[str, Any]] = None):
        """Called after agents are created, writes specs to manifest"""
        if self.manifest is None or not model_config:
            return
        from utils.manifest import AgentSpec
        spec = AgentSpec(
            name=agent_name,
            role=str(model_config.get("provider") or ""),
            model=model_config.get("model"),
            provider=model_config.get("provider"),
            base_url=model_config.get("base_url"),
            temperature=model_config.get("temperature"),
            max_tokens=model_config.get("max_tokens"),
            extra={k: v for k, v in model_config.items()
                   if k not in ("model", "provider", "base_url", "temperature", "max_tokens")},
        )
        self.manifest.add_agent(spec)

    def save_manifest(self) -> Optional[Path]:
        """Called at the end of the meeting, saves manifest to disk"""
        if self.manifest is None:
            return None
        try:
            return self.manifest.save()
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}", exc_info=True)
            return None

    def set_user_message_provider(self, provider):
        """Set callback to fetch pending user messages"""
        self._user_msg_provider = provider

    def add_external_context(self, content: str):
        """Add external content (e.g. file recognition results) to context"""
        self._external_context.append(content)

    def _drain_user_messages_to_callback(self):
        """Fetch any pending user messages.

        The user-facing echo is sent directly from app.py when the user sends
        a message, so here we only update internal conversation history without
        re-broadcasting (which would create duplicates in the UI).
        """
        if not self._user_msg_provider:
            return
        try:
            pending = self._user_msg_provider() or []
        except Exception as e:
            logger.error(f"Failed to fetch user messages: {e}", exc_info=True)
            return
        for msg in pending:
            content = (msg.get('content') if isinstance(msg, dict) else str(msg)) or ''
            if not content:
                continue
            # Add to conversation history so subsequent agents can reference it
            self.conversation.append(
                Message(role='You', content=content, timestamp=time.time(), turn=self.current_turn)
            )
            self.vector_memory.add_conversation('You', content, self.current_turn, 'user')

    def _consume_external_context(self) -> List[str]:
        """Drain and return any pending external context items"""
        items = list(self._external_context)
        self._external_context.clear()
        return items

    def _extract_problem_segments(self, raw_topic: str) -> List[str]:
        """Extract problem segments from topic text (supports Chinese and English enumeration patterns)"""
        if not raw_topic:
            return []
        text = raw_topic.strip()
        if not text:
            return []

        segments: List[str] = []
        enumerated_pattern = re.compile(
            r'((?:Problem|Question|Q|Task)\s*\d+|[（(]?\d+[)）]|[1-9]\d*\.)\s*[：:、．.]?',
            re.IGNORECASE
        )
        matches = list(enumerated_pattern.finditer(text))
        if matches:
            for idx, match in enumerate(matches):
                start = match.end()
                end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
                snippet = text[start:end].strip()
                if not snippet:
                    continue
                label = match.group().strip()
                label = re.sub(r'[：:、．.]+$', '', label)
                if label:
                    segments.append(f"{label}：{snippet}")
                else:
                    segments.append(snippet)

        if not segments:
            line_segments = [seg.strip() for seg in re.split(r'[\n\r;；]+', text) if seg.strip()]
            if len(line_segments) > 1:
                segments = line_segments

        if not segments:
            sentence_segments = [seg.strip() for seg in re.split(r'[。!?！？]+', text) if seg.strip()]
            keyword_sentences = [
                seg for seg in sentence_segments
                if any(kw in seg for kw in ["problem", "issue", "goal", "requirement"])
            ]
            if len(keyword_sentences) > 1:
                segments = keyword_sentences

        if not segments:
            segments = [text]

        return segments

    def _get_problem_focus_prompt(self, agent_name: Optional[str] = None) -> str:
        """Get prompt for multi-problem focus when user submits multiple sub-problems"""
        if len(self.problem_segments) <= 1:
            return ""
        listed = "\n".join([f"{idx + 1}. {segment}" for idx, segment in enumerate(self.problem_segments, 1)])
        manager_clause = ""
        moderator_name = self._get_actual_role_name("Project Manager")
        if agent_name == moderator_name:
            manager_clause = (
                f"\n- **{moderator_name} Responsibilities**: Track each sub-problem above item by item, report current status, blocking points, and next steps,"
                " and prohibit advancing to the next phase until all sub-problems have sufficient input."
            )
        return (
            "**Multiple Problem Reminder**: The user has submitted multiple independent modeling sub-problems at once, each must be analyzed and delivered separately.\n"
            f"{listed}\n"
            "- **Speaking Requirements**: Clearly indicate which sub-problem is being addressed in each speech segment, do not merge them vaguely."
            "\n- **Delivery Requirements**: Each sub-problem must complete the full chain (Analysis -> Model -> Implementation -> Verification -> Paper), all steps are required."
            f"{manager_clause}"
        )

    def setup_default_agents(self):
        """V2 uses new role system: Project Manager / Domain Expert / Critic / Model Architect / Paper Writer.
        Supports user-selected models via selected_models (dict category->name OR list of {role, model_name}).
        """
        from utils.model_manager import get_model_manager
        manager = get_model_manager()

        # Map of role name -> category for resolving user selections
        role_to_category = {
            "Project Manager": "text_dialogue",
            "Domain Expert": "deep_thinking",
            "Critic": "deep_thinking",
            "Model Architect": "text_dialogue",
            "Paper Writer": "academic_writing",
        }

        def resolve_config(category: str, default_name: str, role_name: Optional[str] = None) -> Dict[str, Any]:
            """Resolve model config for a role.

            Priority:
            1. User's selection → validated in the correct category
               (if the name doesn't exist in the library at all → warn, fall back)
            2. Category default (ModelLibrary)
            """
            from model_library import get_model_config_by_category, ModelCategory, get_model_library

            cat_map = {
                "text_dialogue": ModelCategory.TEXT_DIALOGUE,
                "deep_thinking": ModelCategory.DEEP_THINKING,
                "academic_writing": ModelCategory.ACADEMIC_WRITING,
                "code_programming": ModelCategory.CODE_PROGRAMMING,
            }
            target_cat = cat_map.get(category, ModelCategory.TEXT_DIALOGUE)
            selected_name: Optional[str] = None

            if self.selected_models:
                if isinstance(self.selected_models, dict):
                    selected_name = self.selected_models.get(category)
                    if not selected_name and role_name:
                        selected_name = self.selected_models.get(role_name)
                elif isinstance(self.selected_models, list):
                    for item in self.selected_models:
                        if not isinstance(item, dict):
                            continue
                        item_role = item.get("role")
                        if item_role and (item_role == category or (role_name and item_role == role_name)):
                            selected_name = item.get("model_name") or item.get("name")
                            break

            if selected_name:
                # 2a) Check if it's a custom user model first
                user_cfg = manager.get_config_for_agent(selected_name)
                if user_cfg:
                    logger.info(f"[{role_name or category}] Using custom model: {selected_name}")
                    return user_cfg

                # 2b) Check if it exists in ANY built-in category
                library = get_model_library()
                name_exists = any(
                    m.name == selected_name for m in library.models.values()
                )

                if not name_exists:
                    logger.warning(
                        f"[{role_name or category}] Selected model '{selected_name}' "
                        f"not found in model library — falling back to "
                        f"'{default_name}' (category: {category})."
                    )
                    selected_name = None
                else:
                    # Name is valid in the library, but may be the wrong category.
                    # Find the model's ACTUAL category and use its real config.
                    library = get_model_library()
                    actual_model_info: Optional[Any] = None
                    for m in library.models.values():
                        if m.name == selected_name:
                            actual_model_info = m
                            break
                    if actual_model_info is None:
                        # Should not happen given name_exists=True, but guard anyway
                        selected_name = None
                    elif actual_model_info.category != target_cat:
                        # Model exists but in a different category.
                        # Get its config from its actual category.
                        cfg = get_model_config_by_category(actual_model_info.category, selected_name)
                        logger.warning(
                            f"[{role_name or category}] Model '{selected_name}' is in "
                            f"category '{actual_model_info.category.value}', not "
                            f"'{category}' — using it anyway (provider={cfg.get('provider')}, "
                            f"model={cfg.get('model')}). If unavailable in your dmxapi.cn "
                            f"subscription, meetings will fail."
                        )
                        return cfg
                    else:
                        # Same category, use normally
                        cfg = get_model_config_by_category(target_cat, selected_name)
                        return cfg

            cfg = get_model_config_by_category(target_cat, default_name)
            return cfg

        pm_cfg = resolve_config("text_dialogue", "gemini-2.5-flash", role_name="Project Manager")
        de_cfg = resolve_config("deep_thinking", "gemini-2.5-flash-thinking", role_name="Domain Expert")
        cr_cfg = resolve_config("deep_thinking", "gemini-2.5-flash-thinking", role_name="Critic")
        ar_cfg = resolve_config("text_dialogue", "gemini-2.5-flash", role_name="Model Architect")
        pw_cfg = resolve_config("academic_writing", "gemini-2.5-flash", role_name="Paper Writer")

        self.agent_roles = {
            'Project Manager': AgentRoleFactory.create_project_manager(pm_cfg),
            'Domain Expert': AgentRoleFactory.create_domain_expert("Domain Expert", "Technical", de_cfg),
            'Critic': AgentRoleFactory.create_critic(cr_cfg),
            'Model Architect': AgentRoleFactory.create_architect(ar_cfg),
            'Paper Writer': AgentRoleFactory.create_paper_writer(pw_cfg)
        }

        def _create(name, role_type, cfg, mt=None):
            agent = AIAgent(
                name, role_type, cfg,
                conversation_type='discussion',
                max_tokens=mt or self.agent_roles[name].max_tokens
            )
            if self.socketio and self.session_id:
                agent.stream_callback = self._make_stream_callback(name)
            return agent

        moderator = _create("Project Manager", RoleType.MODERATOR, pm_cfg)
        expert1 = _create("Domain Expert", RoleType.EXPERT, de_cfg)
        analyst = _create("Critic", RoleType.ANALYST, cr_cfg)
        architect = _create("Model Architect", RoleType.EXPERT, ar_cfg)
        paper_writer = _create("Paper Writer", RoleType.EXPERT, pw_cfg)

        # Mirror the template-driven metadata so _should_skip_agent works uniformly
        moderator.participates_in = {"Problem Analysis", "Model Design", "Model Building", "Paper Writing"}
        moderator.functional_category = "general"
        expert1.participates_in = {"Problem Analysis", "Model Design"}
        expert1.functional_category = "analysis"
        analyst.participates_in = {"Problem Analysis", "Model Design", "Model Building", "Paper Writing"}
        analyst.functional_category = "general"
        architect.participates_in = {"Model Design", "Model Building"}
        architect.functional_category = "build"
        paper_writer.participates_in = {"Paper Writing"}
        paper_writer.functional_category = "write"

        self.agents = [moderator, expert1, analyst, architect, paper_writer]

        # v2: Write each agent's model config snapshot to manifest for reproducibility
        for name, cfg in (
            ("Project Manager", pm_cfg),
            ("Domain Expert", de_cfg),
            ("Critic", cr_cfg),
            ("Model Architect", ar_cfg),
            ("Paper Writer", pw_cfg),
        ):
            self.register_agent_spec(name, cfg)

    # -----------------------------------------------------------------
    # Team Template support
    # -----------------------------------------------------------------
    def load_team_template(self, template_id: str):
        """Load a builtin or custom template by id. Raises ValueError if missing."""
        from models.team_templates import find_template
        tpl = find_template(template_id)
        if tpl is None:
            raise ValueError(f"Team template not found: {template_id}")
        return tpl

    def apply_team_template(self, template, slot_model_overrides: Optional[Dict[str, str]] = None):
        """Replace self.agents with the roles from the given template.

        ``slot_model_overrides`` maps ``slot.role_name`` -> ``model_name``.
        If a slot is not in the overrides, ``slot.default_model`` is used.
        Records the template id and slot->model mapping in the manifest.
        """
        from models import RoleType
        from ai_agent import AIAgent
        from models.agent_roles import AgentRoleFactory
        from models.team_templates import AgentSlot
        from utils.manifest import AgentSpec

        slot_model_overrides = slot_model_overrides or {}

        new_agents = []
        new_specs: List[AgentSpec] = []

        for slot in template.slots:
            model_name = slot_model_overrides.get(slot.role_name) or slot.default_model
            try:
                role = AgentRoleFactory.create_from_slot(slot, model_name)
            except Exception as e:
                logger.error(f"create_from_slot failed for slot {slot.role_name}: {e}", exc_info=True)
                continue

            cfg = role.model_config or {}
            rt = self._infer_role_type(role.name)
            agent = AIAgent(
                role.name,
                rt,
                cfg,
                conversation_type='discussion',
                max_tokens=role.max_tokens,
            )
            # Store template-specific metadata on the agent object so phase routing works
            from models.team_templates import ALL_PHASES
            agent.participates_in = set(slot.phase_participation if slot.phase_participation else ALL_PHASES)
            agent.functional_category = slot.functional_category
            if self.socketio and self.session_id:
                agent.stream_callback = self._make_stream_callback(role.name)
            new_agents.append(agent)

            spec = AgentSpec(
                name=role.name,
                role=slot.role_name,
                model=cfg.get("model"),
                provider=cfg.get("provider"),
                base_url=cfg.get("base_url"),
                temperature=cfg.get("temperature"),
                max_tokens=cfg.get("max_tokens"),
                extra={"from_template": template.template_id, "slot_weight": slot.weight},
            )
            new_specs.append(spec)

        self.agents = new_agents
        # Update agent_roles map so role lookups elsewhere keep working
        from models.team_templates import ALL_PHASES
        self.agent_roles = {}
        for s in template.slots:
            model_name = slot_model_overrides.get(s.role_name) or s.default_model
            try:
                r = AgentRoleFactory.create_from_slot(s, model_name)
            except Exception:
                continue
            r.participates_in = set(s.phase_participation if s.phase_participation else ALL_PHASES)
            r.functional_category = s.functional_category
            self.agent_roles[r.name] = r

        # Record in manifest
        if self.manifest is not None:
            try:
                self.manifest.update_extra(
                    team_template_id=template.template_id,
                    team_template_label=template.label,
                    team_template_slots=[s.to_dict() for s in template.slots],
                    slot_model_overrides=slot_model_overrides,
                )
                # Replace any default agents with the new specs
                self.manifest.manifest.agents = new_specs
            except Exception as e:
                logger.warning(f"Failed to update manifest with team template: {e}")

        logger.info(f"Applied team template {template.template_id} with {len(new_agents)} agents")
        return new_agents

    @staticmethod
    def _infer_role_type(role_name: str):
        """Map role name to RoleType used by AIAgent."""
        from models import RoleType
        lower = (role_name or "").lower()
        if "critic" in lower or "reviewer" in lower or "methodologist" in lower:
            return RoleType.ANALYST
        if "manager" in lower or "lead" in lower or "pi" == lower:
            return RoleType.MODERATOR
        return RoleType.EXPERT

    def _make_stream_callback(self, agent_name):
        """Create a streaming callback that pushes token chunks via socketio"""
        if not (self.socketio and self.session_id):
            return None
        session_id = self.session_id
        socketio = self.socketio

        import uuid as _uuid
        message_id = _uuid.uuid4().hex
        started = [False]

        def callback(delta: str, full_text: str, finished: bool):
            try:
                if not started[0] and (delta or finished):
                    started[0] = True
                    socketio.emit('stream_start', {
                        'message_id': message_id,
                        'role': agent_name,
                        'turn': self.current_turn,
                    }, to=session_id, namespace='/')
                if delta:
                    socketio.emit('stream_chunk', {
                        'message_id': message_id,
                        'delta': delta,
                    }, to=session_id, namespace='/')
                if finished:
                    socketio.emit('stream_end', {
                        'message_id': message_id,
                        'role': agent_name,
                        'turn': self.current_turn,
                        'content': full_text,
                    }, to=session_id, namespace='/')
            except Exception as e:
                logger.error(f'Stream callback error: {e}')
        return callback

    async def run_meeting(self, stop_check_callback=None) -> Dict[str, Any]:
        """V2: Phased meeting workflow"""
        if self.message_callback:
            self.message_callback({'type': 'status', 'status': 'starting', 'message': f'Starting meeting: {self.topic}'})

        print(f"=== Starting meeting: {self.topic} ===")
        logger.info(f"Meeting topic: {self.topic}, auto-planning discussion rounds")

        self.context_file_manager.emit_context_files_event(self.message_callback)

        # Inject external context (e.g. file recognition results) at start
        for ctx in self._external_context:
            self.message_callback({
                'type': 'message',
                'role': 'System',
                'content': f"[External Context] {ctx}",
                'turn': self.current_turn,
                'link': None
            })

        # Drain pending user messages at start
        self._drain_user_messages_to_callback()

        try:
            if self.message_callback:
                self.message_callback({'type': 'phase', 'phase': 'Problem Analysis', 'message': 'Entering Problem Analysis phase'})
            await self._phase_problem_analysis(stop_check_callback)

            if stop_check_callback and stop_check_callback():
                raise InterruptedError("Meeting stopped by user")

            if self.message_callback:
                self.message_callback({'type': 'phase', 'phase': 'Model Design', 'message': 'Entering Model Design phase'})
            try:
                await self._phase_model_design(stop_check_callback)
                logger.info("Model Design phase returned, checking if stop is needed")
            except Exception as e:
                logger.error(f"Model Design phase execution error: {e}", exc_info=True)
                raise

            if stop_check_callback and stop_check_callback():
                raise InterruptedError("Meeting stopped by user")

            if self.message_callback:
                self.message_callback({'type': 'phase', 'phase': 'Model Building', 'message': 'Entering Model Building phase'})
            logger.info("Preparing to enter Model Building phase")
            try:
                await self._phase_model_building(stop_check_callback)
                logger.info("Model Building phase completed")
            except Exception as e:
                logger.error(f"Model Building phase execution error: {e}", exc_info=True)
                raise

            if stop_check_callback and stop_check_callback():
                raise InterruptedError("Meeting stopped by user")

            if self.message_callback:
                self.message_callback({'type': 'phase', 'phase': 'Paper Writing', 'message': 'Entering Paper Writing phase'})
            await self._phase_paper_writing(stop_check_callback)

            if stop_check_callback and stop_check_callback():
                raise InterruptedError("Meeting stopped by user")

            if self.message_callback:
                self.message_callback({'type': 'status', 'status': 'generating_report', 'message': 'Generating final report...'})
            final_report = await self._generate_final_report()

            self._save_conversation_to_file()

            if self.message_callback:
                self.message_callback({'type': 'status', 'status': 'completed', 'message': 'Meeting ended'})
                self.message_callback({'type': 'report', 'report': final_report})

            print(f"\n=== Meeting ended ===")
            self.save_manifest()
            return self._prepare_meeting_result(final_report)

        except InterruptedError as e:
            if self.message_callback:
                self.message_callback({'type': 'status', 'status': 'stopped', 'message': str(e)})
            print(f"\n=== Meeting stopped ===")
            self.save_manifest()
            return self._prepare_meeting_result("Meeting stopped by user")
        except Exception as e:
            if self.message_callback:
                self.message_callback({'type': 'status', 'status': 'error', 'message': f'Meeting error: {str(e)}'})
            logger.error(f'Meeting execution error: {e}', exc_info=True)
            print(f"\n=== Meeting error: {e} ===")
            raise

    async def _phase_problem_analysis(self, stop_check_callback=None):
        """Problem Analysis Phase: Domain Expert -> Critic -> Project Manager discuss in order for at least one round"""
        self.meeting_phase = "Problem Analysis"
        self.progress_tracker.start_phase(self.meeting_phase)
        self._reset_phase_control_state()
        
        turn = 0
        max_safe_turns = self.phase_turns.get("Problem Analysis", 20)
        
        while turn <= max_safe_turns:
            if stop_check_callback and stop_check_callback():
                return
            
            turn += 1
            self.progress_tracker.register_turn(self.meeting_phase)
            if self.message_callback:
                self.message_callback({'type': 'turn', 'turn': turn, 'max_turns': None})
            
            # Template-driven speaker order:
            # 1. Analysis expert (RoleType.EXPERT, functional_category="analysis")
            # 2. Analyst/Critic (RoleType.ANALYST)
            # 3. Moderator (RoleType.MODERATOR) — summary + advance decision
            moderator = next((agent for agent in self.agents if agent.role == RoleType.MODERATOR), None)
            analysis_expert = next((agent for agent in self.agents
                                   if agent.role == RoleType.EXPERT and
                                   getattr(agent, 'functional_category', None) == "analysis"), None)
            critic = next((agent for agent in self.agents if agent.role == RoleType.ANALYST), None)

            if analysis_expert and not self._should_skip_agent(analysis_expert):
                await self._agent_speak_with_focus(analysis_expert, "Problem Analysis", stop_check_callback=stop_check_callback)

            if critic and not self._should_skip_agent(critic):
                await self._agent_speak_with_focus(critic, "Problem Analysis", stop_check_callback=stop_check_callback)

            if moderator:
                await self._agent_speak_with_focus(moderator, "Problem Analysis", check_phase_advance=True, stop_check_callback=stop_check_callback)
            
            self.current_turn += 1
            
            if self._phase_hold_signal:
                self._phase_hold_signal = False
            
            logger.info(f"Problem Analysis phase completed round {turn}.")
            
            # If reached set number of rounds, automatically advance to next phase
            if turn >= max_safe_turns:
                logger.info(f"Problem Analysis phase reached set rounds ({max_safe_turns}), automatically advancing to next phase.")
                break
            
            # If Project Manager explicitly suggests advancing in summary, still requires user confirmation
            if self._phase_advance_signal:
                self._phase_advance_signal = False
                # Even with advance signal, request user confirmation
                logger.info("Project Manager suggests advancing, requesting user confirmation for next phase.")
                should_advance = await self._request_external_phase_confirmation("Problem Analysis", stop_check_callback)
                if should_advance:
                    logger.info("External confirmation received to advance from Problem Analysis to next phase.")
                    break
                else:
                    logger.info("External chose to continue Problem Analysis phase, entering next round.")
                    continue
            
            # After each round, request external confirmation on whether to advance
            logger.info(f"Problem Analysis phase round {turn} completed, requesting user confirmation for next phase.")
            should_advance = await self._request_external_phase_confirmation("Problem Analysis", stop_check_callback)
            if should_advance:
                logger.info("External confirmation received to advance from Problem Analysis to next phase.")
                break
            else:
                logger.info("External chose to continue Problem Analysis phase, entering next round.")


    async def _phase_model_design(self, stop_check_callback=None):
        """Model Design Phase: Analysis expert -> Build expert -> Critic -> Moderator speak in order.

        Both analysis (domain expert / senior researcher) and build (model architect /
        solution architect) experts are included because model design requires both
        problem understanding and technical architecture input.
        """
        logger.info("Starting Model Design phase execution")
        self.meeting_phase = "Model Design"
        self.progress_tracker.start_phase(self.meeting_phase)
        self._reset_phase_control_state()
        turn = 0
        max_safe_turns = self.phase_turns.get("Model Design", 20)

        while turn <= max_safe_turns:
            if stop_check_callback and stop_check_callback():
                break
            turn += 1
            self.progress_tracker.register_turn(self.meeting_phase)
            if self.message_callback:
                self.message_callback({'type': 'turn', 'turn': turn, 'max_turns': None})

            # Template-driven speaker order:
            # 1. Analysis expert (functional_category="analysis")
            # 2. Build expert (functional_category="build")
            # 3. Analyst/Critic (RoleType.ANALYST)
            # 4. Moderator (RoleType.MODERATOR) — summary + advance decision
            moderator = next((agent for agent in self.agents if agent.role == RoleType.MODERATOR), None)
            analysis_expert = next((agent for agent in self.agents
                                    if agent.role == RoleType.EXPERT and
                                    getattr(agent, 'functional_category', None) == "analysis"), None)
            build_expert = next((agent for agent in self.agents
                                 if agent.role == RoleType.EXPERT and
                                 getattr(agent, 'functional_category', None) == "build"), None)
            critic = next((agent for agent in self.agents if agent.role == RoleType.ANALYST), None)

            # 1. Analysis expert speaks
            if analysis_expert and not self._should_skip_agent(analysis_expert):
                await self._agent_speak_with_focus(analysis_expert, "Model Design", stop_check_callback=stop_check_callback)

            # 2. Build expert speaks
            if build_expert and not self._should_skip_agent(build_expert):
                await self._agent_speak_with_focus(build_expert, "Model Design", stop_check_callback=stop_check_callback)

            # 3. Critic speaks
            if critic and not self._should_skip_agent(critic):
                await self._agent_speak_with_focus(critic, "Model Design", stop_check_callback=stop_check_callback)

            # 4. Moderator speaks (summary and decision, check phase advance)
            if moderator and not self._should_skip_agent(moderator):
                await self._agent_speak_with_focus(moderator, "Model Design", check_phase_advance=True, stop_check_callback=stop_check_callback)

            self.current_turn += 1

            if self._phase_hold_signal:
                self._phase_hold_signal = False

            # If reached set number of rounds, automatically advance to next phase
            if turn >= max_safe_turns:
                logger.info(f"Model Design phase reached set rounds ({max_safe_turns}), automatically advancing to next phase.")
                logger.info("Model Design phase about to end, preparing to return")
                break

            # If Moderator explicitly suggests advancing in summary, still requires user confirmation
            if self._phase_advance_signal:
                self._phase_advance_signal = False
                logger.info("Moderator suggests advancing, requesting user confirmation for next phase.")
                should_advance = await self._request_external_phase_confirmation("Model Design", stop_check_callback)
                if should_advance:
                    logger.info("External confirmation received to advance from Model Design to next phase.")
                    logger.info("Model Design phase about to end, preparing to return")
                    break
                else:
                    logger.info("External chose to continue Model Design phase discussion.")
                    continue

            # After each round, request external confirmation on whether to advance
            logger.info(f"Model Design phase round {turn} completed, requesting user confirmation for next phase.")
            should_advance = await self._request_external_phase_confirmation("Model Design", stop_check_callback)
            if should_advance:
                logger.info("External confirmation received to advance from Model Design to next phase.")
                logger.info("Model Design phase about to end, preparing to return")
                break
            else:
                logger.info("External chose to continue Model Design phase discussion.")


        logger.info("Model Design phase execution completed, preparing to return")

    async def _phase_model_building(self, stop_check_callback=None):
        """Model Building Phase: Build expert -> Analyst/Critic -> Moderator speak in order.

        The build expert (functional_category="build", e.g. Model Architect or
        Solution Architect) is the primary content producer for this phase.
        """
        logger.info("Starting Model Building phase execution")
        self.meeting_phase = "Model Building"
        self.progress_tracker.start_phase(self.meeting_phase)
        self._reset_phase_control_state()
        turn = 0
        max_safe_turns = self.phase_turns.get("Model Building", 20)

        while turn <= max_safe_turns:
            if stop_check_callback and stop_check_callback():
                break
            turn += 1
            self.progress_tracker.register_turn(self.meeting_phase)
            if self.message_callback:
                self.message_callback({'type': 'turn', 'turn': turn, 'max_turns': None})

            # Template-driven speaker order:
            # 1. Build expert (functional_category="build")
            # 2. Analyst/Critic (RoleType.ANALYST)
            # 3. Moderator (RoleType.MODERATOR) — summary + advance decision
            moderator = next((agent for agent in self.agents if agent.role == RoleType.MODERATOR), None)
            build_expert = next((agent for agent in self.agents
                                 if agent.role == RoleType.EXPERT and
                                 getattr(agent, 'functional_category', None) == "build"), None)
            critic = next((agent for agent in self.agents if agent.role == RoleType.ANALYST), None)

            # 1. Build expert speaks
            if build_expert and not self._should_skip_agent(build_expert):
                await self._agent_speak_with_focus(build_expert, "Model Building", stop_check_callback=stop_check_callback)

            # 2. Critic speaks
            if critic and not self._should_skip_agent(critic):
                await self._agent_speak_with_focus(critic, "Model Building", stop_check_callback=stop_check_callback)

            # 3. Moderator speaks (summary and decision, check phase advance)
            if moderator and not self._should_skip_agent(moderator):
                await self._agent_speak_with_focus(moderator, "Model Building", check_phase_advance=True, stop_check_callback=stop_check_callback)

            self.current_turn += 1

            if self._phase_hold_signal:
                self._phase_hold_signal = False

            # If reached set number of rounds, automatically advance to next phase
            if turn >= max_safe_turns:
                logger.info(f"Model Building phase reached set rounds ({max_safe_turns}), automatically advancing to next phase.")
                break

            # If Moderator explicitly suggests advancing in summary, still requires user confirmation
            if self._phase_advance_signal:
                self._phase_advance_signal = False
                logger.info("Moderator suggests advancing, requesting user confirmation for next phase.")
                should_advance = await self._request_external_phase_confirmation("Model Building", stop_check_callback)
                if should_advance:
                    logger.info("External confirmation received to advance from Model Building to next phase.")
                    break
                else:
                    logger.info("External chose to continue Model Building phase discussion.")
                    continue

            # After each round, request external confirmation on whether to advance
            logger.info(f"Model Building phase round {turn} completed, requesting user confirmation for next phase.")
            should_advance = await self._request_external_phase_confirmation("Model Building", stop_check_callback)
            if should_advance:
                logger.info("External confirmation received to advance from Model Building to next phase.")
                break
            else:
                logger.info("External chose to continue Model Building phase discussion.")


    async def _phase_paper_writing(self, stop_check_callback=None):
        """Paper Writing Phase: Paper Writer -> Critic -> Project Manager speak in order, supports multiple rounds to generate complete 25-page paper"""
        self.meeting_phase = "Paper Writing"
        self.progress_tracker.start_phase(self.meeting_phase)
        self._reset_phase_control_state()
        turn = 0
        max_safe_turns = self.phase_turns.get("Paper Writing", 30)  # Increased max rounds to support 25-page paper generation (MCM template has 12 sections, at least 12 rounds needed)
        paper_sections = [
            ("Cover Page (Abstract + Keywords)", "Write the cover page including abstract and keywords. **IMPORTANT: The abstract must clearly specify which model is used for each problem (Problem 1, Problem 2, Problem 3). For example: 'For Problem 1, we use [model name/type]; for Problem 2, we use [model name/type]; for Problem 3, we use [model name/type].' The abstract should summarize the problem, methodology (including model specifications for each problem), and main results. Keywords should be relevant to the mathematical modeling competition."),
            ("Introduction (Background + Tasks + Methodology Overview + Flowchart)", "Write the introduction section including: background of the problem, task description, methodology overview, and a flowchart showing the overall approach."),
            ("Modeling Preparation (Assumptions + Notation + Data Preprocessing)", "Write the modeling preparation section including: model assumptions, notation/symbol definitions, and data preprocessing steps."),
            ("Problem 1 Modeling (Method + Model + Results + Figures/Tables)", "**IMPORTANT: Problem 1 Modeling must be the MOST DETAILED and COMPREHENSIVE section (suggested 6-8 pages).** Write the modeling section for Problem 1 with the most detailed content, including: comprehensive methodology, complete mathematical model establishment, detailed solution process, in-depth results analysis, and rich figures/tables (at least 3-5 figures/tables). This section should contain the most extensive model analysis, solution steps, result presentation, and discussion."),
            ("Problem 2 Modeling (Method + Model + Results + Figures/Tables)", "**IMPORTANT: Problem 2 Modeling should have MODERATE content (suggested 4-5 pages), less than Problem 1 but still detailed.** Write the modeling section for Problem 2 with moderate content, including: complete methodology, mathematical model establishment, solution process, results analysis, and supporting figures/tables (at least 2-3 figures/tables)."),
            ("Problem 3 Modeling / New Insights (Originality Analysis + Recommendations)", "**IMPORTANT: Problem 3 Modeling should have the LEAST content (suggested 2-3 pages), less than Problem 2.** Write the modeling section for Problem 3 or new insights section with concise content, including: model establishment, results presentation (at least 1-2 figures/tables), originality analysis, and recommendations."),
            ("Problem 4 Modeling (Method + Model + Results + Figures/Tables)", "**IMPORTANT: Problem 4 Modeling should be COMPREHENSIVE and DETAILED (suggested 4-6 pages).** Write the modeling section for Problem 4 with detailed content, including: complete methodology, mathematical model establishment, solution process, results analysis, and supporting figures/tables (at least 2-4 figures/tables). This section should provide thorough analysis and discussion."),
            ("Sensitivity / Robustness Analysis", "Write the sensitivity and robustness analysis section, analyzing how model results change with parameter variations."),
            ("Model Evaluation (Advantages + Disadvantages)", "Write the model evaluation section, discussing the advantages and disadvantages of the proposed model."),
            ("Memo (Letter to Decision Makers)", "Write a memo/letter to decision makers, summarizing key findings and recommendations in a clear, executive-friendly format."),
            ("References", "Compile and format all references used in the paper according to academic citation standards."),
            ("AI Usage Report (if applicable)", "If AI tools were used, write a report documenting their usage, including which tools were used and how they contributed to the work."),
        ]
        
        while turn <= max_safe_turns:
            if stop_check_callback and stop_check_callback():
                break
            turn += 1
            self.progress_tracker.register_turn(self.meeting_phase)
            if self.message_callback:
                self.message_callback({'type': 'turn', 'turn': turn, 'max_turns': None})
            
            section_idx = min(turn - 1, len(paper_sections) - 1)
            section_title, section_scope = paper_sections[section_idx]
            shared_turn_context = (
                f"【Paper Writing Phase Round {turn}】\n"
                f"- Current Section: {section_title}\n"
                f"- Scope Requirements: {section_scope}\n"
                "Important: Only handle the above section in this round. Do not write the entire paper or expand to subsequent sections prematurely."
            )
            
            # Template-driven speaker order:
            # 1. Write expert (functional_category="write", e.g. Paper Writer or Report Writer)
            # 2. Analyst/Critic (RoleType.ANALYST)
            # 3. Moderator (RoleType.MODERATOR) — summary + advance decision
            moderator = next((agent for agent in self.agents if agent.role == RoleType.MODERATOR), None)
            write_expert = next((agent for agent in self.agents
                                 if agent.role == RoleType.EXPERT and
                                 getattr(agent, 'functional_category', None) == "write"), None)
            critic = next((agent for agent in self.agents if agent.role == RoleType.ANALYST), None)

            # 1. Write expert speaks (write or refine paper)
            if write_expert and not self._should_skip_agent(write_expert):
                # Add special guidance based on round
                special_guidance = ""
                if turn == 4:  # Problem 1 Modeling
                    special_guidance = "\n**【Core Requirement】Problem 1 Modeling section must be the most detailed (suggested 6-8 pages):**\n- Include the most detailed model establishment process, complete solution steps, in-depth result analysis\n- Provide rich figures/tables (at least 3-5)\n- Include detailed discussion and explanations\n- This is the core section of the paper and must have the largest portion\n"
                elif turn == 5:  # Problem 2 Modeling
                    special_guidance = "\n**【Core Requirement】Problem 2 Modeling section should have moderate content (suggested 4-5 pages), less than Problem 1 but still detailed:**\n- Include complete model establishment, solution process and result analysis\n- Provide figures/tables (at least 2-3)\n- Content should be less than Problem 1 but still detailed\n"
                elif turn == 6:  # Problem 3 Modeling
                    special_guidance = "\n**【Core Requirement】Problem 3 Modeling/New Insights section should have the least content (suggested 2-3 pages), less than Problem 2:**\n- Include model establishment, results presentation (at least 1-2 figures/tables) and originality analysis\n- Content should be less than Problem 2\n"
                elif turn == 7:  # Problem 4 Modeling
                    special_guidance = "\n**【Core Requirement】Problem 4 Modeling section should be detailed (suggested 4-6 pages):**\n- Include complete model establishment, solution process and result analysis\n- Provide figures/tables (at least 2-4)\n- Content should be detailed but slightly less than Problem 1\n"
                
                writer_guidance = (
                    f"{shared_turn_context}\n"
                    f"{special_guidance}"
                    "Writing Strategy:\n"
                    "1. First provide the section heading, then output the complete LaTeX content for this section.\n"
                    "2. **【Important】Must use Markdown code block format for LaTeX code (wrapped with ```latex or ```)**\n"
                    "3. **Output Format Example:**\n"
                    "   ```latex\n"
                    "   \\section{Section Title}\n"
                    "   Content here...\n"
                    "   ```\n"
                    "4. Maintain logical progression within paragraphs, can use formulas/tables, but only within this section scope.\n"
                    "5. **【Important】Must use formulas where mathematical explanations are needed:**\n"
                    "   - Model establishment process must be expressed with mathematical formulas (objective function, constraints, state equations, etc.)\n"
                    "   - Solution algorithms and steps must be explained with mathematical formulas (iteration formulas, optimization process, etc.)\n"
                    "   - Mathematical relationships in result analysis and discussion must be expressed with formulas\n"
                    "   - Do not describe mathematical relationships with plain text, must use LaTeX math formulas ($...$ or \\[...\\])\n"
                    "6. If referencing previous work, cite completed sections, do not rewrite.\n"
                    "7. Output length should sufficiently cover this section, at least 500 characters, but do not expand to other sections."
                )
                await self._agent_speak_with_focus(
                    write_expert,
                    "Paper Writing",
                    per_turn_guidance=writer_guidance,
                    stop_check_callback=stop_check_callback
                )
            
            # 2. Critic speaks (evaluate paper quality, provide feedback and improvement suggestions)
            if critic and not self._should_skip_agent(critic):
                critic_guidance = (
                    f"{shared_turn_context}\n"
                    "Review Strategy:\n"
                    "1. Only evaluate this round's section structure, logic, mathematical rigor and readability.\n"
                    "2. List points that need supplementation or correction, avoid requesting early completion of other sections.\n"
                    "3. Point out connection issues with previous sections, ensure natural transitions.\n"
                    "4. If out-of-scope writing is found, clearly indicate and request splitting to subsequent rounds."
                )
                await self._agent_speak_with_focus(
                    critic,
                    "Paper Writing",
                    per_turn_guidance=critic_guidance,
                    stop_check_callback=stop_check_callback
                )
            
            # 3. Project Manager speaks (summary and decision, check phase advance)
            if moderator and not self._should_skip_agent(moderator):
                pm_guidance = (
                    f"{shared_turn_context}\n"
                    "Project Management Requirements:\n"
                    "1. Summarize whether this section meets completion standards, list gaps.\n"
                    "2. Clarify next round's section focus, remind team to maintain iterative section-based approach.\n"
                    "3. Only issue advance directive when all sections are complete and meet 25-page requirement.\n"
                    "4. If section has not met standards, request Paper Writer to continue in next round."
                )
                await self._agent_speak_with_focus(
                    moderator,
                    "Paper Writing",
                    check_phase_advance=True,
                    per_turn_guidance=pm_guidance,
                    stop_check_callback=stop_check_callback
                )
            
            self.current_turn += 1

            if self._phase_hold_signal:
                self._phase_hold_signal = False

            # If reached set number of rounds, automatically advance (Paper Writing is the last phase, meeting will end)
            if turn >= max_safe_turns:
                logger.info(f"Paper Writing phase reached set rounds ({max_safe_turns}), automatically completing meeting and generating final report.")
                break

            # If Project Manager explicitly suggests advancing in summary, still requires user confirmation
            if self._phase_advance_signal:
                self._phase_advance_signal = False
                # Even with advance signal, request user confirmation
                logger.info("Project Manager suggests advancing, requesting user confirmation to complete meeting and generate final report.")
                should_advance = await self._request_external_phase_confirmation("Paper Writing", stop_check_callback)
                if should_advance:
                    logger.info("External confirmation received to complete meeting and generate final report.")
                    break
                else:
                    logger.info("External chose to continue Paper Writing phase discussion, continuing to refine paper.")
                    continue

            # After each round, request external confirmation on whether to advance
            logger.info(f"Paper Writing phase round {turn} completed, requesting user confirmation to complete meeting and generate final report.")
            should_advance = await self._request_external_phase_confirmation("Paper Writing", stop_check_callback)
            if should_advance:
                logger.info("External confirmation received to complete meeting and generate final report.")
                break
            else:
                logger.info("External chose to continue Paper Writing phase discussion, continuing to refine paper.")


    def _should_skip_agent(self, agent: AIAgent) -> bool:
        """Determine whether to skip an agent based on the current phase.

        Uses the agent's ``participates_in`` set (set by apply_team_template or
        setup_default_agents) when available.  Falls back to the legacy
        name-based rules only for default agents that were created without a
        template and therefore lack that attribute.
        """
        current_phase = self.meeting_phase

        if hasattr(agent, 'participates_in') and agent.participates_in is not None:
            # Template-driven routing: skip if the current phase is not in the set
            return current_phase not in agent.participates_in

        # -----------------------------------------------------------------
        # Legacy fallback: only for default agents created by setup_default_agents()
        # -----------------------------------------------------------------
        if agent.name == "Paper Writer":
            return current_phase != "Paper Writing"
        if agent.name == "Critic":
            return current_phase not in {"Problem Analysis", "Model Design", "Model Building", "Paper Writing"}
        if agent.name == "Model Architect":
            return current_phase not in {"Model Design", "Model Building"}
        if agent.name == "Domain Expert":
            return current_phase not in {"Problem Analysis", "Model Design"}
        # Unrecognised name → participate (don't skip)
        return False

    def _reset_phase_control_state(self):
        self.awaiting_external_confirmation = False
        self.pending_phase_transition = None
        self._phase_advance_signal = False
        self._phase_hold_signal = False
        self._phase_confirmation_event = None
        self._phase_confirmation_result = None

    def _next_phase_name(self, phase: str) -> Optional[str]:
        if phase not in self.phase_order:
            return None
        idx = self.phase_order.index(phase)
        if idx + 1 < len(self.phase_order):
            return self.phase_order[idx + 1]
        return None

    async def _request_external_phase_confirmation(self, phase: str, stop_check_callback=None) -> bool:
        """
        Request external (frontend UI) confirmation for phase advancement.
        Returns True for advancing, False for continuing current phase.
        If no confirmation received within 30 seconds, automatically advances (to avoid getting stuck).
        :param stop_check_callback: stop check callback function
        """
        # Check stop request before waiting
        if stop_check_callback and stop_check_callback():
            raise InterruptedError("Meeting stopped by user")
        
        self.awaiting_external_confirmation = True
        self.pending_phase_transition = phase
        next_phase = self._next_phase_name(phase)
        
        # Paper Writing is the last phase, completing it will end the meeting
        is_final_phase = (next_phase is None)
        
        # Create event for waiting confirmation
        self._phase_confirmation_event = asyncio.Event()
        self._phase_confirmation_result = None
        
        # Notify frontend that confirmation is needed
        self._notify_control_event('phase_decision_required', {
            'turn': self.current_turn,
            'phase': phase,
            'next_phase': next_phase,
            'is_final_phase': is_final_phase
        })
        if is_final_phase:
            logger.info(f"{phase} phase reached switching condition, waiting for external confirmation to complete meeting and generate final report.")
        else:
            logger.info(f"{phase} phase reached switching condition, waiting for external confirmation to enter {next_phase}.")
        
        # Wait for confirmation, check stop request and confirmation status every 0.5 seconds
        check_interval = 0.5
        while True:
            if stop_check_callback and stop_check_callback():
                raise InterruptedError("Meeting stopped by user")
            
            if self._phase_confirmation_event.is_set():
                break
            
            await asyncio.sleep(check_interval)
        
        # Received confirmation
        if self._phase_confirmation_result is None:
            logger.warning("Received phase advancement confirmation, but result is empty, defaulting to advancing to next phase")
            result = True  # Default to advance, since system has already determined it should advance
        else:
            result = self._phase_confirmation_result
        
        self.awaiting_external_confirmation = False
        self.pending_phase_transition = None
        self._phase_confirmation_event = None
        self._phase_confirmation_result = None
        
        return result
    
    def handle_external_phase_confirmation(self, advance: bool):
        """
        Handle external phase advancement confirmation.
        advance: True means advance to next phase, False means continue current phase
        """
        if not self.awaiting_external_confirmation:
            logger.warning("Received phase advancement confirmation, but currently not waiting for confirmation")
            return False
        
        self._phase_confirmation_result = advance
        if self._phase_confirmation_event:
            self._phase_confirmation_event.set()
        
        next_phase = self._next_phase_name(self.pending_phase_transition or self.meeting_phase)
        if advance:
            self._notify_control_event('phase_advance_confirmed', {
                'turn': self.current_turn,
                'phase': self.pending_phase_transition or self.meeting_phase,
                'next_phase': next_phase
            })
            logger.info(f"External confirmation received to enter {next_phase or 'next phase'}")
        else:
            self._notify_control_event('phase_extend_requested', {
                'turn': self.current_turn,
                'phase': self.pending_phase_transition or self.meeting_phase
            })
            logger.info(f"External chose to continue current phase discussion")
        
        return True

    def _notify_control_event(self, action: str, payload: Dict[str, Any]):
        if self.message_callback:
            self.message_callback({
                'type': 'control',
                'action': action,
                **payload
            })

    async def _agent_speak_with_focus(
        self,
        agent,
        phase_context: str = None,
        check_phase_advance: bool = True,
        per_turn_guidance: str = None,
        stop_check_callback=None,
    ):
        """
        Agent speaking method.
        :param agent: agent object
        :param phase_context: phase context
        :param check_phase_advance: whether to check phase advance (only for Project Manager)
        :param per_turn_guidance: extra prompt for the current round (used for section-level guidance)
        :param stop_check_callback: stop check callback
        """
        # Check stop request before starting
        if stop_check_callback and stop_check_callback():
            raise InterruptedError("Meeting stopped by user")

        effective_phase = phase_context or self.meeting_phase

        # Pull any pending user messages and external context (vision recognition) before this turn
        self._drain_user_messages_to_callback()
        external_ctx_items = self._consume_external_context()
        external_ctx_block = ""
        if external_ctx_items:
            joined = "\n\n".join(external_ctx_items)
            external_ctx_block = (
                "\n\n[External Context - e.g. file recognition results]\n"
                f"{joined}\n\n"
                "**Important**: The above external context has been added during the meeting. "
                "Please reference it in your analysis and decisions."
            )

        phase_prompt = self._build_phase_prompt(agent.role, effective_phase, agent.name)
        context = self.vector_memory.get_context_for_agent(self.topic, agent.role.value, self.current_turn)
        solution_summary = self.solution_tracker.get_solutions_summary()

        # Add pre-meeting reference file content (ensure all models can read files)
        context_files_content = ""
        if self.context_file_manager.context_files_info:
            context_files_content = self.context_file_manager.format_context_files_content()
            if context_files_content:
                context_files_content = f"\n\n{context_files_content}\n\n**Important**: Please carefully read the above pre-meeting reference files - they contain critical background data and information to inform your analysis and decisions."

        # If an AgentRole exists, use its prompt context to enrich the system prompt
        agent_role_context = ""
        if agent.name in self.agent_roles:
            agent_role = self.agent_roles[agent.name]
            # Create a temporary TaskTemplate for role context generation
            from models.task_template import TaskTemplate, TaskType
            temp_template = TaskTemplate(
                task_type=TaskType.PROBLEM_SOLVING,
                background=f"User submitted problem: {self.topic}",
                goal=f"Solve the following problem: {self.topic}",
                constraints=[],
                output_format="",
                requirements=[]
            )
            agent_role_context = agent_role.get_prompt_context(temp_template, phase_context or self.meeting_phase)
            # Add team info and responsibility boundaries
            team_boundaries = self._get_team_info_and_boundaries(agent.name)
            if team_boundaries:
                agent_role_context = f"{agent_role_context}\n\n{team_boundaries}"

        base_context = f"{phase_prompt}\n\n{context}\n\n{solution_summary}{context_files_content}{external_ctx_block}"

        full_context = f"{agent_role_context}\n\n{base_context}" if agent_role_context else base_context
        moderator_name = self._get_actual_role_name("Project Manager")
        if agent.name == moderator_name:
            full_context = (
                f"{full_context}\n\n"
                "Output Format: Please strictly organize your entire response in Markdown format, "
                "with clear headings, lists, or tables to ensure readability."
            )
        if per_turn_guidance:
            full_context = f"{full_context}\n\n{per_turn_guidance}"

        conversation_context = self._build_conversation_context(agent.name)

        # Check stop request before calling AI
        if stop_check_callback and stop_check_callback():
            raise InterruptedError("Meeting stopped by user")

        try:
            response = await agent.generate_response(
                [Message("system", full_context, time.time(), 0)] + conversation_context,
                self.topic
            )
        except Exception as e:
            # Handle API call failure
            error_msg = str(e)
            logger.error(f"{agent.name} failed to generate response: {error_msg}", exc_info=True)

            # If the error message already contains details (from _call_llm_api), use it directly
            # otherwise build a simple error message
            if error_msg.startswith(f"[{agent.name}") and ("API call failed" in error_msg):
                # Already detailed, use directly
                response = f"⚠️ {error_msg}"
            elif "API call failed" in error_msg:
                # Simplified error message with prefix
                response = f"⚠️ **{agent.name} API call failed**\n\n{error_msg}\n\nPlease check network connection, API configuration, or try again later."
            elif "timeout" in error_msg.lower() or "Request timed out" in error_msg:
                response = f"⚠️ **{agent.name} API call timeout**\n\nThe request took too long. The prompt may be too large or there is a network issue. Please try again later.\n\nDetails: {error_msg}"
            else:
                response = f"⚠️ **{agent.name} generation error**\n\n{error_msg}\n\nPlease try again later."

            # Send error message to user
            if self.message_callback:
                self.message_callback({
                    'type': 'error',
                    'agent': agent.name,
                    'message': response
                })

        # Check stop request immediately after AI response
        if stop_check_callback and stop_check_callback():
            raise InterruptedError("Meeting stopped by user")

        if getattr(agent, 'functional_category', None) == "write":
            response = await self._enforce_latex_output(agent, full_context, conversation_context, response, stop_check_callback)
            # Check stop request again
            if stop_check_callback and stop_check_callback():
                raise InterruptedError("Meeting stopped by user")

        # Simple repetition detection and one-time regeneration
        previous = self._last_agent_output.get(agent.name, "")
        if previous:
            sim_ratio = difflib.SequenceMatcher(None, previous, response).ratio()
            # Only strictly judge longer texts to avoid false positives for short texts
            if max(len(previous), len(response)) >= 100 and sim_ratio >= 0.92:
                # Check stop request before regeneration
                if stop_check_callback and stop_check_callback():
                    raise InterruptedError("Meeting stopped by user")
                
                refine_prompt_suffix = (
                    "\n\nPlease avoid repeating the previous round's content. Based on the points already given, supplement new perspectives, refine actionable steps,"
                    "or advance to the next decision point. Provide new information, do not restate."
                )
                refined_context = full_context + refine_prompt_suffix
                refined = await agent.generate_response(
                    [Message("system", refined_context, time.time(), 0)] + self.conversation[-10:],
                    self.topic
                )
                # Check stop request after regeneration
                if stop_check_callback and stop_check_callback():
                    raise InterruptedError("Meeting stopped by user")
                
                refined_ratio = difflib.SequenceMatcher(None, previous, refined).ratio()
                if max(len(previous), len(refined)) >= 100 and refined_ratio >= 0.92:
                    # Still similar, give "supplementary points" guidance to encourage model to add different information
                    refined = "Supplementary points (avoid repetition):\n" + refined
                response = refined

        # Track solutions
        if any(keyword in response for keyword in ['plan', 'solution', 'method', 'suggestion', 'step', 'approach', 'strategy', 'implement', 'execute', 'proposal']):
            self.solution_tracker.add_solution(response, agent.name, self.current_turn)
        self._add_message(
            agent.name,
            response,
            model_id=agent.model_config.get("model"),
            provider=agent.model_config.get("provider"),
        )
        self.vector_memory.add_conversation(agent.name, response, self.current_turn, agent.conversation_type)
        self._last_agent_output[agent.name] = response
        moderator_name = self._get_actual_role_name("Project Manager")
        if agent.name == moderator_name and not self.context_file_manager._context_files_announced:
            self.context_file_manager._context_files_announced = True
        self.progress_tracker.note_message(self.meeting_phase, agent.name, response)
        print(f"\n【{agent.name}】{response}")

        await asyncio.sleep(self.THINKING_PAUSE_SECONDS)

    def _build_conversation_context(self, agent_name: str) -> List[Message]:
        """
        Control conversation history length for different roles to avoid hitting model input limits.
        Paper Writer retains more history, but enforces message count and character limits.
        """
        if agent_name != "Paper Writer":
            return self.conversation[-10:]

        if not self.conversation:
            return []

        selected: List[Message] = []
        char_count = 0

        for msg in reversed(self.conversation):
            msg_content = msg.content or ""
            msg_length = len(msg_content)

            if selected and (
                len(selected) >= self.PAPER_WRITER_MAX_HISTORY_MESSAGES
                or char_count + msg_length > self.PAPER_WRITER_MAX_HISTORY_CHARS
            ):
                break

            if not selected and msg_length > self.PAPER_WRITER_MAX_HISTORY_CHARS:
                truncated_content = msg_content[: self.PAPER_WRITER_MAX_HISTORY_CHARS]
                selected.append(
                    Message(
                        role=msg.role,
                        content=truncated_content,
                        timestamp=msg.timestamp,
                        turn=msg.turn,
                    )
                )
                char_count = len(truncated_content)
                break

            selected.append(msg)
            char_count += msg_length

        trimmed_history = list(reversed(selected))

        if len(trimmed_history) < len(self.conversation):
            notice = Message(
                role="system",
                content=(
                    "⚠️ Conversation history truncated, retaining only the most recent "
                    f"{len(trimmed_history)} messages (approximately {char_count} characters) for paper writing. "
                    "For earlier discussions, refer to phase summaries and solution overviews."
                ),
                timestamp=time.time(),
                turn=self.current_turn,
            )
            return [notice] + trimmed_history

        return trimmed_history

    def _contains_latex_block(self, text: str) -> bool:
        """Check if text contains LaTeX code blocks"""
        if not text:
            return False
        # Check for ```latex or ``` code blocks
        return bool(re.search(r"```(?:latex)?[\s\S]+?```", text))

    def _is_pure_latex(self, text: str) -> bool:
        """Check if text is already in pure LaTeX format (contains LaTeX commands)"""
        if not text:
            return False
        
        # Check for common LaTeX commands
        latex_indicators = [
            r"\\documentclass",
            r"\\begin\{document\}",
            r"\\section\{",
            r"\\subsection\{",
            r"\\begin\{equation\}",
            r"\\begin\{figure\}",
            r"\\begin\{table\}",
            r"\\\[",
            r"\\\]",
            r"\$[^$]+\$",  # Inline formulas
        ]
        
        for indicator in latex_indicators:
            if re.search(indicator, text):
                return True
        
        return False

    async def _enforce_latex_output(self, agent: AIAgent, base_context: str, conversation_context: List[Message], response: str, stop_check_callback=None) -> str:
        """Ensure Paper Writer outputs LaTeX code in code block format"""
        if agent.name != "Paper Writer":
            return response
        
        # If output already contains LaTeX code blocks, return directly
        if self._contains_latex_block(response):
            logger.info("Detected Paper Writer output contains LaTeX code block, format is correct.")
            return response
        
        # If output is pure LaTeX format, wrap it in code blocks
        if self._is_pure_latex(response):
            logger.info("Detected Paper Writer output is pure LaTeX format, auto-wrapping as code block.")
            return f"```latex\n{response}\n```"
        
        # If output is neither code blocks nor pure LaTeX, try to reinforce prompt
        # Check stop request before retry
        if stop_check_callback and stop_check_callback():
            raise InterruptedError("Meeting stopped by user")
        
        logger.warning("Paper Writer output missing LaTeX format, triggering LaTeX enforcement prompt.")
        enforcement_suffix = (
            "\n\n⚠️ **Strict Requirement: You must immediately output complete LaTeX formatted paper content in Markdown code block format.**\n"
            "**Output Requirements:**\n"
            "1. **Must use Markdown code block format for LaTeX code (wrapped with ```latex or ```)**\n"
            "2. If outputting complete paper, must include \\documentclass, \\begin{document}, \\end{document} and other LaTeX document structure\n"
            "3. If outputting paper fragments, must use standard LaTeX commands (\\section{}, \\subsection{}, $...$, \\[...\\], etc.)\n"
            "4. All mathematical formulas must use LaTeX math formula syntax ($...$ for inline, \\[...\\] for display)\n"
            "5. Ensure output LaTeX code is complete and compilable\n"
            "6. **Must use code block format for output, for example:**\n"
            "   ```latex\n"
            "   \\documentclass{article}\n"
            "   \\begin{document}\n"
            "   ...\n"
            "   \\end{document}\n"
            "   ```\n"
            "Please re-output, ensuring LaTeX code is wrapped in code blocks."
        )
        retry_context = base_context + enforcement_suffix
        retry_response = await agent.generate_response(
            [Message("system", retry_context, time.time(), 0)] + conversation_context,
            self.topic
        )
        
        # Check stop request after retry
        if stop_check_callback and stop_check_callback():
            raise InterruptedError("Meeting stopped by user")
        
        # Check output after retry again
        if self._contains_latex_block(retry_response):
            logger.info("Detected LaTeX code block after retry, format is correct.")
            return retry_response
        
        if self._is_pure_latex(retry_response):
            logger.info("Detected pure LaTeX format after retry, auto-wrapping as code block.")
            return f"```latex\n{retry_response}\n```"
        
        # If still not LaTeX format, try auto-wrapping as code block
        logger.warning("Paper Writer output still not standard LaTeX format, auto-wrapping as code block.")
        return f"```latex\n{retry_response}\n```"

    def _get_actual_role_name(self, canonical: str) -> str:
        """Map a canonical MCM/ICM role name to the actual name in the current template.

        Returns the first matching agent's name, falling back to the canonical name
        when no template has been applied (i.e. using the default competition team).
        """
        mapping = {
            "Project Manager": (RoleType.MODERATOR, None),
            "Domain Expert":   (RoleType.EXPERT, "analysis"),
            "Model Architect":  (RoleType.EXPERT, "build"),
            "Paper Writer":    (RoleType.EXPERT, "write"),
            "Critic":         (RoleType.ANALYST, None),
        }
        if canonical not in mapping:
            return canonical
        target_role, target_cat = mapping[canonical]
        for agent in self.agents:
            if agent.role == target_role:
                if target_cat is None or getattr(agent, 'functional_category', None) == target_cat:
                    return agent.name
        return canonical  # fallback: use canonical name

    def _get_team_info_and_boundaries(self, current_agent_name: str) -> str:
        """Generate team info and role responsibility boundaries"""
        if not self.agent_roles:
            return ""
        
        # Get all role information
        team_members = []
        current_role_info = None
        
        for role_name, role_def in self.agent_roles.items():
            role_desc = f"- **{role_name}** (weight: {role_def.weight}): {role_def.description}"
            responsibilities = "\n  ".join([f"  - {resp}" for resp in role_def.responsibilities])
            role_info = f"{role_desc}\n  Main responsibilities:\n{responsibilities}"
            team_members.append(role_info)
            
            if role_name == current_agent_name:
                current_role_info = role_def
        
        team_info = "\n".join(team_members)
        
        # Generate responsibility boundary descriptions
        if current_role_info:
            other_roles = [name for name in self.agent_roles.keys() if name != current_agent_name]
            
            # Add specific collaboration relationship descriptions based on current functional category
            current_cat = getattr(current_role_info, 'functional_category', None) or "general"
            build_name = self._get_actual_role_name("Model Architect")
            analysis_name = self._get_actual_role_name("Domain Expert")
            moderator_name = self._get_actual_role_name("Project Manager")

            connection_rules = ""
            if current_cat == "build":
                connection_rules = f"""
   - **Must wait for {analysis_name} to clearly state modeling requirements before starting architecture work**
   - **Must strictly follow {analysis_name}'s modeling requirements and constraints when architecting models**
   - **If {analysis_name} has not yet provided clear modeling requirements, should proactively ask or wait**
"""
            elif current_cat == "analysis":
                connection_rules = f"""
   - **Must provide clear modeling requirements and constraints for {build_name}**
   - **When proposing modeling requirements, be clear, specific, and actionable**
"""
            elif current_cat == "general" and current_agent_name == moderator_name:
                connection_rules = """
   - **Your requirements must be responded to and followed by all roles**
   - **All roles must respond to your instructions and decisions**
"""
            
            boundaries = f"""
**Important: Responsibility Boundaries and Collaboration Principles**

1. **Your Core Responsibilities**:
{chr(10).join([f"   - {resp}" for resp in current_role_info.responsibilities])}

2. **Other Roles in Team** (sorted by weight):
{team_info.replace(f"- **{current_agent_name}**", f"- ~~{current_agent_name}~~ (yourself)")}

3. **Strictly Follow Principles**:
   - **Focus on your own**: Only do work within your responsibility scope
   - **Do not overstep**: Do not complete other roles' ({', '.join(other_roles)}) work for them
   - **Collaborate, not replace**: If you need other roles' work, propose requirements or suggestions, do not complete it yourself
   - **Clear division of labor**: If you notice content that other roles should handle, you may remind or suggest, but do not do it for them
   - **Stay focused**: Always focus on your core responsibilities, ensure high-quality completion of your own work
   - **Respect weight**: Requirements from higher-weight roles have higher priority
{connection_rules}
4. **Collaboration Methods**:
   - Can build your work based on other roles' outputs
   - Can propose suggestions or point out issues, but do not directly complete other roles' work
   - If other roles need to be involved, you may suggest or remind, but do not replace their work
   - **Project Manager's requirements must be responded to and followed by all roles**
"""
        else:
            boundaries = ""
        
        return boundaries

    def _get_meeting_workflow_info(self, current_phase: str, agent_name: str = None) -> str:
        """Generate meeting workflow description"""
        phase_descriptions = {
            "Problem Analysis": {
                "description": "In-depth analysis of the problem, identifying core issues and key challenges, abstracting problems from a mathematical modeling perspective",
                "participants": ["Domain Expert", "Critic", "Project Manager"],
                "min_turns": 1,
                "turn_tasks": {
                    1: "Domain Expert performs problem abstraction, defining variables/parameters/sets, inputs/outputs, constraint boundaries and objectives; Critic evaluates and questions, identifying potential risks and blind spots; Project Manager summarizes insights and judges whether to advance to Model Design phase"
                },
                "key_tasks": [
                    "Domain Expert: Define variables/parameters/sets, inputs/outputs, constraint boundaries and objectives",
                    "Critic: Question adequacy of problem decomposition, identify potential risks and blind spots",
                    "Project Manager: Summarize insights, clarify phase conclusions and next decision inputs"
                ],
                "completion_criteria": [
                    "Domain Expert has completed problem abstraction from mathematical modeling perspective and clarified modeling objectives",
                    "Critic has provided questions and supplements on problem definition, assumptions and risks",
                    "Key assumptions, data requirements and uncertainty sources have been identified",
                    "Project Manager has confirmed sufficient input preparation to enter Model Design phase"
                ],
                "speech_order": "Domain Expert performs problem abstraction -> Critic evaluates and questions -> Project Manager summarizes and judges whether to advance"
            },
            "Model Design": {
                "description": "Design executable mathematical model solutions, select model paradigm and explain selection rationale",
                "participants": ["Project Manager", "Domain Expert", "Critic", "Model Architect"],
                "min_turns": 2,
                "turn_tasks": {
                    1: "Domain Expert proposes model design solution, selects model paradigm (optimization, differential equations, probabilistic graphical, game theory, simulation, reinforcement learning, etc.), provides objective function/constraints/state equations; Critic evaluates and questions feasibility and effectiveness of solution; Project Manager coordinates discussion",
                    2: "Based on first round feedback, Domain Expert refines design solution, proposes parameter calibration and cross-validation/sensitivity analysis process; Critic further evaluates; Project Manager summarizes and judges whether to advance to Model Building phase"
                },
                "key_tasks": [
                    "Select model paradigm (optimization, differential equations, probabilistic graphical, game theory, simulation, reinforcement learning, etc.)",
                    "Provide objective function/constraints/state equations/transition and observation models",
                    "Design solution algorithm and complexity evaluation",
                    "Propose parameter calibration and cross-validation/sensitivity analysis process"
                ],
                "completion_criteria": [
                    "Domain Expert has provided clear design solution and modeling requirements",
                    "Critic has evaluated and questioned the solution",
                    "Solution is clear and complete enough to provide clear modeling requirements and constraints for Model Building phase",
                    "Project Manager has confirmed solution meets conditions for entering Model Building phase"
                ],
                "speech_order": "Domain Expert proposes solution -> Critic evaluates and questions -> Project Manager (summarizes phase and judges whether to advance)"
            },
            "Model Building": {
                "description": "Build complete mathematical models, evaluate model solutions and output implementation roadmap",
                "participants": ["Project Manager", "Model Architect", "Critic"],
                "min_turns": 2,
                "turn_tasks": {
                    1: "Model Architect strictly follows Domain Expert's modeling requirements to perform mathematical modeling, building complete mathematical models; Critic evaluates model solutions from dimensions of solvability, robustness, generalizability, interpretability, etc.; Project Manager coordinates discussion",
                    2: "Based on first round feedback, Model Architect refines models, outputs implementation roadmap (data preparation, parameter calibration, solver configuration, etc.), provides evaluation metrics and acceptance criteria; Critic further evaluates; Project Manager summarizes and judges whether to advance to Paper Writing phase"
                },
                "key_tasks": [
                    "Evaluate model solutions from dimensions of solvability, robustness, generalizability, interpretability, etc.",
                    "Output implementation roadmap: data preparation, parameter calibration, solver configuration, etc.",
                    "Provide milestones and person-hour estimates",
                    "Provide evaluation metrics and acceptance criteria"
                ],
                "completion_criteria": [
                    "Model Architect has strictly followed Domain Expert's modeling requirements to complete mathematical modeling",
                    "Critic has evaluated and questioned the models",
                    "Models are complete and executable, capable of supporting Paper Writing phase's description of models and solutions",
                    "Project Manager has confirmed models meet conditions for entering Paper Writing phase"
                ],
                "speech_order": "Model Architect performs mathematical modeling -> Critic evaluates and questions -> Project Manager (summarizes phase and judges whether to advance)"
            },
            "Paper Writing": {
                "description": "Write complete academic paper based on discussion content, output in LaTeX format, goal to generate complete 25-page paper. Paper structure must follow MCM/ICM standard template. **Important: Problem modeling sections (Problem 1, Problem 2, Problem 3, Problem 4) must occupy the largest portion of the paper, with Problem 1 having the most content, Problem 2 second, Problem 3 and Problem 4 detailed but slightly less**",
                "participants": ["Project Manager", "Paper Writer", "Critic"],
                "min_turns": 12,
                "turn_tasks": {
                    1: "Paper Writer writes cover page (Abstract + Keywords, abstract must clearly list models used for each problem: Problem 1 uses what model, Problem 2 uses what model, Problem 3 uses what model); Critic evaluates abstract quality (including whether models used for each problem are clearly listed) and keyword relevance; Project Manager coordinates discussion",
                    2: "Paper Writer writes introduction (background + methodology overview + flowchart); Critic evaluates introduction logic and flowchart clarity; Project Manager coordinates discussion",
                    3: "Paper Writer writes modeling preparation (assumptions + notation + data preprocessing); Critic evaluates assumption reasonableness and notation standardization; Project Manager coordinates discussion",
                    4: "Paper Writer writes Problem 1 modeling (method + model + results + figures/tables, most detailed content, suggested 6-8 pages, including most detailed model establishment, solution process, result analysis and rich figures/tables); Critic evaluates model accuracy and result presentation; Project Manager coordinates discussion",
                    5: "Paper Writer writes Problem 2 modeling (method + model + results + figures/tables, moderate content, suggested 4-5 pages, including complete model establishment, solution process and result analysis); Critic evaluates model accuracy and result presentation; Project Manager coordinates discussion",
                    6: "Paper Writer writes Problem 3 modeling/New Insights (originality analysis + recommendations, least content, suggested 2-3 pages, including model establishment, result presentation and originality analysis); Critic evaluates originality and recommendation feasibility; Project Manager coordinates discussion",
                    7: "Paper Writer writes Problem 4 modeling (method + model + results + figures/tables, detailed content, suggested 4-6 pages, including complete model establishment, solution process and result analysis); Critic evaluates model accuracy and result presentation; Project Manager coordinates discussion",
                    8: "Paper Writer writes sensitivity/robustness analysis; Critic evaluates analysis depth and comprehensiveness; Project Manager coordinates discussion",
                    9: "Paper Writer writes model evaluation (advantages/disadvantages); Critic evaluates evaluation objectivity and comprehensiveness; Project Manager coordinates discussion",
                    10: "Paper Writer writes memo (letter to decision makers); Critic evaluates memo clarity and executability; Project Manager coordinates discussion",
                    11: "Paper Writer compiles references; Critic evaluates citation format and completeness; Project Manager coordinates discussion",
                    12: "Paper Writer writes AI usage report (if applicable); Critic final evaluation; Project Manager summarizes and ends meeting"
                },
                "key_tasks": [
                    "Organize content according to MCM/ICM standard paper template structure (12 sections)",
                    "Output paper content in LaTeX format, all content in English",
                    "**Problem modeling sections (Problem 1, Problem 2, Problem 3, Problem 4) must occupy the largest portion of the paper, this is the core content**",
                    "**Problem 1 modeling has the most detailed content (suggested 6-8 pages), Problem 2 modeling has second most (suggested 4-5 pages), Problem 3 and Problem 4 modeling detailed but slightly less (suggested 2-3 and 4-6 pages)**",
                    "Ensure paper content is complete, logically clear, and meets academic standards",
                    "Gradually refine paper through multiple iterations, ultimately generating complete 25-page paper",
                    "Critic evaluates paper quality, provides feedback and improvement suggestions"
                ],
                "completion_criteria": [
                    "Paper Writer has completed paper writing based on all discussion content",
                    "Paper is in LaTeX format, structurally complete, conforming to MCM/ICM standard template (11 sections)",
                    "All paper content written in English",
                    "**Problem modeling sections (Problem 1, Problem 2, Problem 3, Problem 4) occupy the largest portion of the paper, Problem 1 has the most content, Problem 2 second, Problem 3 and Problem 4 detailed but slightly less**",
                    "Paper content is complete, logically clear, meeting complete 25-page paper requirements",
                    "Critic has evaluated and provided feedback on the paper, paper quality meets requirements",
                    "This is the last phase, meeting ends upon completion"
                ],
                "speech_order": "Paper Writer writes/refines paper -> Critic evaluates and provides feedback -> Project Manager (summarizes phase and ends meeting)"
            }
        }
        
        workflow_info = f"""
**Meeting Workflow Description**

This meeting proceeds through the following 4 phases in order. Each phase must meet completion criteria before advancing to the next phase:

"""
        
        current_phase_index = self.phase_order.index(current_phase) if current_phase in self.phase_order else -1
        
        for idx, phase in enumerate(self.phase_order):
            phase_info = phase_descriptions.get(phase, {})
            phase_desc = phase_info.get("description", "")
            participants = phase_info.get("participants", [])
            key_tasks = phase_info.get("key_tasks", [])
            completion_criteria = phase_info.get("completion_criteria", [])
            speech_order = phase_info.get("speech_order", "")
            min_turns = phase_info.get("min_turns", 1)
            
            # Mark current phase
            if phase == current_phase:
                phase_marker = "【Current Phase】"
                phase_emphasis = "**"
            else:
                phase_marker = ""
                phase_emphasis = ""
            
            workflow_info += f"{idx + 1}. {phase_emphasis}{phase}{phase_marker}{phase_emphasis}\n"
            workflow_info += f"   - Phase Objective: {phase_desc}\n"
            workflow_info += f"   - Participating Roles: {', '.join(participants)}\n"
            workflow_info += f"   - Minimum Rounds: {min_turns} rounds\n"
            if speech_order:
                workflow_info += f"   - Speaking Order: {speech_order}\n"
            if key_tasks:
                workflow_info += f"   - Key Tasks:\n"
                for task in key_tasks:
                    workflow_info += f"     - {task}\n"
            if completion_criteria:
                workflow_info += f"   - Completion Criteria (all must be met to advance):\n"
                for criterion in completion_criteria:
                    workflow_info += f"     ✓ {criterion}\n"
            workflow_info += "\n"
        
        # Add phase advancement mechanism description
        workflow_info += """
**Phase Advancement Mechanism**

1. **Advancement Decision Authority**: Project Manager is responsible for judging whether current phase is complete and deciding whether to advance to next phase
2. **Advancement Criteria**:
   - All completion criteria for current phase have been met
   - Main participating roles have completed their core work
   - Discussion has reached necessary consensus
   - Required inputs and preparations for next phase are ready
3. **Advancement Methods**:
   - When Project Manager judges current phase is complete in phase summary, can explicitly state "advance to next phase", "proceed to [phase name]", etc.
   - System will automatically detect advancement signal and switch to next phase
   - If current phase is not complete, Project Manager should continue guiding discussion until completion criteria are met
4. **Forced Advancement**: After team reaches consensus, Project Manager can exercise forced advancement authority to lock action plan and advance phase

"""
        
        # Add detailed description for current phase
        if current_phase in phase_descriptions:
            phase_info = phase_descriptions[current_phase]
            participants = phase_info.get("participants", [])
            completion_criteria = phase_info.get("completion_criteria", [])
            speech_order = phase_info.get("speech_order", "")
            min_turns = phase_info.get("min_turns", 1)
            turn_tasks = phase_info.get("turn_tasks", {})
            
            workflow_info += f"**Current Phase ({current_phase}) Detailed Description**\n\n"
            
            workflow_info += f"**Phase Round Arrangement**: This phase requires at least {min_turns} rounds of discussion.\n\n"
            
            if turn_tasks:
                workflow_info += f'**Round-by-Round Description** (Note: "Rounds", not "Days"):\n'
                for turn_num in sorted(turn_tasks.keys()):
                    workflow_info += f"  - **Round {turn_num}**: {turn_tasks[turn_num]}\n"
                workflow_info += "\n"
            
            if speech_order:
                workflow_info += f"**Speaking Order and Timing**: {speech_order}\n\n"
            
            if completion_criteria:
                workflow_info += f"**Phase Completion Criteria** (all must be met to advance to next phase):\n"
                for criterion in completion_criteria:
                    workflow_info += f"  ✓ {criterion}\n"
                workflow_info += "\n"
            
            # Determine participation based on role
            if agent_name:
                if agent_name in participants:
                    workflow_info += f"**Your Participation Description**:\n"
                    workflow_info += f"- You ({agent_name}) are one of the main participating roles for the current phase\n"
                    workflow_info += f"- Please actively participate in discussion according to your responsibilities and current phase objectives\n"
                    moderator_name = self._get_actual_role_name("Project Manager")
                    if agent_name == moderator_name:
                        workflow_info += f"- **Important**: You are responsible for judging whether current phase is complete and deciding whether to advance to next phase upon meeting completion criteria\n"
                        workflow_info += f'- When advancing, explicitly state "advance to next phase", "proceed to [phase name]", etc., system will automatically detect and switch phases\n'
                else:
                    workflow_info += f"**Your Participation Description**:\n"
                    workflow_info += f"- You ({agent_name}) are not a main participating role for the current phase\n"
                    workflow_info += f"- Main participating roles for this phase: {', '.join(participants)}\n"
                    workflow_info += f"- If invited to participate, please provide relevant suggestions; otherwise, please wait for your main participation phase\n"

        # Substitute canonical role names with actual template-specific names
        canonical_map = {
            "Project Manager": self._get_actual_role_name("Project Manager"),
            "Domain Expert":   self._get_actual_role_name("Domain Expert"),
            "Model Architect": self._get_actual_role_name("Model Architect"),
            "Paper Writer":    self._get_actual_role_name("Paper Writer"),
            "Critic":          self._get_actual_role_name("Critic"),
        }
        for canonical, actual in canonical_map.items():
            if actual != canonical:
                workflow_info = workflow_info.replace(canonical, actual)

        return workflow_info

    def _build_phase_prompt(self, role: RoleType, phase: str, agent_name: str = None) -> str:
        # Get team info and responsibility boundaries
        team_boundaries = self._get_team_info_and_boundaries(agent_name) if agent_name else ""
        # Get meeting workflow description
        workflow_info = self._get_meeting_workflow_info(phase, agent_name)
        problem_focus = self._get_problem_focus_prompt(agent_name)
        if problem_focus:
            workflow_info = f"{workflow_info}\n{problem_focus}\n"
        
        # Role-based dispatch using actual template-specific names
        analysis_name = self._get_actual_role_name("Domain Expert")
        build_name = self._get_actual_role_name("Model Architect")
        critic_name = self._get_actual_role_name("Critic")
        write_name = self._get_actual_role_name("Paper Writer")

        # Build expert special handling (mathematical modeling)
        if agent_name == build_name:
            # Get analysis expert's latest speech (containing modeling requirements)
            expert_messages = [msg.content for msg in self.conversation if msg.role == analysis_name]
            expert_requirements = "\n".join(expert_messages[-3:]) if expert_messages else f"No clear modeling requirements from {analysis_name} yet"

            # Get pre-meeting reference file content
            context_files_content = ""
            if self.context_file_manager.context_files_info:
                context_files_content = self.context_file_manager.format_context_files_content()
            
            # Use PromptBuilder to construct Model Architect prompt
            return PromptBuilder.build_architect_prompt(
                workflow_info=workflow_info,
                phase=phase,
                expert_requirements=expert_requirements,
                context_files_content=context_files_content,
                team_boundaries=team_boundaries
            )
        
        # Critic/Analyst special handling (evaluation and questioning, modeling prohibited)
        if agent_name == critic_name:
            # Get build expert's latest speech (containing model design)
            architect_messages = [msg.content for msg in self.conversation if msg.role == build_name]
            architect_design = "\n".join(architect_messages[-3:]) if architect_messages else f"No model design from {build_name} yet"

            paper_writer_messages = [msg.content for msg in self.conversation if msg.role == write_name]
            paper_content = "\n".join(paper_writer_messages[-3:]) if paper_writer_messages else f"No paper content from {write_name} yet"

            # Use PromptBuilder to construct Critic prompt
            return PromptBuilder.build_critic_prompt(
                workflow_info=workflow_info,
                phase=phase,
                architect_design=architect_design,
                paper_content=paper_content,
                team_boundaries=team_boundaries
            )
        
        # Write expert special handling
        if agent_name == write_name:
            # Get complete discussion content (get more history to ensure completeness)
            discussion_summary = "\n".join([msg.content for msg in self.conversation if msg.role != write_name])
            solution_summary = self.solution_tracker.get_solutions_summary()
            
            # Use PromptBuilder to construct Paper Writer prompt
            return PromptBuilder.build_paper_writer_prompt(
                workflow_info=workflow_info,
                phase=phase,
                topic=self.topic,
                discussion_summary=discussion_summary,
                solution_summary=solution_summary,
                team_boundaries=team_boundaries
            )
        
        phase_prompts = {
            "Problem Analysis": {
                RoleType.EXPERT: f"""{workflow_info}

Current Phase: Problem Analysis. Please deeply analyze the problem: {self.topic}, identifying core issues and key challenges. Provide specific problem decomposition and analysis.""",
                RoleType.ANALYST: f"""{workflow_info}

Current Phase: Problem Analysis. Please raise key questions about the technical expert's analysis, helping clarify the nature and boundaries of the problem.""",
                RoleType.MODERATOR: f"""{workflow_info}

Current Phase: Problem Analysis. Please act as moderator, summarizing current insights in concise bullet points, and raising key questions that other roles need to answer, keeping discussion focused. Prohibit directly outputting complete solutions or papers.""",
            },
            "Model Design": {
                RoleType.EXPERT: f"""{workflow_info}

Current Phase: Model Design. Please propose specific model design solutions based on problem analysis. Solutions should be clear, specific, and actionable.""",
                RoleType.ANALYST: f"""{workflow_info}

Current Phase: Model Design. Please evaluate the feasibility and effectiveness of model design solutions, proposing improvement suggestions.""",
                RoleType.MODERATOR: f"""{workflow_info}

Current Phase: Model Design. Please moderate discussion, first summarizing existing design approaches, then guiding the formation of executable solutions. After each role speaks, you may choose to reply, comment, or summarize. Prohibit directly writing complete deliverables.""",
            },
            "Model Building": {
                RoleType.EXPERT: f"""{workflow_info}

Current Phase: Model Building. Please provide specific model building steps and technical suggestions.""",
                RoleType.ANALYST: f"""{workflow_info}

Current Phase: Model Building. Please analyze risks and considerations in the model building process.""",
                RoleType.MODERATOR: f"""{workflow_info}

Current Phase: Model Building. Please moderate discussion, first summarizing existing building approaches, then guiding the formation of executable solutions. After each role speaks, you may choose to reply, comment, or summarize. Prohibit directly writing complete deliverables.

**Important**: In the Model Building phase, ensure Model Architect outputs complete implementation roadmap including derivations, variable definitions, solution processes, and experimental plans needed for Paper Writing. Guide Model Architect to supplement all technical details needed for the paper.""",
            },
            "Paper Writing": {
                RoleType.EXPERT: f"""{workflow_info}

Current Phase: Paper Writing. Please write the paper based on discussion content. **Important: Paper must strictly follow the 12 sections of MCM/ICM standard template, all content must be written in English (All content must be written in English)**.""",
                RoleType.ANALYST: f"""{workflow_info}

Current Phase: Paper Writing. Please analyze risks and considerations in the paper writing process. **Important: Ensure paper conforms to the 12 sections of MCM/ICM standard template, all content in English**.""",
                RoleType.MODERATOR: f"""{workflow_info}

Current Phase: Paper Writing. Please guide Paper Writer to complete the paper writing, and summarize and decide after Paper Writer completes. **Important: Ensure paper strictly follows the 12 sections of MCM/ICM standard template, all content in English**.""",
            },
            "Model Design Guidance": {
                RoleType.MODERATOR: f"""{workflow_info}

Current Phase: Model Design. Please moderate discussion, first summarizing existing design approaches, then guiding the formation of executable solutions. After each role speaks, you may choose to reply, comment, or summarize. Prohibit directly writing complete deliverables."""
            },
            "Model Building Guidance": {
                RoleType.MODERATOR: f"""{workflow_info}

Current Phase: Model Building. Please moderate discussion, first summarizing existing building approaches, then guiding the formation of executable solutions. After each role speaks, you may choose to reply, comment, or summarize. Prohibit directly writing complete deliverables."""
            },
            "Model Design Reply": {
                RoleType.MODERATOR: f"""{workflow_info}

Current Phase: Model Design. A role just spoke, please reply, comment on, or summarize the speech, then guide the next step of discussion."""
            },
            "Model Building Reply": {
                RoleType.MODERATOR: f"""{workflow_info}

Current Phase: Model Building. A role just spoke, please reply, comment on, or summarize the speech, then guide the next step of discussion."""
            },
            "Meeting Summary": {
                RoleType.MODERATOR: "Currently in meeting wrap-up phase. Please concisely summarize action plan execution status, remind of incomplete deliverables, and confirm whether to continue in this phase."
            }
        }
        phase_prompt_dict = phase_prompts.get(phase, {})
        base_prompt = phase_prompt_dict.get(role, f"""{workflow_info}

Current Phase: {phase}. Please actively participate in discussion.""")

        moderator_name = self._get_actual_role_name("Project Manager")
        if agent_name == moderator_name:
            if (
                phase == "Problem Analysis"
                and self.context_file_manager.context_files_overview
                and not self.context_file_manager._context_files_announced
            ):
                files_instruction = (
                    "【File Interpretation】The following reference files have been loaded before the meeting started. In this phase's speech, "
                    "please summarize the core content of each file, its relationship to the current topic, and key inputs to pass to subsequent roles. "
                    "Make sure all members understand these reference materials:\n"
                    f"{self.context_file_manager.context_files_overview}"
                )
                base_prompt = f"{files_instruction}\n\n{base_prompt}"
            
            # Adjust moderation guidelines based on phase
            if self.meeting_phase == "Problem Analysis":
                guidance_rules = (
                "【Moderation Guidelines】\n"
                    "1. Summarize current phase progress in bullet points;\n"
                    "2. Raise 1-2 focused questions for in-depth problem analysis;\n"
                    "3. Prohibit replacing others to output complete solutions, code or papers, only assume coordination and decision-making responsibilities.\n"
                )
            elif self.meeting_phase in ["Model Design", "Model Building"]:
                guidance_rules = (
                    "【Moderation Guidelines】\n"
                    "1. Summarize current phase progress in bullet points, and guide discussion;\n"
                    "2. Raise 1-2 focused questions, guide other roles to supplement key information;\n"
                    "3. Prohibit replacing others to output complete solutions, code or papers, only assume coordination and decision-making responsibilities.\n"
                )
            else:  # Paper Writing
                guidance_rules = (
                    "【Moderation Guidelines】\n"
                    "1. Summarize current phase progress in bullet points, and guide work;\n"
                    "2. Raise 1-2 focused questions, guide other roles to complete work;\n"
                    "3. Prohibit replacing others to output complete solutions, code or papers, only assume coordination and decision-making responsibilities.\n"
                )
            if len(self.problem_segments) > 1:
                guidance_rules = (
                    f"{guidance_rules.rstrip()}\n"
                    "4. **Track each item**: In summary, must separately state each sub-problem's progress, risks, and next steps, prohibit merged descriptions.\n"
                )
            # Project Manager's phase advancement confirmation authority has been cancelled, phase advancement is now automatically determined by system
            base_prompt = f"{base_prompt}\n\n{guidance_rules}"
        
        # Add team info and responsibility boundaries for all roles (if agent_name is provided)
        if agent_name and team_boundaries:
            return f"{base_prompt}\n\n{team_boundaries}"
        return base_prompt

    def _recent_min_length_ok(self, n_messages: int = 3, min_chars: int = 120) -> bool:
        """
        Determine whether recent n messages collectively meet minimum information requirements
        """
        recent = self.conversation[-n_messages:] if n_messages > 0 else []
        if not recent:
            return False
        total = sum(len(msg.content) for msg in recent)
        return total >= min_chars * max(1, len(recent) // 2)

    def _recent_diversity_ok(self, n_messages: int = 3, max_similarity: float = 0.9) -> bool:
        """
        Determine whether recent n messages have sufficient diversity among them (pairwise similarity should not be too high)
        """
        recent = self.conversation[-n_messages:] if n_messages > 1 else []
        if len(recent) <= 1:
            return False
        contents = [m.content for m in recent]
        for i in range(len(contents)):
            for j in range(i + 1, len(contents)):
                a, b = contents[i], contents[j]
                if max(len(a), len(b)) < 100:
                    # Short text tolerance, do not force similarity judgment
                    continue
                if difflib.SequenceMatcher(None, a, b).ratio() >= max_similarity:
                    return False
        return True

    def _extract_link_from_content(self, content: str) -> str:
        """Extract links from content (prioritize image links)"""
        if not content:
            return None
        
        # Regex pattern for matching URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, content)
        
        if not urls:
            return None
        
        # Prioritize returning image links
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.bmp']
        for url in urls:
            url_lower = url.lower()
            if any(ext in url_lower for ext in image_extensions):
                return url
        
        # If no image links, return first link
        return urls[0] if urls else None

    def _add_message(
        self,
        role: str,
        content: str,
        *,
        message_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        branch_id: Optional[str] = None,
        model_id: Optional[str] = None,
        provider: Optional[str] = None,
        prompt_hash: Optional[str] = None,
        token_usage: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Append a message to conversation, and trigger message_callback to push to frontend.

        v2 DAG field description:
        - message_id : Unique ID for this message; None for auto-generation
        - parent_id  : Parent message ID (default = ID of last message in conversation)
        - branch_id  : Branch ID; used when multiple models run in parallel from same parent; None means main line
        - model_id / provider : Actual model identifier called
        - prompt_hash / token_usage : Used for reproducibility and cost tracking
        """
        mid = message_id or uuid.uuid4().hex
        pid = parent_id

        # Default parent: most recent message (linear tracking, effective when branch_id is empty)
        if pid is None and self.conversation:
            last = self.conversation[-1]
            # If last belongs to same branch, parent follows last; otherwise parent follows that branch's most recent root
            if not last.branch_id:
                pid = last.message_id
            else:
                # Find most recent node with same branch_id as parent
                for prev in reversed(self.conversation):
                    if prev.branch_id == last.branch_id:
                        pid = prev.message_id
                        break

        message = Message(
            role=role,
            content=content,
            timestamp=time.time(),
            turn=self.current_turn,
            message_id=mid,
            parent_id=pid,
            branch_id=branch_id,
            model_id=model_id,
            provider=provider,
            prompt_hash=prompt_hash,
            token_usage=token_usage,
            metadata=dict(metadata) if metadata else {},
        )
        self.conversation.append(message)

        # Extract links from content
        link = self._extract_link_from_content(content)

        if self.message_callback:
            message_data = {
                'type': 'message',
                'role': role,
                'content': content,
                'turn': self.current_turn,
                'timestamp': message.timestamp,
                # ---- v2 DAG fields ----
                'message_id': mid,
                'parent_id': pid,
                'branch_id': branch_id,
                'model_id': model_id,
                'provider': provider,
            }
            if link:
                message_data['link'] = link
            self.message_callback(message_data)

        return message


    async def run_branch_nodes(
        self,
        branches: List[Dict[str, Any]],
        *,
        parent_id: Optional[str] = None,
        branch_reason: str = "user_compare",
    ) -> List[NodeExecutionResult]:
        """Run multiple branches in parallel (each branch can be different model answering same question).

        Parameters
        ----------
        branches : list of dicts, each dict must have:
            - agent      : AIAgent instance (already bound with model_config)
            - model_override : Overridden model name (optional)
            - prompt_kwargs : Parameters to pass to _build_full_prompt (optional)
        parent_id : DAG parent node ID (default takes last message in conversation)
        branch_reason : Branch reason label (written to message.metadata.branch_reason)

        Returns
        -------
        list of NodeExecutionResult, returned in branches order

        Behavior:
        - All branches execute concurrently (asyncio.gather)
        - Each branch gets unique branch_id (parallel subtree under same parent_id)
        - After completion, push each branch result via message_callback
        - manifest records prompt templates for all branches
        """
        import asyncio
        import uuid

        if not branches:
            return []

        # Determine parent_id
        pid = parent_id
        if pid is None and self.conversation:
            pid = self.conversation[-1].message_id

        # All branches share same branch_id (parallel subtree under same parent node)
        shared_branch_id = uuid.uuid4().hex

        # Prepare context for each branch (using current conversation)
        results: List[NodeExecutionResult] = []

        async def run_one_branch(idx: int, spec: Dict[str, Any]) -> NodeExecutionResult:
            agent = spec["agent"]
            # Optional: override model
            effective_config = dict(agent.model_config)
            if "model_override" in spec and spec["model_override"]:
                effective_config["model"] = spec["model_override"]
            if "temperature" in spec:
                effective_config["temperature"] = spec["temperature"]

            # Build context (same logic as _agent_speak_with_focus)
            context = self._build_conversation_context(agent.name)
            full_prompt = agent.build_full_prompt(
                context=context,
                topic=self.topic,
                per_turn_guidance=spec.get("per_turn_guidance", ""),
            )

            mid = uuid.uuid4().hex
            t0 = time.time()
            try:
                # Temporarily create runner with model override config
                original_config = agent.model_config
                agent.model_config = effective_config
                response = await agent.generate_response(
                    [Message("system", full_prompt, time.time(), 0)] + context,
                    self.topic,
                )
                agent.model_config = original_config
                error_text: Optional[str] = None
            except Exception as e:
                logger.error(f"Branch {idx} failed: {e}", exc_info=True)
                response = f"⚠️ **Branch {idx} generation error**: {e}"
                error_text = str(e)
                try:
                    agent.model_config = original_config
                except Exception:
                    pass
            duration_ms = int((time.time() - t0) * 1000)

            in_tokens = max(1, len(full_prompt) // 4)
            out_tokens = max(1, len(response) // 4)

            from utils.manifest import fingerprint_prompt
            prompt_hash = fingerprint_prompt(full_prompt)

            message = Message(
                role=agent.name,
                content=response,
                timestamp=time.time(),
                turn=self.current_turn,
                message_id=mid,
                parent_id=pid,
                branch_id=shared_branch_id,
                model_id=effective_config.get("model"),
                provider=effective_config.get("provider"),
                prompt_hash=prompt_hash,
                token_usage={"input": in_tokens, "output": out_tokens},
                metadata={
                    "duration_ms": duration_ms,
                    "branch_reason": branch_reason,
                    "branch_index": idx,
                    "error": error_text,
                },
            )
            self.conversation.append(message)

            # Push to frontend
            if self.message_callback:
                self.message_callback({
                    "type": "branch_result",
                    "role": agent.name,
                    "content": response,
                    "turn": self.current_turn,
                    "timestamp": message.timestamp,
                    "message_id": mid,
                    "parent_id": pid,
                    "branch_id": shared_branch_id,
                    "model_id": effective_config.get("model"),
                    "provider": effective_config.get("provider"),
                    "branch_index": idx,
                    "total_branches": len(branches),
                    "finished": error_text is None,
                    "error": error_text,
                })

            # manifest record
            if self.manifest is not None:
                key = f"branch:{shared_branch_id}:{mid}"
                self.manifest.add_prompt_template(
                    key=key,
                    body=full_prompt,
                    path=f"branch_prompts/{shared_branch_id[:8]}.txt",
                )

            return NodeExecutionResult(
                message=message,
                prompt=full_prompt,
                duration_ms=duration_ms,
                finished=error_text is None,
                error=error_text,
            )

        # Execute all branches concurrently
        tasks = [run_one_branch(i, spec) for i, spec in enumerate(branches)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions (asyncio.gather wraps exceptions as Exception objects)
        final_results: List[NodeExecutionResult] = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(f"Branch {i} raised exception: {r}")
                err_msg = Message(
                    role=branches[i]["agent"].name,
                    content=f"⚠️ **Branch {i} exception**: {r}",
                    timestamp=time.time(),
                    turn=self.current_turn,
                    message_id=uuid.uuid4().hex,
                    parent_id=pid,
                    branch_id=shared_branch_id,
                    metadata={"branch_reason": branch_reason, "branch_index": i, "error": str(r)},
                )
                self.conversation.append(err_msg)
                final_results.append(NodeExecutionResult(
                    message=err_msg, prompt="", duration_ms=0, finished=False, error=str(r)
                ))
            else:
                final_results.append(r)

        # Notify frontend all branches complete
        if self.message_callback:
            self.message_callback({
                "type": "branches_complete",
                "branch_id": shared_branch_id,
                "count": len(final_results),
            })

        return final_results

    def rollback_to(self, message_id: str) -> int:
        """Rollback conversation to specified node (excluding that node itself), return number of deleted nodes.

        Note: This is only data-level rollback, will not trigger regeneration.
        Frontend needs to receive 'rollback' event via message_callback and delete DOM nodes itself.
        """
        try:
            idx = next(i for i, m in enumerate(self.conversation) if m.message_id == message_id)
        except StopIteration:
            logger.warning(f"rollback_to: message_id={message_id} not found")
            return 0

        removed = self.conversation[idx + 1:]
        self.conversation = self.conversation[:idx + 1]
        logger.info(f"Rolled back {len(removed)} messages, now at message_id={message_id}")
        if self.message_callback:
            self.message_callback({
                "type": "rollback",
                "message_id": message_id,
                "removed_count": len(removed),
                "removed_ids": [m.message_id for m in removed],
            })
        return len(removed)

    def get_subtree_messages(self, message_id: str) -> List[Message]:
        """Get subtree rooted at specified node (including that node)"""
        try:
            root_idx = next(i for i, m in enumerate(self.conversation) if m.message_id == message_id)
        except StopIteration:
            return []

        subtree = [self.conversation[root_idx]]
        # BFS to find all descendants (through parent_id chain)
        queue = [message_id]
        seen = {message_id}
        while queue:
            pid = queue.pop(0)
            for m in self.conversation:
                if m.parent_id == pid and m.message_id not in seen:
                    subtree.append(m)
                    seen.add(m.message_id)
                    queue.append(m.message_id)
        return subtree

    async def _generate_final_report(self) -> str:
        # Get pre-meeting reference file content
        context_files_content = ""
        if self.context_file_manager.context_files_info:
            context_files_content = self.context_file_manager.format_context_files_content()
        
        # Use ReportGenerator to generate report
        report_generator = ReportGenerator(
            self.topic,
            self.conversation,
            self.solution_tracker,
            context_files_content=context_files_content
        )
        return await report_generator.generate_final_report()

    def _prepare_meeting_result(self, final_report: str) -> Dict[str, Any]:
        conversation_data = []
        for msg in self.conversation:
            conversation_data.append({
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "turn": msg.turn
            })
        return {
            "topic": self.topic,
            "conversation": conversation_data,
            "final_report": final_report,
            "turns": self.current_turn
        }

    def set_context_files(self, file_specs: List[Any]):
        """Allow external setting of pre-meeting reference file list"""
        self.context_file_manager.set_context_files(file_specs)
    
    def set_phase_turns(self, phase_turns: Dict[str, int]):
        """Allow external setting of round configuration for each phase"""
        self.phase_turns.update(phase_turns)
        logger.info(f"Phase round configuration updated: {self.phase_turns}")

    def _save_conversation_to_file(self):
        try:
            from config import Config
            from utils.logger import logger as _logger
            logs_dir = Config.LOGS_DIR
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir, exist_ok=True)
            filename = os.path.join(logs_dir, f"meeting_{int(time.time())}.json")
            meeting_data = self._prepare_meeting_result("")
            meeting_data["final_report"] = "Meeting report has been generated, please see output above"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(meeting_data, f, ensure_ascii=False, indent=2)
            _logger.info(f"Meeting record saved to file: {filename}")
            print(f"\nMeeting record saved to file: {filename}")
        except Exception as e:
            logger.error(f"Failed to save meeting record: {e}", exc_info=True)
            print(f"\nFailed to save meeting record: {e}")
