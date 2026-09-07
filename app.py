#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Web application main entry.
Uses Flask + Flask-SocketIO for real-time communication.
"""
import asyncio
import threading
import os
import uuid
from pathlib import Path
from typing import Dict, List, Any

from flask import Flask, render_template, request, jsonify, session, send_from_directory
from flask_socketio import SocketIO, emit, join_room

from meeting_manager import MeetingManagerV2
from config import Config
from utils.logger import logger
from utils.validators import validate_meeting_request
from utils.meeting_state import MeetingStateManager, MeetingStatus
from utils.model_manager import get_model_manager
from utils.conversation_storage import save_conversation, list_conversations, load_conversation, delete_conversation
from utils.latex_compiler import (
    compile_latex,
    list_tex_files,
    read_tex_file,
    save_tex_file,
    COMPILED_DIR,
    find_miktex_bin,
)
from utils.code_executor import (
    execute_python as python_execute_code,
    get_python_info as python_get_info,
    install_package as python_install_package,
    list_run_files as python_list_files,
    WORKSPACE_DIR as PYTHON_WORKSPACE_DIR,
)
from utils.vision_handler import recognize_files

# Validate configuration
try:
    config_validation = Config.validate()
    if not config_validation['valid']:
        error_msg = f'Configuration validation failed: {config_validation["errors"]}'
        logger.error(error_msg)
        print("\n" + "=" * 50)
        print("Configuration validation failed!")
        print("=" * 50)
        for error in config_validation['errors']:
            print(f"Error: {error}")
        print("\nPlease check the .env file or environment variable settings")
        print("=" * 50 + "\n")
        raise ValueError(error_msg)

    if config_validation['warnings']:
        for warning in config_validation['warnings']:
            logger.warning(f'Configuration warning: {warning}')
except Exception as e:
    print(f"\nStartup failed: {e}\n")
    raise

app = Flask(__name__)
app.config['SECRET_KEY'] = Config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_FILE_SIZE * Config.MAX_FILES
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

state_manager = MeetingStateManager(max_concurrent=Config.MAX_CONCURRENT_MEETINGS)
active_meetings = {}

# Per-session user message queues (for human-AI interaction)
session_user_messages: dict = {}
session_stop_flags: dict = {}

# Per-session chat history (for Chat Mode)
chat_sessions: dict = {}


def create_message_callback(session_id):
    def callback(data):
        try:
            socketio.emit('meeting_update', data, to=session_id, namespace='/')
        except Exception as e:
            logger.error(f'Failed to send message (session: {session_id}): {e}')
    return callback


@socketio.on('connect')
def handle_connect():
    session_id = request.sid
    join_room(session_id)
    logger.info(f'Client connected: {session_id}')
    emit('connected', {'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    session_id = request.sid
    logger.info(f'Client disconnected: {session_id}')

    state = state_manager.get_state(session_id)
    if state:
        if state.status == MeetingStatus.RUNNING:
            state.request_stop()
        state_manager.remove_state(session_id)

    if session_id in active_meetings:
        del active_meetings[session_id]
    session_user_messages.pop(session_id, None)
    session_stop_flags.pop(session_id, None)
    chat_sessions.pop(session_id, None)


@socketio.on('start_meeting')
def handle_start_meeting(data):
    session_id = request.sid

    try:
        valid, error, validated_data = validate_meeting_request(data)
        if not valid:
            logger.warning(f'Meeting request validation failed (session: {session_id}): {error}')
            emit('error', {'message': error})
            return

        topic = validated_data['topic']
        max_turns = validated_data.get('max_turns', 5)

        phase_turns = data.get('phase_turns', {})
        default_phase_turns = {
            'Problem Analysis': 20,
            'Model Design': 20,
            'Model Building': 20,
            'Paper Writing': 30
        }
        phase_turns = {**default_phase_turns, **phase_turns}

        task_template_data = data.get('task_template', {})

        existing_state = state_manager.get_state(session_id)
        if existing_state and existing_state.status in [MeetingStatus.RUNNING, MeetingStatus.STARTING]:
            logger.warning(f'Meeting already in progress (session: {session_id})')
            emit('error', {'message': 'You already have a meeting in progress, please wait for it to complete'})
            return

        try:
            state = state_manager.create_state(session_id, topic, max_turns)
        except RuntimeError as e:
            logger.error(f'Failed to create meeting state (session: {session_id}): {e}')
            emit('error', {'message': 'Server is busy, please try again later'})
            return

        message_callback = create_message_callback(session_id)

        # User-selected models (optional). Falls back to default roles.
        selected_models = data.get('selected_models') or None

        # Team template (optional). When present, replaces the default 5 agents
        # with the slots defined by the template. Falls back to selected_models
        # path otherwise.
        template_id = (data.get('template_id') or '').strip()
        slot_model_overrides = data.get('slot_model_overrides') or {}

        meeting = MeetingManagerV2(
            topic, max_turns, message_callback,
            session_id=session_id,
            socketio=socketio,
            selected_models=selected_models,
        )

        if template_id:
            try:
                tpl = meeting.load_team_template(template_id)
                # If frontend sent the legacy selected_models dict, fold it into overrides
                if isinstance(selected_models, list):
                    for item in selected_models:
                        if isinstance(item, dict) and item.get('role') and item.get('model_name'):
                            slot_model_overrides.setdefault(item['role'], item['model_name'])
                meeting.apply_team_template(tpl, slot_model_overrides)
            except Exception as e:
                logger.error(f'apply_team_template failed for {template_id}: {e}', exc_info=True)
                emit('error', {'message': f'Failed to apply team template: {e}'})
                state_manager.remove_state(session_id)
                return
        else:
            meeting.setup_default_agents()

        if hasattr(meeting, 'set_phase_turns'):
            meeting.set_phase_turns(phase_turns)

        if task_template_data and hasattr(meeting, 'set_task_template'):
            meeting.set_task_template(task_template_data)

        context_files = validated_data.get('context_files') or []
        if context_files and hasattr(meeting, 'set_context_files'):
            meeting.set_context_files(context_files)

        active_meetings[session_id] = meeting
        session_user_messages[session_id] = []
        session_stop_flags[session_id] = False

        state.start()

        run_meeting_async(meeting, session_id, state)

        emit('meeting_started', {
            'topic': topic,
            'max_turns': max_turns,
            'version': 'v2',
            'message': 'Meeting started',
            'team_template_id': template_id or None,
        })

        logger.info(f'Meeting started (session: {session_id}, topic: {topic}, version: v2)')

    except Exception as e:
        logger.error(f'Failed to start meeting (session: {session_id}): {e}', exc_info=True)
        emit('error', {'message': f'Failed to start meeting: {str(e)}'})
        state_manager.remove_state(session_id)
        if session_id in active_meetings:
            del active_meetings[session_id]


@socketio.on('stop_meeting')
def handle_stop_meeting():
    session_id = request.sid

    try:
        state = state_manager.get_state(session_id)
        if not state:
            emit('error', {'message': 'No meeting in progress found'})
            return

        if state.request_stop():
            session_stop_flags[session_id] = True
            emit('meeting_stopped', {'message': 'Meeting stopped'})
            logger.info(f'Meeting stop request processed (session: {session_id})')
        else:
            emit('error', {'message': 'Meeting cannot be stopped, may have already ended'})

    except Exception as e:
        logger.error(f'Failed to stop meeting (session: {session_id}): {e}', exc_info=True)
        emit('error', {'message': f'Failed to stop meeting: {str(e)}'})


@socketio.on('phase_confirmation')
def handle_phase_confirmation(data):
    session_id = request.sid

    try:
        meeting = active_meetings.get(session_id)
        if not meeting:
            emit('error', {'message': 'No meeting in progress found'})
            return

        advance = data.get('advance', False)
        success = meeting.handle_external_phase_confirmation(advance)

        if success:
            logger.info(f'Phase advancement confirmation processed (session: {session_id}, advance: {advance})')
        else:
            emit('error', {'message': 'Not currently waiting for phase confirmation'})

    except Exception as e:
        logger.error(f'Failed to handle phase advancement confirmation (session: {session_id}): {e}', exc_info=True)
        emit('error', {'message': f'Failed to handle phase advancement confirmation: {str(e)}'})


# ===== Human-AI Interaction: receive user messages =====
@socketio.on('user_message')
def handle_user_message(data):
    """Receive user message injected during the meeting"""
    session_id = request.sid
    content = (data.get('content') or '').strip()
    if not content:
        return
    queue = session_user_messages.setdefault(session_id, [])
    queue.append({'content': content, 'turn': 0, 'speaker': 'You'})
    # Echo back so the user immediately sees their message even if the meeting
    # worker thread hasn't drained the queue yet
    socketio.emit('user_message', {
        'role': 'You',
        'content': content,
        'turn': 0,
    }, to=session_id, namespace='/')
    logger.info(f'User message queued (session: {session_id}, length: {len(content)})')


@socketio.on('compare_models')
def handle_compare_models(data):
    """Trigger parallel multi-model comparison (same agent, different model)"""
    session_id = request.sid
    meeting = active_meetings.get(session_id)
    if not meeting:
        emit('error', {'message': 'No active meeting'})
        return

    agent_name = data.get('agent')        # e.g. "Critic"
    models = data.get('models', [])       # list of model names, e.g. ["gpt-4.1-mini", "claude-haiku-4-20250514"]
    branch_reason = data.get('branch_reason', 'user_compare')

    if not agent_name or not models:
        emit('error', {'message': 'agent and models are required'})
        return
    if len(models) > 5:
        emit('error', {'message': 'At most 5 models can be compared at once'})
        return

    # Find the matching agent
    agent = next((a for a in meeting.agents if a.name == agent_name), None)
    if not agent:
        emit('error', {'message': f'Agent not found: {agent_name}'})
        return

    # Build branches
    branches = []
    for model in models:
        branches.append({
            "agent": agent,
            "model_override": model,
        })

    # Run asynchronously (do not block socket)
    def _do_compare():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(meeting.run_branch_nodes(
                branches,
                branch_reason=branch_reason,
            ))
        except Exception as e:
            logger.error(f'compare_models failed: {e}', exc_info=True)
            socketio.emit('error', {'message': f'Comparison failed: {e}'}, to=session_id)
        finally:
            loop.close()

    import threading
    threading.Thread(target=_do_compare, daemon=True).start()
    emit('compare_started', {
        'agent': agent_name,
        'models': models,
        'branch_reason': branch_reason,
    })
    logger.info(f'compare_models: agent={agent_name}, models={models}')


@socketio.on('rollback')
def handle_rollback(data):
    """Roll back the conversation to a specified node"""
    session_id = request.sid
    meeting = active_meetings.get(session_id)
    if not meeting:
        emit('error', {'message': 'No active meeting'})
        return
    message_id = data.get('message_id')
    if not message_id:
        emit('error', {'message': 'message_id is required'})
        return
    removed = meeting.rollback_to(message_id)
    emit('rollback_done', {
        'message_id': message_id,
        'removed_count': removed,
    })
    logger.info(f'rollback: to={message_id}, removed={removed}')


# ===== Team Templates =====
@socketio.on('get_templates')
def handle_get_templates():
    """Return all (builtin + custom) team templates to the frontend."""
    try:
        from models.team_templates import list_all_templates

        templates_payload = []
        for t in list_all_templates():
            templates_payload.append({
                "template_id": t.template_id,
                "label": t.label,
                "description": t.description,
                "applicable_problems": list(t.applicable_problems),
                "is_builtin": t.is_builtin,
                "slots": [
                    {
                        "role_name": s.role_name,
                        "default_model": s.default_model,
                        "allowed_models": list(s.allowed_models),
                        "required": s.required,
                        "weight": s.weight,
                        "phase_participation": list(s.phase_participation),
                        "functional_category": s.functional_category,
                        "system_prompt": s.system_prompt,
                    } for s in t.slots
                ],
            })
        emit('templates_list', {'templates': templates_payload})
    except Exception as e:
        logger.error(f'get_templates failed: {e}', exc_info=True)
        emit('error', {'message': f'Failed to load templates: {e}'})


@socketio.on('save_template')
def handle_save_template(data):
    """Save a custom team template (user-created)."""
    session_id = request.sid
    try:
        from models.team_templates import (
            AgentSlot, TeamTemplate, add_custom_template,
        )

        if not isinstance(data, dict):
            emit('error', {'message': 'Invalid payload'})
            return
        label = (data.get('label') or '').strip()
        if not label:
            emit('error', {'message': 'Template label is required'})
            return
        slots_in = data.get('slots') or []
        if not isinstance(slots_in, list) or len(slots_in) == 0:
            emit('error', {'message': 'At least one role slot is required'})
            return

        slots = []
        valid_phases = {"Problem Analysis", "Model Design", "Model Building", "Paper Writing"}
        valid_categories = {"analysis", "build", "write", "general"}
        for s in slots_in:
            phases_raw = list(s.get('phase_participation') or [])
            phases = [p for p in phases_raw if p in valid_phases]
            cat = s.get('functional_category') or 'general'
            if cat not in valid_categories:
                cat = 'general'
            slots.append(AgentSlot(
                role_name=(s.get('role_name') or '').strip() or 'Domain Expert',
                system_prompt=s.get('system_prompt') or '',
                allowed_models=list(s.get('allowed_models') or []),
                default_model=s.get('default_model') or '',
                required=bool(s.get('required', True)),
                weight=float(s.get('weight') or 1.0),
                phase_participation=phases,
                functional_category=cat,
            ))

        tpl = TeamTemplate(
            template_id=data.get('template_id') or '',
            label=label,
            description=data.get('description') or '',
            applicable_problems=list(data.get('applicable_problems') or ['general']),
            slots=slots,
            is_builtin=False,
        )
        tpl = add_custom_template(tpl)
        emit('template_saved', {'template_id': tpl.template_id, 'label': tpl.label})
        logger.info(f'Custom template saved (session: {session_id}, id: {tpl.template_id})')
    except Exception as e:
        logger.error(f'save_template failed: {e}', exc_info=True)
        emit('error', {'message': f'Failed to save template: {e}'})


@socketio.on('delete_template')
def handle_delete_template(data):
    """Delete a custom template by id. Rejects builtin templates."""
    session_id = request.sid
    try:
        from models.team_templates import delete_custom_template

        if not isinstance(data, dict):
            emit('error', {'message': 'Invalid payload'})
            return
        template_id = data.get('template_id')
        if not template_id:
            emit('error', {'message': 'template_id is required'})
            return
        ok = delete_custom_template(template_id)
        if not ok:
            emit('error', {'message': f'Cannot delete template (builtin or not found): {template_id}'})
            return
        emit('template_deleted', {'template_id': template_id})
        logger.info(f'Custom template deleted (session: {session_id}, id: {template_id})')
    except Exception as e:
        logger.error(f'delete_template failed: {e}', exc_info=True)
        emit('error', {'message': f'Failed to delete template: {e}'})


# ===== Chat Mode: continuous conversation =====
def _resolve_chat_model_config(model_name: str = '') -> Dict[str, Any]:
    """Resolve chat model config: user-defined → built-in text_dialogue → fallback."""
    from model_library import ModelCategory, get_model_config_by_category

    manager = get_model_manager()
    if model_name:
        # 1) User-defined model
        cfg = manager.get_config_for_agent(model_name)
        if cfg:
            return cfg
        # 2) Built-in model (treat the name as the actual model id)
        try:
            cfg = get_model_config_by_category(ModelCategory.TEXT_DIALOGUE, model_name)
            if cfg:
                return cfg
        except Exception as e:
            logger.warning(f'Built-in chat model lookup failed for {model_name}: {e}')

    # 3) Default: cheapest text_dialogue model
    try:
        cfg = get_model_config_by_category(ModelCategory.TEXT_DIALOGUE)
        if cfg:
            return cfg
    except Exception as e:
        logger.warning(f'Failed to resolve default chat model: {e}')

    # 4) Final fallback
    return {
        'provider': 'openai',
        'model': 'gpt-4.1-mini',
        'api_max_retries': Config.API_MAX_RETRIES,
    }


def _call_chat_llm(messages: List[Dict[str, str]], model_config: Dict[str, Any]) -> str:
    """Synchronously call the LLM with a list of {role, content} messages.

    Uses the sync wrapper around `call_api`, which auto-selects between
    Chat Completions and Responses API formats based on the model.
    """
    from utils.retry import retry_sync
    from utils.api_client import call_api_sync

    api_key = model_config.get('api_key') or Config.OPENAI_API_KEY
    if not api_key:
        raise ValueError('API key not configured (model and global config both empty)')

    max_tokens = model_config.get('max_tokens')
    max_retries = model_config.get('api_max_retries') or Config.API_MAX_RETRIES

    def _do_call() -> str:
        return call_api_sync(
            messages=messages,
            model_config=model_config,
            max_output_tokens=max_tokens,
        )

    return retry_sync(
        _do_call,
        max_retries=max_retries,
        delay=Config.API_RETRY_DELAY,
        exceptions=(Exception,),
    )


@socketio.on('chat_message')
def handle_chat_message(data):
    """Chat mode: receive user message, respond via single LLM call with history."""
    session_id = request.sid
    try:
        content = (data.get('content') or '').strip()
        if not content:
            return

        model_name = (data.get('model_name') or '').strip()
        history = chat_sessions.setdefault(session_id, [])
        history.append({'role': 'user', 'content': content})

        # Truncate to keep prompt size reasonable (last 40 turns = 80 messages)
        MAX_TURNS = 40
        if len(history) > MAX_TURNS * 2:
            history[:] = history[-(MAX_TURNS * 2):]

        try:
            model_config = _resolve_chat_model_config(model_name)
        except Exception as e:
            logger.error(f'Failed to resolve chat model: {e}', exc_info=True)
            socketio.emit('chat_error', {'message': f'Model configuration error: {e}'}, to=session_id, namespace='/')
            history.pop()  # Roll back user message so they can retry
            return

        socketio.emit('chat_status', {
            'status': 'thinking',
            'message': 'AI is thinking...',
            'model': model_config.get('model'),
        }, to=session_id, namespace='/')

        def run_chat():
            try:
                messages = [{'role': 'system', 'content': 'You are a helpful AI assistant. Answer the user clearly and concisely.'}]
                messages.extend(history)
                response_text = _call_chat_llm(messages, model_config)
                history.append({'role': 'assistant', 'content': response_text})
                socketio.emit('chat_response', {
                    'role': 'Assistant',
                    'content': response_text,
                    'model': model_config.get('model'),
                }, to=session_id, namespace='/')
                socketio.emit('chat_status', {'status': 'idle', 'message': 'Ready'}, to=session_id, namespace='/')
                logger.info(f'Chat response sent (session: {session_id}, model: {model_config.get("model")})')
            except Exception as e:
                logger.error(f'Chat error (session: {session_id}): {e}', exc_info=True)
                history.pop()  # Roll back user message so they can retry
                socketio.emit('chat_error', {'message': f'Chat failed: {e}'}, to=session_id, namespace='/')
                socketio.emit('chat_status', {'status': 'idle', 'message': 'Ready'}, to=session_id, namespace='/')

        threading.Thread(target=run_chat, daemon=True).start()
    except Exception as e:
        logger.error(f'Failed to handle chat_message (session: {session_id}): {e}', exc_info=True)
        socketio.emit('chat_error', {'message': f'Failed to handle message: {e}'}, to=session_id, namespace='/')


@socketio.on('clear_chat')
def handle_clear_chat(data=None):
    """Clear chat history for this session."""
    session_id = request.sid
    chat_sessions.pop(session_id, None)
    socketio.emit('chat_cleared', {}, to=session_id, namespace='/')
    logger.info(f'Chat history cleared (session: {session_id})')


# ===== Multi-modal recognition =====
@socketio.on('recognize_files')
def handle_recognize_files(data):
    """Trigger image/PDF recognition via vision model.
    
    Supports both OpenAI-compatible and Hugging Face models.
    Data format:
    {
        "files": ["/path/to/file1", "/path/to/file2"],
        "provider": "auto" | "openai" | "huggingface",  // default: auto
        "model": "gpt-4o" | "microsoft/trocr-base-printed" | etc.  // optional, uses default if not specified
    }
    """
    session_id = request.sid
    files = data.get('files') or []
    if not files:
        return

    provider = data.get('provider', 'auto')  # "auto", "openai", or "huggingface"
    model_name = data.get('model')  # Optional specific model
    prompt = data.get('prompt')  # Optional custom prompt

    def run_recognition():
        try:
            from utils.vision_handler import recognize_files_enhanced, get_vision_provider_info
            from utils.huggingface_client import list_popular_vision_models
            
            # Build configuration based on provider
            openai_config = {
                "api_key": Config.OPENAI_API_KEY,
                "base_url": Config.OPENAI_BASE_URL,
            }
            hf_config = {
                "api_key": Config.HF_API_KEY,
            }
            
            # Log provider info for debugging
            provider_info = get_vision_provider_info()
            logger.info(f"Vision provider info: openai_available={provider_info['openai_compatible']['available']}, "
                       f"hf_available={provider_info['huggingface']['available']}")
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                if provider == "huggingface":
                    # Force Hugging Face
                    if not Config.HF_API_KEY:
                        socketio.emit('meeting_update', {
                            'type': 'error',
                            'message': 'Hugging Face API key not configured. Please set HF_API_KEY in .env file.',
                        }, to=session_id, namespace='/')
                        return
                    
                    result = loop.run_until_complete(
                        recognize_files_enhanced(
                            file_paths=files,
                            provider="huggingface",
                            model_name=model_name,
                            prompt=prompt,
                            hf_config=hf_config,
                        )
                    )
                elif provider == "openai":
                    # Force OpenAI
                    result = loop.run_until_complete(
                        recognize_files_enhanced(
                            file_paths=files,
                            provider="openai",
                            model_name=model_name,
                            prompt=prompt,
                            openai_config=openai_config,
                        )
                    )
                else:
                    # Auto: try OpenAI first, then Hugging Face
                    result = loop.run_until_complete(
                        recognize_files_enhanced(
                            file_paths=files,
                            provider="auto",
                            model_name=model_name,
                            prompt=prompt,
                            openai_config=openai_config,
                            hf_config=hf_config,
                        )
                    )
            finally:
                loop.close()

            if result.get("success"):
                # Inject recognition results into meeting context as a system message
                for r in result.get("results", []):
                    name = r.get('name') or r.get('path', '')
                    content = r.get('content') or r.get('error', '')
                    if content:
                        socketio.emit('meeting_update', {
                            'type': 'message',
                            'role': 'System',
                            'content': f"📎 File recognized: {name}\n\n{content}",
                            'turn': 0,
                            'link': None,
                        }, to=session_id, namespace='/')

                # Also add to meeting conversation list (if any)
                meeting = active_meetings.get(session_id)
                if meeting and hasattr(meeting, 'add_external_context'):
                    for r in result.get("results", []):
                        name = r.get('name') or r.get('path', '')
                        content = r.get('content') or r.get('error', '')
                        if content:
                            meeting.add_external_context(f"[File: {name}] {content}")

                socketio.emit('meeting_update', {
                    'type': 'status',
                    'status': 'recognition_done',
                    'message': f'File recognition completed, {len(result.get("results", []))} files processed',
                }, to=session_id, namespace='/')
            else:
                socketio.emit('meeting_update', {
                    'type': 'error',
                    'message': f'File recognition failed: {result.get("error", "unknown")}',
                }, to=session_id, namespace='/')
        except Exception as e:
            logger.error(f'Recognition thread error: {e}', exc_info=True)
            socketio.emit('meeting_update', {
                'type': 'error',
                'message': f'Recognition error: {e}',
            }, to=session_id, namespace='/')

    threading.Thread(target=run_recognition, daemon=True).start()


def run_meeting_async(meeting, session_id, state):
    """Run the meeting in a background thread"""
    def run_in_thread():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            state.set_running()

            # Inject helper for meeting to fetch pending user messages
            def get_user_msgs():
                msgs = session_user_messages.get(session_id, [])
                if msgs:
                    session_user_messages[session_id] = []
                return msgs

            if hasattr(meeting, 'set_user_message_provider'):
                meeting.set_user_message_provider(get_user_msgs)

            result = loop.run_until_complete(run_meeting_with_stop_check(meeting, state))

            loop.close()

            if state.is_stop_requested():
                socketio.emit('meeting_stopped', {
                    'message': 'Meeting has been stopped',
                    'partial_result': result
                }, room=session_id)
                state.set_error('Meeting stopped by user')
            else:
                socketio.emit('meeting_complete', {
                    'result': result
                }, room=session_id)
                state.complete()

            if session_id in active_meetings:
                del active_meetings[session_id]

            threading.Timer(5.0, lambda: state_manager.remove_state(session_id)).start()

        except Exception as e:
            logger.error(f'Meeting execution error (session: {session_id}): {e}', exc_info=True)
            socketio.emit('error', {
                'message': f'Meeting execution error: {str(e)}'
            }, room=session_id)
            state.set_error(str(e))
            if session_id in active_meetings:
                del active_meetings[session_id]
            threading.Timer(5.0, lambda: state_manager.remove_state(session_id)).start()

    thread = threading.Thread(target=run_in_thread)
    thread.daemon = True
    thread.start()


async def run_meeting_with_stop_check(meeting, state):
    def stop_check():
        return state.is_stop_requested()
    return await meeting.run_meeting(stop_check_callback=stop_check)


# ===== Web Routes =====
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/settings')
def settings():
    return render_template('settings.html')


@app.route('/models')
def models_page():
    return render_template('models.html')


@app.route('/latex')
def latex_page():
    return render_template('latex_editor.html')

@app.route('/python')
def python_page():
    return render_template('python_repl.html')


@app.route('/api/meeting/status')
def get_meeting_status():
    try:
        session_id = request.args.get('session_id')
        if not session_id:
            return jsonify({'error': 'Missing session_id parameter'}), 400
        state = state_manager.get_state(session_id)
        if not state:
            return jsonify({'error': 'Meeting state not found'}), 404
        return jsonify(state.to_dict())
    except Exception as e:
        logger.error(f'Failed to get meeting status: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/system/stats')
def get_system_stats():
    try:
        return jsonify({
            'concurrent_meetings': state_manager.get_concurrent_count(),
            'max_concurrent': Config.MAX_CONCURRENT_MEETINGS,
            'active_sessions': len(active_meetings)
        })
    except Exception as e:
        logger.error(f'Failed to get system stats: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500


# ===== Model Management API =====
@app.route('/api/models/list', methods=['GET'])
def api_models_list():
    """List all available models (built-in + user-defined)"""
    from model_library import get_model_library, ModelCategory
    try:
        library = get_model_library()
        manager = get_model_manager()
        # Built-in models
        builtin = []
        for category in ModelCategory:
            for cfg in library.get_all_models_for_category(category):
                builtin.append({
                    "name": cfg.get("model"),
                    "provider": cfg.get("provider"),
                    "category": cfg.get("category"),
                    "is_user_defined": False,
                    "enabled": True,
                })
        # Deduplicate
        seen = set()
        builtin_unique = []
        for m in builtin:
            if m["name"] not in seen:
                seen.add(m["name"])
                builtin_unique.append(m)
        # User-defined
        user_models = manager.list_models()
        return jsonify({
            "success": True,
            "builtin_models": builtin_unique,
            "user_models": user_models,
        })
    except Exception as e:
        logger.error(f'Failed to list models: {e}', exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/models/add', methods=['POST'])
def api_models_add():
    try:
        data = request.get_json(force=True)
        manager = get_model_manager()
        model = manager.add_model(data)
        return jsonify({"success": True, "model": {
            "name": model.name,
            "provider": model.provider,
            "model": model.model,
            "category": model.category,
            "vision_capable": model.vision_capable,
            "enabled": model.enabled,
            "description": model.description,
        }})
    except Exception as e:
        logger.error(f'Failed to add model: {e}', exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/models/remove/<name>', methods=['DELETE'])
def api_models_remove(name):
    try:
        manager = get_model_manager()
        ok = manager.remove_model(name)
        return jsonify({"success": ok})
    except Exception as e:
        logger.error(f'Failed to remove model: {e}', exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/models/toggle/<name>', methods=['POST'])
def api_models_toggle(name):
    try:
        manager = get_model_manager()
        m = manager.get_model(name)
        if not m:
            return jsonify({"success": False, "error": "Model not found"}), 404
        m.enabled = not m.enabled
        manager.add_model({
            "name": m.name,
            "provider": m.provider,
            "model": m.model,
            "category": m.category,
            "base_url": m.base_url,
            "api_key": m.api_key,
            "max_tokens": m.max_tokens,
            "api_max_retries": m.api_max_retries,
            "vision_capable": m.vision_capable,
            "enabled": m.enabled,
            "description": m.description,
        })
        return jsonify({"success": True, "enabled": m.enabled})
    except Exception as e:
        logger.error(f'Failed to toggle model: {e}', exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# ===== Hugging Face Vision Models API =====
@app.route('/api/vision/models', methods=['GET'])
def api_vision_models():
    """Get available vision models from both OpenAI-compatible and Hugging Face providers"""
    try:
        from utils.vision_handler import get_vision_provider_info
        from utils.huggingface_client import list_popular_vision_models, get_available_vision_models
        from config import Config
        
        info = get_vision_provider_info()
        
        return jsonify({
            "success": True,
            "providers": info,
            "popular_hf_models": list_popular_vision_models(),
            "all_hf_models": get_available_vision_models(),
            "default_openai_model": "gpt-4o",
            "default_hf_model": Config.HF_VISION_MODEL,
        })
    except Exception as e:
        logger.error(f'Failed to get vision models: {e}', exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/vision/recognize', methods=['POST'])
def api_vision_recognize():
    """API endpoint for vision recognition (REST alternative to socket)"""
    try:
        from utils.vision_handler import recognize_files_enhanced
        import asyncio
        
        data = request.get_json(force=True) or {}
        files = data.get('files') or []
        if not files:
            return jsonify({"success": False, "error": "No files provided"}), 400
        
        provider = data.get('provider', 'auto')
        model_name = data.get('model')
        prompt = data.get('prompt')
        
        openai_config = {
            "api_key": Config.OPENAI_API_KEY,
            "base_url": Config.OPENAI_BASE_URL,
        }
        hf_config = {
            "api_key": Config.HF_API_KEY,
        }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                recognize_files_enhanced(
                    file_paths=files,
                    provider=provider,
                    model_name=model_name,
                    prompt=prompt,
                    openai_config=openai_config if provider in ('auto', 'openai') else None,
                    hf_config=hf_config if provider in ('auto', 'huggingface') else None,
                )
            )
        finally:
            loop.close()
        
        return jsonify(result)
    except Exception as e:
        logger.error(f'Vision recognition API error: {e}', exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/vision/test', methods=['POST'])
def api_vision_test():
    """Test vision recognition with a specific model"""
    try:
        from utils.huggingface_client import recognize_image_via_api, is_hf_available
        from config import Config
        import asyncio
        
        data = request.get_json(force=True) or {}
        file_path = data.get('file_path')
        model_name = data.get('model', Config.HF_VISION_MODEL)
        prompt = data.get('prompt', "Please describe this image in detail.")
        
        if not file_path:
            return jsonify({"success": False, "error": "No file_path provided"}), 400
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({"success": False, "error": f"File not found: {file_path}"}), 400
        
        if not Config.HF_API_KEY:
            return jsonify({"success": False, "error": "Hugging Face API key not configured"}), 400
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                recognize_image_via_api(
                    file_path=file_path,
                    model_name=model_name,
                    prompt=prompt,
                )
            )
        finally:
            loop.close()
        
        return jsonify({
            "success": True,
            "result": result,
            "model": model_name,
        })
    except Exception as e:
        logger.error(f'Vision test error: {e}', exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/vision/status', methods=['GET'])
def api_vision_status():
    """Get vision recognition status and configuration"""
    try:
        from utils.huggingface_client import is_hf_available
        from config import Config
        
        return jsonify({
            "success": True,
            "openai_configured": bool(Config.OPENAI_API_KEY),
            "huggingface_configured": bool(Config.HF_API_KEY),
            "huggingface_available": is_hf_available(),
            "default_hf_model": Config.HF_VISION_MODEL,
        })
    except Exception as e:
        logger.error(f'Vision status error: {e}', exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# ===== Conversation Save/Load API =====
@app.route('/api/conversations/save', methods=['POST'])
def api_save_conversation():
    try:
        data = request.get_json(force=True)
        result = save_conversation(
            session_id=data.get('session_id', 'unknown'),
            name=data.get('name', 'Untitled'),
            topic=data.get('topic', ''),
            messages=data.get('messages', []),
            metadata=data.get('metadata', {}),
        )
        return jsonify({"success": True, "filename": result["filename"]})
    except Exception as e:
        logger.error(f'Failed to save conversation: {e}', exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/conversations/list', methods=['GET'])
def api_list_conversations():
    try:
        items = list_conversations()
        return jsonify({"success": True, "conversations": items})
    except Exception as e:
        logger.error(f'Failed to list conversations: {e}', exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/conversations/load/<path:filename>', methods=['GET'])
def api_load_conversation(filename):
    try:
        data = load_conversation(filename)
        if not data:
            return jsonify({"success": False, "error": "Conversation not found"}), 404
        return jsonify({"success": True, **data})
    except Exception as e:
        logger.error(f'Failed to load conversation: {e}', exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/conversations/delete/<path:filename>', methods=['DELETE'])
def api_delete_conversation(filename):
    try:
        ok = delete_conversation(filename)
        return jsonify({"success": ok})
    except Exception as e:
        logger.error(f'Failed to delete conversation: {e}', exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# ===== LaTeX Compiler API =====
@app.route('/api/latex/compile', methods=['POST'])
def api_latex_compile():
    try:
        data = request.get_json(force=True)
        content = data.get('content', '')
        job_name = data.get('job_name')
        result = compile_latex(content, job_name)
        return jsonify(result)
    except Exception as e:
        logger.error(f'Failed to compile LaTeX: {e}', exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/latex/files', methods=['GET'])
def api_latex_files():
    try:
        files = list_tex_files()
        return jsonify({"success": True, "files": files})
    except Exception as e:
        logger.error(f'Failed to list LaTeX files: {e}', exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/latex/read/<path:filename>', methods=['GET'])
def api_latex_read(filename):
    try:
        content = read_tex_file(filename)
        if content is None:
            return jsonify({"success": False, "error": "File not found"}), 404
        return jsonify({"success": True, "content": content})
    except Exception as e:
        logger.error(f'Failed to read LaTeX file: {e}', exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/latex/save', methods=['POST'])
def api_latex_save():
    try:
        data = request.get_json(force=True)
        filename = data.get('filename', 'document.tex')
        content = data.get('content', '')
        result = save_tex_file(filename, content)
        return jsonify(result)
    except Exception as e:
        logger.error(f'Failed to save LaTeX file: {e}', exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/latex/info', methods=['GET'])
def api_latex_info():
    bin_dir = find_miktex_bin()
    return jsonify({
        "success": True,
        "miktex_found": bin_dir is not None,
        "bin_dir": bin_dir,
    })


@app.route('/latex/pdf/<path:filename>')
def serve_pdf(filename):
    """Serve compiled PDF files"""
    safe_name = Path(filename).name
    return send_from_directory(str(COMPILED_DIR), safe_name, as_attachment=False)


@app.route('/latex/log/<path:filename>')
def serve_log(filename):
    safe_name = Path(filename).name
    return send_from_directory(str(COMPILED_DIR), safe_name, as_attachment=True)


# ===== Python Code Execution =====
@app.route('/api/python/info', methods=['GET'])
def api_python_info():
    """Return local Python environment info."""
    try:
        return jsonify({"success": True, **python_get_info()})
    except Exception as e:
        logger.error(f"Python info error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/python/execute', methods=['POST'])
def api_python_execute():
    """Execute user-provided Python code in the local Python environment."""
    try:
        data = request.get_json(silent=True) or {}
        code = data.get('code', '')
        timeout = data.get('timeout')
        if not code or not str(code).strip():
            return jsonify({"success": False, "error": "Code is empty"}), 400
        result = python_execute_code(code=code, timeout=timeout)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Python execute error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/python/install', methods=['POST'])
def api_python_install():
    """Install a Python package via pip into the local environment."""
    try:
        data = request.get_json(silent=True) or {}
        package = (data.get('package') or '').strip()
        if not package:
            return jsonify({"success": False, "error": "Package name is required"}), 400
        result = python_install_package(package)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Python install error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/python/files', methods=['GET'])
def api_python_files():
    """List Python workspace files."""
    try:
        return jsonify({"success": True, "files": python_list_files()})
    except Exception as e:
        logger.error(f"Python files error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/python/workspace/<path:filename>')
def serve_python_file(filename):
    """Serve files from the Python workspace (read-only)."""
    safe_name = Path(filename).name
    return send_from_directory(str(PYTHON_WORKSPACE_DIR), safe_name, as_attachment=False)


# ===== File Upload =====
UPLOAD_DIR = os.path.join(os.getcwd(), 'uploads')
MAX_FILE_SIZE = Config.MAX_FILE_SIZE
MAX_FILES = Config.MAX_FILES
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.route('/api/upload/files', methods=['POST'])
def upload_files():
    try:
        if 'files' not in request.files:
            return jsonify({'success': False, 'error': 'No files uploaded'}), 400

        files = request.files.getlist('files')
        if not files or len(files) == 0:
            return jsonify({'success': False, 'error': 'No files selected'}), 400

        if len(files) > MAX_FILES:
            return jsonify({'success': False, 'error': f'Maximum {MAX_FILES} files can be uploaded'}), 400

        session_id = request.args.get('session_id') or request.form.get('session_id')
        if not session_id:
            if 'session_id' not in session:
                session['session_id'] = str(uuid.uuid4())
            session_id = session['session_id']

        uploaded_files = []
        session_upload_dir = os.path.join(UPLOAD_DIR, session_id)
        os.makedirs(session_upload_dir, exist_ok=True)

        for file in files:
            if file.filename == '':
                continue

            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)

            if file_size > MAX_FILE_SIZE:
                return jsonify({
                    'success': False,
                    'error': f'File {file.filename} exceeds size limit ({MAX_FILE_SIZE / 1024 / 1024}MB)'
                }), 400

            file_ext = os.path.splitext(file.filename)[1]
            unique_filename = f"{uuid.uuid4().hex}{file_ext}"
            file_path = os.path.join(session_upload_dir, unique_filename)

            file.save(file_path)

            rel_path = os.path.relpath(file_path, os.getcwd())
            rel_path = rel_path.replace('\\', '/')

            uploaded_files.append({
                'name': file.filename,
                'path': rel_path,
                'size': file_size
            })

            logger.info(f'File uploaded: {file.filename} -> {rel_path} ({file_size} bytes)')

        return jsonify({'success': True, 'files': uploaded_files})

    except Exception as e:
        logger.error(f'File upload failed: {e}', exc_info=True)
        return jsonify({'success': False, 'error': f'File upload failed: {str(e)}'}), 500


@app.route('/api/upload/files/<path:file_path>', methods=['DELETE'])
def delete_uploaded_file(file_path):
    try:
        abs_path = os.path.abspath(os.path.join(os.getcwd(), file_path))
        upload_dir_abs = os.path.abspath(UPLOAD_DIR)

        if not abs_path.startswith(upload_dir_abs):
            return jsonify({'success': False, 'error': 'Invalid file path'}), 400

        if os.path.exists(abs_path):
            os.remove(abs_path)
            logger.info(f'File deleted: {file_path}')
            session_dir = os.path.dirname(abs_path)
            try:
                if os.path.exists(session_dir) and not os.listdir(session_dir):
                    os.rmdir(session_dir)
            except:
                pass

        return jsonify({'success': True})
    except Exception as e:
        logger.error(f'Failed to delete file: {e}', exc_info=True)
        return jsonify({'success': False, 'error': f'Failed to delete file: {str(e)}'}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f'Internal server error: {error}', exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    try:
        logger.info("=" * 50)
        logger.info("AI Meeting System Web Edition")
        logger.info("=" * 50)
        logger.info(f"Access http://{Config.HOST}:{Config.PORT} to start")
        logger.info(f"Max concurrent meetings: {Config.MAX_CONCURRENT_MEETINGS}")
        logger.info("=" * 50)
        print("\n" + "=" * 50)
        print("AI Meeting System Web Edition")
        print("=" * 50)
        print(f"Access http://{Config.HOST}:{Config.PORT} to start")
        print(f"Max concurrent meetings: {Config.MAX_CONCURRENT_MEETINGS}")
        print("=" * 50 + "\n")
        # Always use the threading async mode (consistent with the python launcher,
        # to avoid extra monkey-patch side effects from eventlet/gevent).
        socketio.run(
            app,
            host=Config.HOST,
            port=Config.PORT,
            debug=Config.DEBUG,
            use_reloader=Config.WERKZEUG_RELOAD,
            allow_unsafe_werkzeug=True,
        )
    except KeyboardInterrupt:
        print("\n\nServer stopped")
        logger.info("Server stopped (user interrupted)")
    except Exception as e:
        error_msg = f"Failed to start server: {e}"
        logger.error(error_msg, exc_info=True)
        print(f"\n{error_msg}\n")
        raise
