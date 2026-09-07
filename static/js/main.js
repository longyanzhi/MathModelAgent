// Main WebSocket connection and message handling
let socket = null;
let currentSessionId = null;
let isMeetingRunning = false;
let currentTopic = '';

// Role model selection
let builtinModels = [];
let userModels = [];
let currentMaxTurns = null;
let currentPhase = '';
let currentPhaseDecision = null;
let mermaidInitialized = false;

// Mode (meeting | chat)
let currentMode = 'meeting';

// Streaming support
let streamingMessageId = null;
let streamingBuffer = '';
let streamingMessageEl = null;

// Saved uploaded files
let uploadedFiles = [];

// Human-AI interaction: pending user messages sent to backend
let pendingUserMessages = [];

function collectContextFilesInput() {
    try {
        const saved = localStorage.getItem('meeting_settings');
        if (saved) {
            const parsed = JSON.parse(saved);
            if (parsed && Array.isArray(parsed.contextFiles) && parsed.contextFiles.length) {
                return parsed.contextFiles;
            }
        }
    } catch (e) {
        console.warn('Failed to read local settings:', e);
    }
    if (uploadedFiles.length > 0) {
        return uploadedFiles.map(f => f.serverPath);
    }
    return null;
}

// Markdown renderer config state
let markdownConfigured = false;

function configureMarkdown() {
    if (typeof marked === 'undefined') {
        console.warn('marked.js not loaded, will use plain text');
        markdownConfigured = false;
        return false;
    }
    if (window.markdownReady === true) {
        markdownConfigured = true;
        return true;
    }
    try {
        if (typeof marked.use === 'function') {
            marked.use({ breaks: true, gfm: true, langPrefix: 'language-' });
            markdownConfigured = true;
            window.markdownReady = true;
            return true;
        } else if (typeof marked.setOptions === 'function') {
            marked.setOptions({ breaks: true, gfm: true, langPrefix: 'language-' });
            markdownConfigured = true;
            window.markdownReady = true;
            return true;
        } else {
            markdownConfigured = true;
            window.markdownReady = true;
            return true;
        }
    } catch (err) {
        console.error('Failed to configure Markdown renderer:', err);
        markdownConfigured = false;
        return false;
    }
}

function waitForLibraries(callback) {
    let attempts = 0;
    const maxAttempts = 50;

    function check() {
        attempts++;
        if (typeof marked !== 'undefined') {
            configureMarkdown();
            if (callback) callback();
        } else if (attempts < maxAttempts) {
            setTimeout(check, 100);
        } else {
            console.warn('marked.js load timeout, will use plain text');
            markdownConfigured = false;
            if (callback) callback();
        }
    }
    check();
}

document.addEventListener('DOMContentLoaded', function() {
    if (typeof marked !== 'undefined') {
        configureMarkdown();
    }
    waitForLibraries(function() {
        initializeSocket();
        setupEventListeners();
        setupHumanAIInteraction();
        initRoleModels();
        initChatModelSelect();
        setupModeToggle();
        initVisionModels();
    });
});

if (document.readyState === 'loading') {
    // DOM still loading
} else {
    if (typeof marked !== 'undefined') {
        configureMarkdown();
    }
}

// Initialize Socket connection
function initializeSocket() {
    socket = io();

    socket.on('connect', function() {
        console.log('Connected to server');
        currentSessionId = socket.id;
        console.log('Session ID:', currentSessionId);
        updateStatus('idle', 'Connected');
    });

    socket.on('connected', function(data) {
        console.log('Connection confirmed:', data);
        // Fetch team templates once connected
        if (typeof initTeamTemplates === 'function') {
            initTeamTemplates();
        }
    });

    socket.on('meeting_started', function(data) {
        console.log('Meeting started:', data);
        isMeetingRunning = true;
        currentTopic = data.topic;
        currentMaxTurns = data.max_turns ?? null;
        currentPhase = '';
        updateStatus('running', 'Meeting in progress');
        updateUIForMeetingStart();
        if (typeof resetInput === 'function') {
            resetInput();
        }
        hidePhasePrompt();
        hidePhaseFeatures();
    });

    socket.on('meeting_update', function(data) {
        console.log('Meeting update:', data);
        handleMeetingUpdate(data);
    });

    // Streaming: incremental token chunks from a single agent message
    socket.on('stream_chunk', function(data) {
        handleStreamChunk(data);
    });

    socket.on('stream_end', function(data) {
        handleStreamEnd(data);
    });

    socket.on('stream_start', function(data) {
        handleStreamStart(data);
    });

    // User message echo — removed; display is handled optimistically in submitUserMessage()
    // socket.on('user_message', function(data) { addMessage('You', data.content, data.turn, null, 'user'); });

    socket.on('meeting_complete', function(data) {
        console.log('Meeting complete:', data);
        isMeetingRunning = false;
        hidePhasePrompt();
        hidePhaseFeatures();
        currentPhase = '';
        updateStatus('completed', 'Meeting completed');
        updateUIForMeetingEnd();
        if (data.result && data.result.final_report) {
            displayReport(data.result.final_report);
        }
    });

    socket.on('meeting_stopped', function(data) {
        console.log('Meeting stopped:', data);
        isMeetingRunning = false;
        hidePhasePrompt();
        hidePhaseFeatures();
        currentPhase = '';
        updateStatus('idle', 'Meeting stopped');
        updateUIForMeetingEnd();
        addStatusMessage('Meeting stopped');
    });

    socket.on('error', function(data) {
        console.error('Error:', data);
        alert('Error: ' + data.message);
        isMeetingRunning = false;
        updateStatus('idle', 'Error occurred');
        updateUIForMeetingEnd();
        hidePhasePrompt();
        hidePhaseFeatures();
        currentPhase = '';
    });

    // ===== Branch / DAG events (v2) =====
    socket.on('branch_result', function(data) {
        console.log('Branch result:', data);
        handleBranchResult(data);
    });
    socket.on('branches_complete', function(data) {
        console.log('Branches complete:', data);
        handleBranchesComplete(data);
    });
    socket.on('compare_started', function(data) {
        console.log('Compare started:', data);
        handleCompareStarted(data);
    });
    socket.on('rollback', function(data) {
        console.log('Rollback event:', data);
        handleRollback(data);
    });
    socket.on('rollback_done', function(data) {
        console.log('Rollback done:', data);
        handleRollbackDone(data);
    });

    // ===== Team Templates events =====
    socket.on('templates_list', function(data) {
        console.log('Templates list:', data);
        teamTemplatesCache = (data && data.templates) || [];
        if (typeof renderTemplateSelector === 'function') {
            renderTemplateSelector();
        }
    });
    socket.on('template_saved', function(data) {
        console.log('Template saved:', data);
        if (typeof addStatusMessage === 'function') {
            addStatusMessage('Template saved: ' + (data && data.label));
        }
        // Refresh list
        if (typeof initTeamTemplates === 'function') {
            initTeamTemplates();
        }
    });
    socket.on('template_deleted', function(data) {
        console.log('Template deleted:', data);
        if (typeof addStatusMessage === 'function') {
            addStatusMessage('Template deleted: ' + (data && data.template_id));
        }
        activeTemplateId = '';
        if (typeof initTeamTemplates === 'function') {
            initTeamTemplates();
        }
    });

    // ===== Chat Mode events =====
    socket.on('chat_response', function(data) {
        console.log('Chat response:', data);
        handleChatResponse(data);
    });

    socket.on('chat_status', function(data) {
        handleChatStatus(data);
    });

    socket.on('chat_error', function(data) {
        console.error('Chat error:', data);
        handleChatError(data);
    });

    socket.on('chat_cleared', function() {
        handleChatCleared();
    });

    socket.on('disconnect', function() {
        console.log('Disconnected from server');
        updateStatus('idle', 'Disconnected');
        hidePhasePrompt();
        hidePhaseFeatures();
        currentPhase = '';
    });
}

function getTopicMaxHeight() {
    return Math.max(160, Math.min(window.innerHeight * 0.4, 420));
}

function autoResizeTextarea(textarea, maxHeight = 400) {
    if (!textarea) return;
    textarea.style.height = 'auto';
    const newHeight = Math.min(textarea.scrollHeight, maxHeight);
    textarea.style.height = newHeight + 'px';
    if (textarea.scrollHeight > maxHeight) {
        textarea.style.overflowY = 'auto';
    } else {
        textarea.style.overflowY = 'hidden';
    }
}

function autoFormatText(text) {
    if (!text) return text;
    let formatted = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
    formatted = formatted.replace(/[ \t]+$/gm, '');
    formatted = formatted.replace(/\n{3,}/g, '\n\n');
    formatted = formatted.trim();
    return formatted;
}

function setupEventListeners() {
    const startBtn = document.getElementById('startBtn');
    if (startBtn) startBtn.addEventListener('click', startMeeting);

    const stopBtn = document.getElementById('stopBtn');
    if (stopBtn) stopBtn.addEventListener('click', stopMeeting);

    const clearBtn = document.getElementById('clearBtn');
    if (clearBtn) clearBtn.addEventListener('click', clearMessages);

    const exportBtn = document.getElementById('exportBtn');
    if (exportBtn) exportBtn.addEventListener('click', exportMeeting);

    const saveBtn = document.getElementById('saveBtn');
    if (saveBtn) saveBtn.addEventListener('click', showSaveDialog);

    const loadBtn = document.getElementById('loadBtn');
    if (loadBtn) loadBtn.addEventListener('click', showLoadDialog);

    // Save dialog
    const saveDialogConfirm = document.getElementById('saveDialogConfirm');
    const saveDialogCancel = document.getElementById('saveDialogCancel');
    if (saveDialogConfirm) saveDialogConfirm.addEventListener('click', confirmSave);
    if (saveDialogCancel) saveDialogCancel.addEventListener('click', hideSaveDialog);

    // Load dialog
    const loadDialogCancel = document.getElementById('loadDialogCancel');
    if (loadDialogCancel) loadDialogCancel.addEventListener('click', hideLoadDialog);

    const phaseAdvanceBtn = document.getElementById('phaseAdvanceBtn');
    const phaseContinueBtn = document.getElementById('phaseContinueBtn');
    if (phaseAdvanceBtn) {
        phaseAdvanceBtn.addEventListener('click', function() {
            if (socket && currentPhaseDecision) {
                socket.emit('phase_confirmation', { advance: true });
                hidePhasePrompt();
            }
        });
    }
    if (phaseContinueBtn) {
        phaseContinueBtn.addEventListener('click', function() {
            if (socket && currentPhaseDecision) {
                socket.emit('phase_confirmation', { advance: false });
                hidePhasePrompt();
            }
        });
    }

    const topicMainInput = document.getElementById('topicMainInput');
    function bindEnterStart(el) {
        if (!el) return;
        el.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                if (currentMode === 'chat') {
                    e.preventDefault();
                    startMeeting();
                } else if (!isMeetingRunning) {
                    e.preventDefault();
                    startMeeting();
                }
                // If meeting mode + running, the human-AI handler will pick this up
            }
        });
    }
    bindEnterStart(topicMainInput);
    if (topicMainInput) {
        const resizeTopic = () => autoResizeTextarea(topicMainInput, getTopicMaxHeight());
        topicMainInput.addEventListener('input', resizeTopic);
        resizeTopic();
    }

    const fileUploadBtnInInput = document.getElementById('fileUploadBtnInInput');
    if (fileUploadBtnInInput) {
        const hiddenFileInput = document.createElement('input');
        hiddenFileInput.type = 'file';
        hiddenFileInput.multiple = true;
        hiddenFileInput.accept = '.pdf,.png,.jpg,.jpeg,.gif,.webp,.txt,.md,.csv,.json,.docx,.doc';
        hiddenFileInput.style.display = 'none';
        document.body.appendChild(hiddenFileInput);

        fileUploadBtnInInput.addEventListener('click', () => {
            hiddenFileInput.click();
        });

        hiddenFileInput.addEventListener('change', async (e) => {
            if (e.target.files && e.target.files.length > 0) {
                await handleFileUpload(e);
            }
        });
    }
}

// ===== Human-AI Interaction: send messages during meeting =====
function setupHumanAIInteraction() {
    const input = document.getElementById('topicMainInput');
    if (!input) return;

    input.addEventListener('keydown', function(e) {
        if (!isMeetingRunning) return;
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            submitUserMessage();
        }
    });

    // Update placeholder dynamically
    const observer = new MutationObserver(() => {
        if (isMeetingRunning) {
            input.placeholder = 'Meeting in progress, type a message and press Enter to inject your opinion...';
        } else {
            input.placeholder = 'Enter meeting topic or instruction here, press Enter to start/send';
        }
    });
    observer.observe(document.body, { attributes: true });
}

// ===== Role Model Selection =====
function toggleRoleModels() {
    const panel = document.getElementById('roleModelsPanel');
    const arrow = document.getElementById('roleModelsArrow');
    if (!panel || !arrow) return;
    const hidden = panel.classList.toggle('hidden');
    arrow.style.transform = hidden ? '' : 'rotate(180deg)';
}

async function initRoleModels() {
    const roleEls = [
        'roleSelectionProjectManager',
        'roleSelectionDomainExpert',
        'roleSelectionCritic',
        'roleSelectionArchitect',
        'roleSelectionPaperWriter',
    ];
    try {
        const resp = await fetch('/api/models/list');
        const result = await resp.json();
        if (resp.ok && result.success) {
            builtinModels = result.builtin_models || [];
            userModels = result.user_models || [];
            populateRoleSelections();
            loadSavedRoleSelections();
        } else {
            roleEls.forEach(id => {
                const sel = document.getElementById(id);
                if (sel) sel.innerHTML = '<option value="">-- Load failed --</option>';
            });
            console.warn('Failed to load models:', result && result.error);
        }
    } catch (e) {
        roleEls.forEach(id => {
            const sel = document.getElementById(id);
            if (sel) sel.innerHTML = '<option value="">-- Load failed --</option>';
        });
        console.warn('Failed to load models:', e);
    }
}

function populateRoleSelections() {
    const all = [
        ...builtinModels.map(m => ({ name: m.name || m.model, label: `[built-in] ${m.name || m.model}` })),
        ...userModels.filter(m => m.enabled).map(m => ({ name: m.name, label: `[custom] ${m.name}` })),
    ];
    const roleKeys = [
        { el: 'roleSelectionProjectManager' },
        { el: 'roleSelectionDomainExpert' },
        { el: 'roleSelectionCritic' },
        { el: 'roleSelectionArchitect' },
        { el: 'roleSelectionPaperWriter' },
    ];
    for (const { el } of roleKeys) {
        const sel = document.getElementById(el);
        if (!sel) continue;
        sel.innerHTML = '<option value="">-- Default --</option>' +
            all.map(o => `<option value="${escapeAttr(o.name)}">${escapeHtml(o.label)}</option>`).join('');
    }
}

function loadSavedRoleSelections() {
    try {
        const saved = localStorage.getItem('meeting_models');
        if (!saved) return;
        const parsed = JSON.parse(saved);
        const roleKeys = [
            { key: 'Project Manager', el: 'roleSelectionProjectManager' },
            { key: 'Domain Expert', el: 'roleSelectionDomainExpert' },
            { key: 'Critic', el: 'roleSelectionCritic' },
            { key: 'Model Architect', el: 'roleSelectionArchitect' },
            { key: 'Paper Writer', el: 'roleSelectionPaperWriter' },
        ];
        for (const r of roleKeys) {
            const sel = document.getElementById(r.el);
            if (!sel) continue;
            const found = parsed.find(p => p.role === r.key);
            if (found) sel.value = found.model_name;
        }
    } catch (e) { console.warn('Failed to load role selections:', e); }
}

function saveRoleSelections() {
    const roleKeys = [
        { key: 'Project Manager', el: 'roleSelectionProjectManager' },
        { key: 'Domain Expert', el: 'roleSelectionDomainExpert' },
        { key: 'Critic', el: 'roleSelectionCritic' },
        { key: 'Model Architect', el: 'roleSelectionArchitect' },
        { key: 'Paper Writer', el: 'roleSelectionPaperWriter' },
    ];
    const selected = roleKeys
        .map(r => ({ role: r.key, model_name: document.getElementById(r.el)?.value || '' }))
        .filter(p => p.model_name);
    localStorage.setItem('meeting_models', JSON.stringify(selected));
    addStatusMessage('Role model selection saved');
}

// =========================================================================
// Team Templates
// =========================================================================
let teamTemplatesCache = [];           // [{template_id, label, description, slots, is_builtin}]
let activeTemplateId = '';             // '' = no template, use defaults
let editingSlots = [];                 // [{role_name, system_prompt, allowed_models, default_model, weight}]

function toggleTeamTemplates() {
    const panel = document.getElementById('teamTemplatesPanel');
    const arrow = document.getElementById('teamTemplatesArrow');
    if (!panel || !arrow) return;
    const hidden = panel.classList.toggle('hidden');
    arrow.style.transform = hidden ? '' : 'rotate(180deg)';
}

async function initTeamTemplates() {
    try {
        if (!socket || !socket.connected) {
            // Defer until socket connects
            return;
        }
        socket.emit('get_templates');
    } catch (e) {
        console.warn('initTeamTemplates failed:', e);
    }
}

function renderTemplateSelector() {
    const sel = document.getElementById('templateSelector');
    if (!sel) return;
    const prev = sel.value;
    sel.innerHTML = '<option value="">-- Default (no template) --</option>' +
        teamTemplatesCache.map(t => {
            const tag = t.is_builtin ? ' [built-in]' : ' [custom]';
            return `<option value="${escapeAttr(t.template_id)}">${escapeHtml(t.label + tag)}</option>`;
        }).join('');
    // Restore previous if still present
    if (prev && [...sel.options].some(o => o.value === prev)) {
        sel.value = prev;
    }
    onTemplateChanged();
}

function onTemplateChanged() {
    const sel = document.getElementById('templateSelector');
    const descEl = document.getElementById('templateDescription');
    const slotsEl = document.getElementById('templateSlotsPreview');
    const deleteBtn = document.getElementById('deleteTemplateBtn');
    if (!sel) return;
    activeTemplateId = sel.value || '';
    const tpl = teamTemplatesCache.find(t => t.template_id === activeTemplateId);
    if (!tpl) {
        if (descEl) descEl.textContent = 'No template selected - the legacy 5-role team will be used.';
        if (slotsEl) slotsEl.innerHTML = '';
        if (deleteBtn) deleteBtn.classList.add('hidden');
        return;
    }
    if (descEl) descEl.textContent = tpl.description || '';
    if (slotsEl) {
        slotsEl.innerHTML = tpl.slots.map(s => `
            <div class="rounded border border-border p-1.5 bg-background">
                <div class="flex items-center justify-between text-[10px]">
                    <span class="font-medium">${escapeHtml(s.role_name)}</span>
                    <span class="text-muted-foreground">default: ${escapeHtml(s.default_model || '(none)')}</span>
                </div>
            </div>
        `).join('');
    }
    if (deleteBtn) {
        if (tpl.is_builtin) {
            deleteBtn.classList.add('hidden');
        } else {
            deleteBtn.classList.remove('hidden');
        }
    }
}

function toggleTemplateEditor() {
    const editor = document.getElementById('templateEditor');
    if (!editor) return;
    editor.classList.toggle('hidden');
    if (!editor.classList.contains('hidden')) {
        // Seed an empty editor with one default slot
        if (editingSlots.length === 0) {
            addTemplateSlot();
        }
    }
}

function addTemplateSlot() {
    editingSlots.push({
        role_name: 'Domain Expert',
        system_prompt: '',
        allowed_models: [],
        default_model: '',
        weight: 1.0,
        phase_participation: [],
        functional_category: 'general',
    });
    renderEditingSlots();
}

function removeTemplateSlot(idx) {
    editingSlots.splice(idx, 1);
    renderEditingSlots();
}

function renderEditingSlots() {
    const root = document.getElementById('templateSlots');
    if (!root) return;
    if (editingSlots.length === 0) {
        root.innerHTML = '<div class="text-[10px] text-muted-foreground">No slots yet.</div>';
        return;
    }
    const allPhases = ['Problem Analysis', 'Model Design', 'Model Building', 'Paper Writing'];
    root.innerHTML = editingSlots.map((s, idx) => {
        const cat = s.functional_category || 'general';
        const phases = s.phase_participation || [];
        const phaseChecks = allPhases.map(p => {
            const checked = phases.includes(p) ? 'checked' : '';
            return `
                <label class="inline-flex items-center gap-1 text-[10px] text-muted-foreground mr-2">
                    <input type="checkbox" data-idx="${idx}" data-field="phase_participation" data-value="${escapeAttr(p)}" ${checked}
                           class="h-3 w-3 rounded border-input" />
                    <span>${escapeHtml(p.replace(' ', ' '))}</span>
                </label>`;
        }).join('');
        return `
        <div class="border border-border rounded p-2 space-y-1 bg-background">
            <div class="flex items-center justify-between">
                <span class="text-[10px] font-medium text-muted-foreground">Slot #${idx + 1}</span>
                <button type="button" onclick="removeTemplateSlot(${idx})" class="text-[10px] text-destructive hover:underline">Remove</button>
            </div>
            <input data-idx="${idx}" data-field="role_name" placeholder="Role name (e.g. Domain Expert)"
                   value="${escapeAttr(s.role_name)}"
                   class="w-full h-7 px-2 rounded border border-input bg-background text-[11px]" />
            <input data-idx="${idx}" data-field="default_model" placeholder="Default model name (e.g. gemini-2.5-flash-thinking)"
                   value="${escapeAttr(s.default_model)}"
                   class="w-full h-7 px-2 rounded border border-input bg-background text-[11px]" />
            <input data-idx="${idx}" data-field="allowed_models" placeholder="Allowed models (comma-separated)"
                   value="${escapeAttr((s.allowed_models || []).join(', '))}"
                   class="w-full h-7 px-2 rounded border border-input bg-background text-[11px]" />
            <div class="flex items-center gap-2 flex-wrap pt-1">
                <span class="text-[10px] text-muted-foreground">Function:</span>
                <select data-idx="${idx}" data-field="functional_category"
                        class="h-6 px-1 rounded border border-input bg-background text-[10px]">
                    <option value="general"   ${cat === 'general'   ? 'selected' : ''}>General</option>
                    <option value="analysis"  ${cat === 'analysis'  ? 'selected' : ''}>Analysis</option>
                    <option value="build"     ${cat === 'build'     ? 'selected' : ''}>Build</option>
                    <option value="write"     ${cat === 'write'     ? 'selected' : ''}>Write</option>
                </select>
            </div>
            <div class="flex items-start gap-1 flex-wrap pt-1">
                <span class="text-[10px] text-muted-foreground shrink-0 mr-1">Phases:</span>
                ${phaseChecks}
            </div>
            <textarea data-idx="${idx}" data-field="system_prompt" placeholder="Optional custom system prompt"
                      rows="2"
                      class="w-full px-2 py-1 rounded border border-input bg-background text-[11px] resize-none">${escapeHtml(s.system_prompt || '')}</textarea>
        </div>`;
    }).join('');
    // Bind change listeners
    root.querySelectorAll('input, textarea, select').forEach(el => {
        const handler = (e) => {
            const idx = parseInt(e.target.getAttribute('data-idx'), 10);
            const field = e.target.getAttribute('data-field');
            if (Number.isNaN(idx) || !editingSlots[idx]) return;
            if (field === 'phase_participation') {
                const val = e.target.getAttribute('data-value');
                const cur = new Set(editingSlots[idx].phase_participation || []);
                if (e.target.checked) cur.add(val); else cur.delete(val);
                // Empty set means "all phases" — keep the user's empty selection as-is,
                // but normalize stored value: omit default empty so backend interprets it as all.
                editingSlots[idx].phase_participation = Array.from(cur);
            } else if (field === 'functional_category') {
                editingSlots[idx][field] = e.target.value;
            } else {
                let v = e.target.value;
                if (field === 'allowed_models') {
                    v = v.split(',').map(s => s.trim()).filter(Boolean);
                }
                editingSlots[idx][field] = v;
            }
        };
        if (el.type === 'checkbox') {
            el.addEventListener('change', handler);
        } else {
            el.addEventListener('input', handler);
        }
    });
}

function saveTemplate() {
    const label = (document.getElementById('templateLabel')?.value || '').trim();
    const description = (document.getElementById('templateDescriptionInput')?.value || '').trim();
    if (!label) {
        alert('Please enter a template name');
        return;
    }
    if (editingSlots.length === 0) {
        alert('Please add at least one role slot');
        return;
    }
    const payload = {
        label,
        description,
        applicable_problems: ['general'],
        slots: editingSlots,
    };
    socket.emit('save_template', payload);
}

function deleteSelectedTemplate() {
    if (!activeTemplateId) return;
    const tpl = teamTemplatesCache.find(t => t.template_id === activeTemplateId);
    if (!tpl || tpl.is_builtin) return;
    if (!confirm(`Delete custom template "${tpl.label}"?`)) return;
    socket.emit('delete_template', { template_id: activeTemplateId });
}

function getSelectedTemplateId() {
    return activeTemplateId || '';
}

function getSelectedSlotModelOverrides() {
    // Empty for now - UI does not yet support per-slot model override
    // (the slot.default_model from the template is used directly).
    return {};
}

// ===== Mode Toggle =====
function setupModeToggle() {
    const meetingBtn = document.getElementById('modeMeetingBtn');
    const chatBtn = document.getElementById('modeChatBtn');
    if (!meetingBtn || !chatBtn) return;
    meetingBtn.addEventListener('click', () => switchMode('meeting'));
    chatBtn.addEventListener('click', () => switchMode('chat'));
}

// ===== Python Execution =====
// Moved to /python page (static/js/python_repl.js)

function submitUserMessage() {
    const input = document.getElementById('topicMainInput');
    const msg = (input.value || '').trim();
    if (!msg) return;
    if (!socket || !socket.connected) {
        alert('Not connected to server');
        return;
    }
    // Show immediately (optimistic), backend echo will re-render harmlessly
    addMessage('You', msg, 0, null, 'user');
    socket.emit('user_message', { content: msg });
    input.value = '';
    autoResizeTextarea(input);
}

// ===== Chat Mode =====
function switchMode(mode) {
    if (mode !== 'meeting' && mode !== 'chat') return;
    if (mode === 'meeting' && isMeetingRunning) {
        // Cannot switch away from an active meeting
        return;
    }
    if (mode === 'meeting' && hasActiveChatSession()) {
        if (!confirm('Switching to Meeting Mode will end the current chat session. Continue?')) {
            return;
        }
        if (socket && socket.connected) {
            socket.emit('clear_chat');
        }
    }
    currentMode = mode;

    const meetingBtn = document.getElementById('modeMeetingBtn');
    const chatBtn = document.getElementById('modeChatBtn');
    const badge = document.getElementById('modeBadge');
    const chatModelRow = document.getElementById('chatModelRow');
    const startBtn = document.getElementById('startBtn');
    const input = document.getElementById('topicMainInput');

    if (mode === 'chat') {
        meetingBtn?.classList.remove('bg-primary', 'text-primary-foreground', 'border-primary');
        meetingBtn?.classList.add('bg-background');
        chatBtn?.classList.add('bg-primary', 'text-primary-foreground', 'border-primary');
        chatBtn?.classList.remove('bg-background');
        if (badge) {
            badge.textContent = 'Chat';
            badge.classList.remove('bg-primary/10', 'text-primary');
            badge.classList.add('bg-emerald-500/10', 'text-emerald-600');
        }
        chatModelRow?.classList.remove('hidden');
        if (startBtn) {
            const label = startBtn.querySelector('[data-mode-label]');
            if (label) label.textContent = 'Send';
        }
        if (input) {
            input.placeholder = 'Type your message, press Enter to send';
        }
        updateStatus('idle', 'Chat Mode');
    } else {
        chatBtn?.classList.remove('bg-primary', 'text-primary-foreground', 'border-primary');
        chatBtn?.classList.add('bg-background');
        meetingBtn?.classList.add('bg-primary', 'text-primary-foreground', 'border-primary');
        meetingBtn?.classList.remove('bg-background');
        if (badge) {
            badge.textContent = 'Meeting';
            badge.classList.add('bg-primary/10', 'text-primary');
            badge.classList.remove('bg-emerald-500/10', 'text-emerald-600');
        }
        chatModelRow?.classList.add('hidden');
        if (startBtn) {
            const label = startBtn.querySelector('[data-mode-label]');
            if (label) label.textContent = 'Start';
        }
        if (input) {
            input.placeholder = 'Enter meeting topic or instruction here, press Enter to start/send';
        }
        updateStatus('idle', 'Waiting to start');
    }
}

function hasActiveChatSession() {
    const messages = document.querySelectorAll('#messages .message');
    if (!messages || messages.length === 0) return false;
    // Heuristic: if there is any assistant message and not in a meeting, it's an active chat session
    if (isMeetingRunning) return false;
    for (const msg of messages) {
        const role = msg.querySelector('.message-role')?.textContent || '';
        if (role === 'Assistant') return true;
    }
    return false;
}

async function initChatModelSelect() {
    const sel = document.getElementById('chatModelSelect');
    if (!sel) return;
    try {
        const resp = await fetch('/api/models/list');
        const result = await resp.json();
        if (!resp.ok || !result.success) {
            sel.innerHTML = '<option value="">-- Load failed --</option>';
            return;
        }
        const options = [
            { value: '', label: '-- Default (auto) --' },
            ...(result.builtin_models || []).map(m => ({
                value: m.name || m.model,
                label: `[built-in] ${m.name || m.model}`,
            })),
            ...(result.user_models || [])
                .filter(m => m.enabled)
                .map(m => ({ value: m.name, label: `[custom] ${m.name}` })),
        ];
        sel.innerHTML = options
            .map(o => `<option value="${escapeAttr(o.value)}">${escapeHtml(o.label)}</option>`)
            .join('');

        // Restore saved selection
        try {
            const saved = localStorage.getItem('chat_model');
            if (saved) sel.value = saved;
        } catch (e) { /* ignore */ }
        sel.addEventListener('change', () => {
            try { localStorage.setItem('chat_model', sel.value || ''); } catch (e) { /* ignore */ }
        });
    } catch (e) {
        sel.innerHTML = '<option value="">-- Load failed --</option>';
        console.warn('Failed to load chat models:', e);
    }
}

function sendChatMessage(content) {
    if (!socket || !socket.connected) {
        alert('Not connected to server');
        return;
    }
    const modelSel = document.getElementById('chatModelSelect');
    const modelName = modelSel ? modelSel.value : '';

    // Show user message immediately
    addMessage('You', content, 0, null, 'user');
    socket.emit('chat_message', { content, model_name: modelName });

    const input = document.getElementById('topicMainInput');
    if (input) {
        input.value = '';
        autoResizeTextarea(input);
    }
}

function handleChatResponse(data) {
    addMessage(data.role || 'Assistant', data.content, 0, null, getRoleClass(data.role || 'Assistant'));
}

function handleChatStatus(data) {
    if (data.status === 'thinking') {
        updateStatus('running', data.message || 'AI is thinking...');
    } else {
        updateStatus('idle', data.message || 'Chat Mode');
    }
}

function handleChatError(data) {
    addMessage('System', `Warning: ${data.message}`, 0, null, 'status');
    updateStatus('idle', 'Error occurred');
}

function handleChatCleared() {
    clearMessages();
    addStatusMessage('Chat history cleared');
}

// ===== Save / Load Conversation =====
function showSaveDialog() {
    const dlg = document.getElementById('saveDialog');
    const input = document.getElementById('saveNameInput');
    if (input) input.value = `Meeting ${new Date().toLocaleString('en-US')}`;
    if (dlg) dlg.classList.remove('hidden');
}

function hideSaveDialog() {
    const dlg = document.getElementById('saveDialog');
    if (dlg) dlg.classList.add('hidden');
}

async function confirmSave() {
    const input = document.getElementById('saveNameInput');
    const name = (input?.value || '').trim() || `Meeting ${new Date().toLocaleString('en-US')}`;
    try {
        const messages = Array.from(document.querySelectorAll('#messages .message')).map(msg => {
            const role = msg.querySelector('.message-role')?.textContent || '';
            const turn = msg.querySelector('.message-turn')?.textContent || '';
            const content = msg.querySelector('.message-content')?.textContent || msg.textContent || '';
            return { role, turn, content };
        });
        const response = await fetch('/api/conversations/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: currentSessionId,
                name: name,
                topic: currentTopic,
                messages: messages
            })
        });
        const result = await response.json();
        if (response.ok && result.success) {
            addStatusMessage(`Conversation saved: ${result.filename}`);
        } else {
            alert('Save failed: ' + (result.error || 'Unknown error'));
        }
    } catch (err) {
        alert('Save failed: ' + err.message);
    }
    hideSaveDialog();
}

async function showLoadDialog() {
    const dlg = document.getElementById('loadDialog');
    const list = document.getElementById('loadConversationsList');
    if (!dlg || !list) return;
    list.innerHTML = 'Loading...';
    dlg.classList.remove('hidden');
    try {
        const response = await fetch('/api/conversations/list');
        const result = await response.json();
        if (response.ok && result.success) {
            if (result.conversations.length === 0) {
                list.innerHTML = '<div class="text-sm text-muted-foreground">No saved conversations</div>';
            } else {
                list.innerHTML = result.conversations.map(c => `
                    <div class="flex items-center justify-between p-2 border-b border-border">
                        <div class="flex-1">
                            <div class="font-medium">${c.name}</div>
                            <div class="text-xs text-muted-foreground">${new Date(c.created_at).toLocaleString('en-US')} · ${c.message_count} messages</div>
                        </div>
                        <div class="flex gap-2">
                            <button class="px-3 h-8 rounded-md bg-primary text-primary-foreground text-xs" onclick="loadConversation('${c.filename}')">Load</button>
                            <button class="px-3 h-8 rounded-md bg-destructive text-destructive-foreground text-xs" onclick="deleteConversation('${c.filename}')">Delete</button>
                        </div>
                    </div>
                `).join('');
            }
        } else {
            list.innerHTML = '<div class="text-sm text-destructive">Load failed</div>';
        }
    } catch (err) {
        list.innerHTML = '<div class="text-sm text-destructive">' + err.message + '</div>';
    }
}

function hideLoadDialog() {
    const dlg = document.getElementById('loadDialog');
    if (dlg) dlg.classList.add('hidden');
}

async function loadConversation(filename) {
    try {
        const response = await fetch(`/api/conversations/load/${encodeURIComponent(filename)}`);
        const result = await response.json();
        if (response.ok && result.success) {
            clearMessages();
            result.messages.forEach(m => {
                addMessage(m.role, m.content, m.turn || 0, null, 'user');
            });
            currentTopic = result.topic || currentTopic;
            hideLoadDialog();
            addStatusMessage(`Loaded conversation: ${result.name}`);
        } else {
            alert('Load failed: ' + (result.error || 'Unknown error'));
        }
    } catch (err) {
        alert('Load failed: ' + err.message);
    }
}

async function deleteConversation(filename) {
    if (!confirm('Are you sure you want to delete this conversation?')) return;
    try {
        await fetch(`/api/conversations/delete/${encodeURIComponent(filename)}`, { method: 'DELETE' });
        showLoadDialog(); // Refresh list
    } catch (err) {
        alert('Delete failed: ' + err.message);
    }
}

// ===== File Upload =====
async function handleFileUpload(event) {
    const files = Array.from(event.target.files);
    if (files.length === 0) return;

    const maxFiles = 6;
    const currentFileCount = uploadedFiles.length;
    const newFileCount = files.length;

    if (currentFileCount + newFileCount > maxFiles) {
        alert(`Maximum ${maxFiles} files can be uploaded. Currently ${currentFileCount}, trying to add ${newFileCount}.`);
        event.target.value = '';
        return;
    }

    const formData = new FormData();
    files.forEach(file => formData.append('files', file));

    const uploadBtn = document.getElementById('fileUploadBtnInInput');
    const originalText = uploadBtn ? uploadBtn.textContent : 'Upload File';

    try {
        if (uploadBtn) {
            uploadBtn.textContent = 'Uploading...';
            uploadBtn.disabled = true;
        }

        let uploadUrl = '/api/upload/files';
        if (currentSessionId) {
            uploadUrl += `?session_id=${encodeURIComponent(currentSessionId)}`;
        }

        const response = await fetch(uploadUrl, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok && result.success) {
            const newFiles = result.files.map(f => ({
                name: f.name,
                serverPath: f.path,
                size: f.size
            }));
            uploadedFiles.push(...newFiles);
            updateUploadedFilesList();

            // If a vision-capable model is configured, trigger image/PDF recognition automatically
            const hasMedia = newFiles.some(f => /\.(png|jpg|jpeg|gif|webp|pdf)$/i.test(f.name));
            if (hasMedia && socket && socket.connected) {
                // Check for selected vision model
                const visionModelSelect = document.getElementById('visionModelSelect');
                const visionProviderSelect = document.getElementById('visionProviderSelect');
                const provider = visionProviderSelect ? visionProviderSelect.value : 'auto';
                const model = visionModelSelect ? visionModelSelect.value : null;
                
                socket.emit('recognize_files', { 
                    files: newFiles.map(f => f.serverPath),
                    provider: provider,
                    model: model
                });
                addStatusMessage(`Recognizing uploaded images/PDFs with ${provider === 'auto' ? 'auto-selected' : provider} provider${model ? ` (${model})` : ''}...`);
            }
        } else {
            alert(result.error || 'File upload failed');
        }
    } catch (error) {
        console.error('File upload error:', error);
        alert('File upload failed: ' + error.message);
    } finally {
        const uploadBtn = document.getElementById('fileUploadBtnInInput');
        if (uploadBtn) {
            uploadBtn.textContent = originalText;
            uploadBtn.disabled = false;
        }
        event.target.value = '';
    }
}

function updateUploadedFilesList() {
    const listEl = document.getElementById('uploadedFilesList');
    if (!listEl) return;

    if (uploadedFiles.length === 0) {
        listEl.innerHTML = '';
        return;
    }

    const fileCount = uploadedFiles.length;
    const maxFiles = 6;
    const headerHtml = `<div style="font-weight: bold; margin-bottom: 6px; color: #333;">Uploaded Files (${fileCount}/${maxFiles})</div>`;

    const html = uploadedFiles.map((file, index) => {
        const sizeKB = (file.size / 1024).toFixed(1);
        const sizeMB = (file.size / 1024 / 1024).toFixed(2);
        const sizeText = file.size > 1024 * 1024 ? `${sizeMB} MB` : `${sizeKB} KB`;
        return `
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid #eee;">
                <span style="flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin-right: 8px;" title="${file.name}">
                    File: ${file.name}
                </span>
                <span style="color: #666; font-size: 11px; margin-right: 8px;">${sizeText}</span>
                <button type="button" onclick="removeUploadedFile(${index})" style="background: #ff4444; color: white; border: none; border-radius: 3px; padding: 2px 8px; cursor: pointer; font-size: 11px; white-space: nowrap;">Delete</button>
            </div>
        `;
    }).join('');

    listEl.innerHTML = headerHtml + html;
}

async function removeUploadedFile(index) {
    if (index < 0 || index >= uploadedFiles.length) return;
    const file = uploadedFiles[index];
    try {
        const response = await fetch(`/api/upload/files/${encodeURIComponent(file.serverPath)}`, {
            method: 'DELETE'
        });
        if (response.ok) {
            uploadedFiles.splice(index, 1);
            updateUploadedFilesList();
        } else {
            uploadedFiles.splice(index, 1);
            updateUploadedFilesList();
        }
    } catch (error) {
        console.error('Delete file error:', error);
        uploadedFiles.splice(index, 1);
        updateUploadedFilesList();
    }
}

function startMeeting() {
    const mainTopicEl = document.getElementById('topicMainInput');
    const topic = (mainTopicEl?.value || '').trim();

    if (!topic) {
        alert(currentMode === 'chat' ? 'Please enter a message' : 'Please enter a meeting topic');
        return;
    }

    if (currentMode === 'chat') {
        sendChatMessage(topic);
        return;
    }

    if (isMeetingRunning) {
        alert('Meeting is in progress, please wait for it to complete');
        return;
    }

    let maxTurns = 5;
    let enableSearch = false;
    let phaseTurns = {
        'Problem Analysis': 20,
        'Model Design': 20,
        'Model Building': 20,
        'Paper Writing': 30
    };
    let selectedModels = null;
    try {
        const saved = localStorage.getItem('meeting_settings');
        if (saved) {
            const parsed = JSON.parse(saved);
            maxTurns = parseInt(parsed.maxTurns) || maxTurns;
            enableSearch = !!parsed.enableSearch;
            if (parsed.phaseTurns) {
                phaseTurns = {
                    'Problem Analysis': parseInt(parsed.phaseTurns['Problem Analysis']) || phaseTurns['Problem Analysis'],
                    'Model Design': parseInt(parsed.phaseTurns['Model Design']) || phaseTurns['Model Design'],
                    'Model Building': parseInt(parsed.phaseTurns['Model Building']) || phaseTurns['Model Building'],
                    'Paper Writing': parseInt(parsed.phaseTurns['Paper Writing']) || phaseTurns['Paper Writing']
                };
            }
        }
        const modelsSaved = localStorage.getItem('meeting_models');
        if (modelsSaved) {
            selectedModels = JSON.parse(modelsSaved);
        }
    } catch (e) {
        console.warn('Failed to read meeting settings, using defaults', e);
    }

    clearMessages();

    // Show the user's topic as a user message on the page immediately
    // so the user can see what they sent. addMessage() will replace the
    // welcome placeholder added by clearMessages() with this user message.
    addMessage('You', topic, 0, null, 'user');

    const contextFiles = collectContextFilesInput();

    const payload = {
        topic: topic,
        max_turns: maxTurns,
        phase_turns: phaseTurns
    };
    if (contextFiles && contextFiles.length) {
        payload.context_files = contextFiles;
    }
    if (selectedModels && Array.isArray(selectedModels) && selectedModels.length) {
        payload.selected_models = selectedModels;
    }
    // Team template selection (new)
    const tplId = (typeof getSelectedTemplateId === 'function') ? getSelectedTemplateId() : '';
    if (tplId) {
        payload.template_id = tplId;
        const overrides = (typeof getSelectedSlotModelOverrides === 'function')
            ? getSelectedSlotModelOverrides() : {};
        if (overrides && Object.keys(overrides).length) {
            payload.slot_model_overrides = overrides;
        }
    }

    socket.emit('start_meeting', payload);

    resetInput();
}

function resetInput() {
    const mainInput = document.getElementById('topicMainInput');
    const topicInput = document.getElementById('topic');
    const inputElement = mainInput || topicInput;

    if (inputElement) {
        inputElement.value = '';
        inputElement.style.height = 'auto';
        setTimeout(() => {
            if (inputElement.scrollHeight > 0) {
                inputElement.style.height = inputElement.scrollHeight + 'px';
            } else {
                inputElement.style.height = '72px';
            }
            if (typeof autoResizeTextarea === 'function') {
                autoResizeTextarea(inputElement);
            }
        }, 0);
    }
}

function stopMeeting() {
    console.log('Stop meeting function called', { isMeetingRunning, socket: socket ? 'connected' : 'disconnected' });

    if (!isMeetingRunning) {
        alert('No meeting in progress');
        return;
    }

    if (!socket || !socket.connected) {
        alert('Disconnected from server, cannot stop meeting');
        return;
    }

    if (confirm('Are you sure you want to stop the current meeting?')) {
        console.log('Sending stop meeting request');
        socket.emit('stop_meeting');
        isMeetingRunning = false;
        updateStatus('idle', 'Stopping meeting...');
        updateUIForMeetingEnd();
    }
}

function handleMeetingUpdate(data) {
    console.log('Processing meeting update, type:', data.type, 'data:', data);

    if (!data || !data.type) {
        console.warn('Invalid update data:', data);
        return;
    }

    switch(data.type) {
        case 'status':
            handleStatusUpdate(data);
            break;
        case 'phase':
            handlePhaseUpdate(data);
            break;
        case 'control':
            handleControlEvent(data);
            break;
        case 'message':
            console.log('Add message:', { role: data.role, turn: data.turn, contentLength: data.content?.length, dag: { messageId: data.message_id, parentId: data.parent_id, branchId: data.branch_id, modelId: data.model_id } });
            addMessage(data.role, data.content, data.turn, data.link, null, data.message_id, data.parent_id, data.branch_id, data.model_id, data.provider);
            break;
        case 'branch_result':
            // Branch results are emitted through the normal meeting callback as
            // meeting_update events. Handle them here as well as the legacy
            // direct Socket.IO listeners kept below for older server versions.
            handleBranchResult(data);
            break;
        case 'branches_complete':
            handleBranchesComplete(data);
            break;
        case 'rollback':
            handleRollback(data);
            break;
        case 'branch_selected':
            handleBranchSelected(data);
            break;
        case 'turn_start':
        case 'turn':
            handleTurnStart(data);
            break;
        case 'external_resource':
            handleExternalResource(data);
            break;
        case 'report':
            displayReport(data.report);
            break;
        default:
            console.log('Unknown update type:', data.type, 'full data:', data);
    }
}

function handleStatusUpdate(data) {
    const statusMessages = {
        'starting': 'Starting meeting...',
        'generating_report': 'Generating report...',
        'completed': 'Meeting completed'
    };

    const message = statusMessages[data.status] || data.message;
    addStatusMessage(message);
}

function handlePhaseUpdate(data) {
    currentPhase = data.phase || currentPhase || '';
    const phaseLabel = currentPhase ? `Meeting in progress - ${currentPhase} phase` : 'Meeting in progress';
    updateStatus('running', phaseLabel);

    const phaseLabelElement = document.getElementById('currentPhaseLabel');
    if (phaseLabelElement) {
        phaseLabelElement.textContent = currentPhase || '-';
    }

    hidePhasePrompt();
    hidePhaseFeatures();
    const message = data.message || (currentPhase ? `Entering phase: ${currentPhase}` : 'Phase update');
    addStatusMessage(message);
}

function handleControlEvent(data) {
    switch (data.action) {
        case 'phase_decision_required':
            const currentPhaseName = data.phase || currentPhase || 'Current Phase';
            const nextPhaseName = data.next_phase || null;
            const isFinalPhase = data.is_final_phase || (nextPhaseName === null);

            currentPhaseDecision = data;  // Store decision data so button handlers know a decision is pending
            showPhasePrompt(currentPhaseName, nextPhaseName);
            const phasesWithFeatures = ['Model Design', 'Model Building', 'Paper Writing'];
            if (nextPhaseName && phasesWithFeatures.includes(nextPhaseName)) {
                hidePhaseFeatures();
            }
            addStatusMessage(
                isFinalPhase
                    ? `System suggests ending ${currentPhaseName}, please confirm.`
                    : `System suggests advancing from ${currentPhaseName} to ${nextPhaseName}, please confirm.`
            );
            break;
        case 'phase_advance_confirmed':
            hidePhasePrompt();
            const phasesWithFeatures2 = ['Model Design', 'Model Building', 'Paper Writing'];
            if (data.next_phase && phasesWithFeatures2.includes(data.next_phase)) {
                showPhaseFeatures();
            } else {
                hidePhaseFeatures();
            }
            if (data.auto_advance) {
                addStatusMessage(
                    data.next_phase
                        ? `Auto-advance due to timeout, entering ${data.next_phase} phase.`
                        : 'Auto-completed due to timeout.'
                );
            } else {
                addStatusMessage(
                    data.next_phase
                        ? `Project Manager confirmed entering ${data.next_phase} phase.`
                        : 'Project Manager confirmed ending meeting.'
                );
            }
            break;
        case 'phase_extend_requested':
            hidePhasePrompt();
            addStatusMessage('Project Manager decided to continue current phase discussion.');
            break;
        default:
            console.log('Unhandled control event', data);
    }
}

// ===== Streaming handlers =====
function handleStreamStart(data) {
    streamingMessageId = data.message_id;
    streamingBuffer = '';
    // Create a placeholder message
    const messagesContainer = document.getElementById('messages');
    if (!messagesContainer) return;

    const welcomeMsg = messagesContainer.querySelector('.rounded-lg.border-dashed, .welcome-message');
    if (welcomeMsg) welcomeMsg.remove();

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${getRoleClass(data.role)} streaming`;
    messageDiv.setAttribute('data-message-id', data.message_id);

    messageDiv.innerHTML = `
        <div class="message-header">
            <span class="message-role">${data.role || 'AI'}</span>
            <span class="message-turn">Round ${(data.turn || 0) + 1}</span>
            <span class="streaming-indicator text-xs text-muted-foreground ml-2"> typing...</span>
        </div>
        <div class="message-content markdown-body"></div>
    `;
    messagesContainer.appendChild(messageDiv);
    streamingMessageEl = messageDiv;
    scrollMessagesToBottom();
}

function handleStreamChunk(data) {
    if (data.message_id !== streamingMessageId) return;
    streamingBuffer += data.delta || '';
    if (streamingMessageEl) {
        const contentEl = streamingMessageEl.querySelector('.message-content');
        if (contentEl) {
            contentEl.innerHTML = renderMarkdown(streamingBuffer);
        }
        scrollMessagesToBottom();
    }
}

function handleStreamEnd(data) {
    if (data.message_id !== streamingMessageId) return;
    if (streamingMessageEl) {
        streamingMessageEl.classList.remove('streaming');
        const indicator = streamingMessageEl.querySelector('.streaming-indicator');
        if (indicator) indicator.remove();
        // Final re-render with math/code highlighting
        const contentEl = streamingMessageEl.querySelector('.message-content');
        if (contentEl) {
            // Re-attach classes for highlight.js
            setTimeout(() => {
                if (typeof hljs !== 'undefined') {
                    contentEl.querySelectorAll('pre code').forEach((block) => {
                        if (!block.classList.contains('hljs')) {
                            hljs.highlightElement(block);
                        }
                    });
                }
                if (typeof renderMathInElement !== 'undefined') {
                    renderMathInElement(contentEl, {
                        delimiters: [
                            {left: '$$', right: '$$', display: true},
                            {left: '$', right: '$', display: false},
                            {left: '\\[', right: '\\]', display: true},
                            {left: '\\(', right: '\\)', display: false}
                        ],
                        throwOnError: false
                    });
                }
            }, 0);
        }
    }
    streamingMessageId = null;
    streamingMessageEl = null;
    streamingBuffer = '';
}

function scrollMessagesToBottom() {
    const messagesContainer = document.getElementById('messages');
    if (!messagesContainer) return;

    let scrollContainer = messagesContainer.parentElement;
    while (scrollContainer && !scrollContainer.classList.contains('overflow-y-auto')) {
        scrollContainer = scrollContainer.parentElement;
    }

    if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
    } else {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
}

function handleExternalResource(data) {
    const source = (data.source || 'external').toUpperCase();
    const items = Array.isArray(data.items) ? data.items : [];
    addStatusMessage(`External resources loaded: ${source} (${items.length} results)`);

    if (!items.length) return;

    const container = document.getElementById('messages');
    if (!container) return;

    const wrapper = document.createElement('div');
    wrapper.className = 'message external-resource';

    const header = document.createElement('div');
    header.className = 'message-header';
    header.textContent = `${source} External Research Reference`;
    wrapper.appendChild(header);

    const body = document.createElement('div');
    body.className = 'message-content';
    const list = document.createElement('ul');
    list.className = 'external-resource-list';

    items.forEach((item, index) => {
        const li = document.createElement('li');
        const title = item.title || (item.raw && item.raw.title) || `Result ${index + 1}`;
        const published = item.published || (item.raw && item.raw.published) || '';
        const url = item.url || item.pdf_url || (item.raw && (item.raw.url || item.raw.pdf_url));

        const titleSpan = document.createElement('span');
        titleSpan.className = 'external-resource-title';
        titleSpan.textContent = `${index + 1}. ${title}`;
        li.appendChild(titleSpan);

        if (published) {
            const publishedSpan = document.createElement('span');
            publishedSpan.className = 'external-resource-meta';
            publishedSpan.textContent = ` - ${published}`;
            li.appendChild(publishedSpan);
        }

        if (url) {
            const link = document.createElement('a');
            link.href = url;
            link.textContent = 'View';
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            link.className = 'external-resource-link';
            li.appendChild(document.createTextNode(' '));
            li.appendChild(link);
        }

        const summary = item.summary || (item.raw && (item.raw.summary || item.raw.abstract));
        if (summary) {
            const summaryDiv = document.createElement('div');
            summaryDiv.className = 'external-resource-summary';
            summaryDiv.textContent = summary.length > 220 ? `${summary.slice(0, 217)}...` : summary;
            li.appendChild(summaryDiv);
        }

        list.appendChild(li);
    });

    body.appendChild(list);
    wrapper.appendChild(body);
    container.appendChild(wrapper);
    scrollMessagesToBottom();
}

function handleTurnStart(data) {
    const total = (typeof data.max_turns === 'number') ? data.max_turns : null;
    const progress = total ? (data.turn / total) * 100 : 0;
    updateProgress(progress, data.turn, total);
    addStatusMessage(`Starting round ${data.turn} of discussion`);
}

function unwrapMarkdownFence(content) {
    if (!content) return content;
    return content.replace(/```(?:\s*(?:markdown|md))\s*[\r\n]+([\s\S]*?)\s*```/gi, (match, inner) => {
        const trimmedInner = inner.trim();
        const needsTrailingNewline = /\n\s*$/.test(match);
        return trimmedInner + (needsTrailingNewline ? '\n' : '');
    });
}

function renderMarkdown(content) {
    if (!content) return '';

    content = unwrapMarkdownFence(content);

    const mathFormulas = [];
    let processedContent = content;

    const codeBlockPlaceholders = [];
    processedContent = processedContent.replace(/```[\s\S]*?```/g, (match) => {
        const id = `@@@CODE_BLOCK_TEMP_${codeBlockPlaceholders.length}@@@`;
        codeBlockPlaceholders.push(match);
        return id;
    });

    processedContent = processedContent.replace(/\$\$[\s\S]*?\$\$/g, (match) => {
        const id = `@@@MATH_BLOCK_${mathFormulas.length}@@@`;
        mathFormulas.push({ type: 'block', formula: match });
        return id;
    });
    processedContent = processedContent.replace(/\\\[[\s\S]*?\\\]/g, (match) => {
        const id = `@@@MATH_BLOCK_${mathFormulas.length}@@@`;
        mathFormulas.push({ type: 'block', formula: match });
        return id;
    });

    processedContent = processedContent.replace(/\\\([\s\S]*?\\\)/g, (match) => {
        const id = `@@@MATH_INLINE_${mathFormulas.length}@@@`;
        mathFormulas.push({ type: 'inline', formula: match });
        return id;
    });
    processedContent = processedContent.replace(/(^|[^$\\])\$([^$\n]+?)\$([^$]|$)/g, (match, before, formula, after) => {
        if (before !== '$' && after !== '$') {
            const id = `@@@MATH_INLINE_${mathFormulas.length}@@@`;
            mathFormulas.push({ type: 'inline', formula: '$' + formula + '$' });
            return before + id + after;
        }
        return match;
    });

    codeBlockPlaceholders.forEach((codeBlock, index) => {
        processedContent = processedContent.replace(`@@@CODE_BLOCK_TEMP_${index}@@@`, codeBlock);
    });

    if (typeof marked !== 'undefined') {
        try {
            if (!markdownConfigured) configureMarkdown();

            let result = null;

            if (typeof marked.parse === 'function') {
                result = marked.parse(processedContent);
            } else if (typeof marked === 'function') {
                result = marked(processedContent);
            }

            if (result && typeof result === 'string' && result.length > 0) {
                let finalResult = result;
                mathFormulas.forEach((math, index) => {
                    const placeholder = math.type === 'block'
                        ? `@@@MATH_BLOCK_${index}@@@`
                        : `@@@MATH_INLINE_${index}@@@`;
                    if (finalResult.includes(placeholder)) {
                        finalResult = finalResult.replace(placeholder, math.formula);
                    } else {
                        const escapedPlaceholder = placeholder.replace(/@/g, '&#64;');
                        if (finalResult.includes(escapedPlaceholder)) {
                            finalResult = finalResult.replace(escapedPlaceholder, math.formula);
                        }
                    }
                });
                return finalResult;
            }
        } catch (err) {
            console.error('Markdown rendering error:', err);
        }
    }

    const codeBlocks = [];
    let formatted = processedContent.replace(/```[\s\S]*?```/g, (match) => {
        const id = `@@@CODE_BLOCK_${codeBlocks.length}@@@`;
        codeBlocks.push(match);
        return id;
    });

    const inlineCodes = [];
    formatted = formatted.replace(/`([^`\n]+)`/g, (match, code) => {
        const id = `@@@INLINE_CODE_${inlineCodes.length}@@@`;
        inlineCodes.push(escapeHtml(code));
        return id;
    });

    formatted = formatted.replace(/^###\s+(.+)$/gm, '<h3>$1</h3>');
    formatted = formatted.replace(/^##\s+(.+)$/gm, '<h2>$1</h2>');
    formatted = formatted.replace(/^#\s+(.+)$/gm, '<h1>$1</h1>');
    formatted = formatted.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    formatted = formatted.replace(/__([^_]+)__/g, '<strong>$1</strong>');

    formatted = formatted.replace(/^(\d+)\.\s+(.+)$/gm, '<li data-list="ordered" data-order="$1">$2</li>');
    formatted = formatted.replace(/^[-*+]\s+(.+)$/gm, '<li data-list="unordered">$1</li>');

    formatted = formatted.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
    formatted = formatted.replace(/_([^_\n]+)_/g, '<em>$1</em>');

    const htmlTags = [];
    formatted = formatted.replace(/<[^>]+>/g, (match) => {
        const id = `@@@HTML_TAG_${htmlTags.length}@@@`;
        htmlTags.push(match);
        return id;
    });

    formatted = escapeHtml(formatted);

    htmlTags.forEach((tag, index) => {
        formatted = formatted.replace(`@@@HTML_TAG_${index}@@@`, tag);
    });

    formatted = formatted.replace(/(<li data-list="(ordered|unordered)"(?: data-order="(\d+)")?>[\s\S]*?<\/li>(?:\s*<li data-list="\2"(?: data-order="(\d+)")?>[\s\S]*?<\/li>)*)/g, (match, _content, listType) => {
        if (listType === 'ordered') {
            const orderMatches = [...match.matchAll(/data-order="(\d+)"/g)].map((m) => parseInt(m[1], 10));
            const listItems = match.replace(/<li data-list="ordered"(?: data-order="(\d+)")?>([\s\S]*?)<\/li>/g, (liMatch, order, inner) => {
                const valueAttr = order ? ` value="${order}"` : '';
                return `<li${valueAttr}>${inner}</li>`;
            });
            const startAttr = orderMatches.length ? ` start="${orderMatches[0]}"` : '';
            return `<ol${startAttr}>${listItems}</ol>`;
        } else {
            const listItems = match.replace(/<li data-list="unordered">([\s\S]*?)<\/li>/g, '<li>$1</li>');
            return `<ul>${listItems}</ul>`;
        }
    });

    inlineCodes.forEach((code, index) => {
        formatted = formatted.replace(`@@@INLINE_CODE_${index}@@@`, `<code>${code}</code>`);
    });

    codeBlocks.forEach((block, index) => {
        const codeBlockMatch = block.match(/^```(\w+)?\n?([\s\S]*?)```$/);
        if (codeBlockMatch) {
            const language = codeBlockMatch[1] || '';
            const code = codeBlockMatch[2].trim();
            const langClass = language ? ` class="language-${language}"` : '';
            formatted = formatted.replace(`@@@CODE_BLOCK_${index}@@@`, `<pre><code${langClass}>${escapeHtml(code)}</code></pre>`);
        } else {
            const code = block.replace(/```\w*\n?/g, '').replace(/```/g, '').trim();
            formatted = formatted.replace(`@@@CODE_BLOCK_${index}@@@`, `<pre><code>${escapeHtml(code)}</code></pre>`);
        }
    });

    formatted = formatted.split(/\n\n+/).map(para => {
        para = para.trim();
        if (!para) return '';
        if (para.match(/^<[hulol]/)) return para;
        return '<p>' + para + '</p>';
    }).join('\n');

    formatted = formatted.replace(/\n/g, '<br>');

    mathFormulas.forEach((math, index) => {
        const placeholder = math.type === 'block'
            ? `@@@MATH_BLOCK_${index}@@@`
            : `@@@MATH_INLINE_${index}@@@`;
        formatted = formatted.replace(placeholder, math.formula);
    });

    return formatted;
}

function initializeMermaid() {
    if (mermaidInitialized) return true;
    if (typeof mermaid === 'undefined') {
        console.warn('Mermaid library not loaded, cannot initialize');
        return false;
    }
    try {
        mermaid.initialize({
            startOnLoad: false,
            securityLevel: 'loose'
        });
        mermaidInitialized = true;
        return true;
    } catch (err) {
        console.warn('Mermaid initialization failed:', err);
        return false;
    }
}

function renderMermaidDiagrams(container) {
    if (!container) return;
    const codeBlocks = container.querySelectorAll('pre code.language-mermaid, pre code.mermaid');
    if (!codeBlocks || codeBlocks.length === 0) return;

    if (typeof mermaid === 'undefined') {
        console.warn('Mermaid library not loaded, cannot render diagrams');
        return;
    }

    if (!initializeMermaid()) return;

    const nodes = [];

    codeBlocks.forEach((codeBlock) => {
        const pre = codeBlock.closest('pre');
        if (!pre) return;

        const diagramDefinition = codeBlock.textContent;
        const wrapper = document.createElement('div');
        wrapper.className = 'mermaid-diagram';

        const mermaidDiv = document.createElement('div');
        mermaidDiv.className = 'mermaid';
        mermaidDiv.textContent = diagramDefinition;

        wrapper.appendChild(mermaidDiv);
        pre.replaceWith(wrapper);

        nodes.push(mermaidDiv);
    });

    if (nodes.length > 0 && typeof mermaid.run === 'function') {
        mermaid.run({ nodes }).catch(err => {
            console.warn('Mermaid rendering failed:', err);
        });
    } else if (nodes.length > 0 && typeof mermaid.init === 'function') {
        try {
            mermaid.init(undefined, nodes);
        } catch (err) {
            console.warn('Mermaid rendering failed:', err);
        }
    }
}

function addMessage(role, content, turn, link, msgClass, messageId = null, parentId = null, branchId = null, modelId = null, provider = null) {
    console.log('addMessage called:', { role, contentLength: content?.length, turn, messageId, parentId, branchId, modelId, provider });

    const messagesContainer = document.getElementById('messages');
    if (!messagesContainer) {
        console.error('Cannot find message container element #messages');
        return;
    }

    const welcomeMsg = messagesContainer.querySelector('.rounded-lg.border-dashed, .welcome-message');
    if (welcomeMsg) welcomeMsg.remove();

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${msgClass || getRoleClass(role)}`;

    const renderedContent = renderMarkdown(content);

    let linkHtml = '';
    if (link) {
        const imageExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg', '.bmp'];
        const isImage = imageExtensions.some(ext => link.toLowerCase().includes(ext)) ||
                       link.match(/\.(jpg|jpeg|png|gif|webp|svg|bmp)(\?|$)/i);

        if (isImage) {
            linkHtml = `<div class="message-link"><img src="${escapeHtml(link)}" alt="image" style="max-width: 100%; height: auto; border-radius: 4px; margin-top: 8px;" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';"><a href="${escapeHtml(link)}" target="_blank" style="display: none; color: #0066cc; text-decoration: none; margin-top: 8px;">View Image</a></div>`;
        } else {
            linkHtml = `<div class="message-link"><a href="${escapeHtml(link)}" target="_blank" style="color: #0066cc; text-decoration: none; margin-top: 8px; display: inline-block;">View Link</a></div>`;
        }
    }

    // v2 DAG: show model name, branch label
    const modelBadge = (modelId || provider) ? `<span class="message-model-badge" title="Model: ${escapeHtml(modelId || provider)}">${escapeHtml(modelId || provider)}</span>` : '';
    const branchBadge = branchId ? `<span class="message-branch-badge" title="Branch: ${escapeHtml(branchId)}">Branch ${escapeHtml(branchId.slice(0, 4))}</span>` : '';

    messageDiv.innerHTML = `
        <div class="message-header">
            <span class="message-role">${role}</span>
            ${modelBadge}${branchBadge}
            <span class="message-turn">Round ${(turn || 0) + 1}</span>
        </div>
        <div class="message-content markdown-body">${renderedContent}${linkHtml}</div>
        <div class="message-actions">
            <button class="msg-action-btn" data-action="copy" title="Copy">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                <span>Copy</span>
            </button>
            ${role !== 'You' ? `
            <button class="msg-action-btn" data-action="regenerate" title="Regenerate">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>
                <span>Regenerate</span>
            </button>
            <button class="msg-action-btn" data-action="rollback" title="Rollback to here">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"></path><path d="M3 3v5h5"></path></svg>
                <span>Rollback</span>
            </button>` : ''}
        </div>
    `;
    if (messageId) {
        messageDiv.setAttribute('data-message-id', messageId);
    }
    if (parentId) {
        messageDiv.setAttribute('data-parent-id', parentId);
    }
    if (branchId) {
        messageDiv.setAttribute('data-branch-id', branchId);
    }
    if (modelId) {
        messageDiv.setAttribute('data-model-id', modelId);
    }
    if (provider) {
        messageDiv.setAttribute('data-provider', provider);
    }
    if (role) {
        messageDiv.setAttribute('data-message-role', role);
    }

    messagesContainer.appendChild(messageDiv);

    // Bind action buttons
    messageDiv.querySelectorAll('.msg-action-btn').forEach((btn) => {
        btn.addEventListener('click', (ev) => {
            ev.stopPropagation();
            const action = btn.getAttribute('data-action');
            if (action === 'copy') {
                copyMessageContent(messageDiv);
            } else if (action === 'regenerate') {
                const mid = messageDiv.getAttribute('data-message-id');
                const rname = messageDiv.getAttribute('data-message-role');
                regenerateMessage(mid, rname);
            } else if (action === 'rollback') {
                const mid = messageDiv.getAttribute('data-message-id');
                requestRollback(mid);
            }
        });
    });
    console.log('Message added to DOM:', { role, turn, contentLength: content?.length });

    scrollMessagesToBottom();

    // Update DAG visualization after adding message
    setTimeout(() => {
        renderDag();
    }, 100);

    setTimeout(() => {
        if (typeof hljs !== 'undefined') {
            messageDiv.querySelectorAll('pre code').forEach((block) => {
                const isMermaid = block.classList.contains('language-mermaid') || block.classList.contains('mermaid');
                const isLatex = block.classList.contains('language-latex') || block.classList.contains('latex');
                if (isMermaid || isLatex) return;
                try {
                    if (!block.classList.contains('hljs')) {
                        hljs.highlightElement(block);
                    }
                } catch (err) {
                    console.warn('Code highlighting failed:', err);
                }
            });
        }

        renderMermaidDiagrams(messageDiv);

        if (typeof renderMathInElement !== 'undefined') {
            try {
                renderMathInElement(messageDiv, {
                    delimiters: [
                        {left: '$$', right: '$$', display: true},
                        {left: '$', right: '$', display: false},
                        {left: '\\[', right: '\\]', display: true},
                        {left: '\\(', right: '\\)', display: false}
                    ],
                    throwOnError: false,
                    strict: false
                });
            } catch (err) {
                console.warn('Math formula rendering failed:', err);
            }
        } else if (typeof katex !== 'undefined') {
            try {
                messageDiv.querySelectorAll('.message-content').forEach(element => {
                    element.innerHTML = element.innerHTML.replace(/\$\$([\s\S]*?)\$\$/g, (match, formula) => {
                        try {
                            return katex.renderToString(formula.trim(), { displayMode: true, throwOnError: false });
                        } catch (e) {
                            return match;
                        }
                    });
                    element.innerHTML = element.innerHTML.replace(/([^$]|^)\$([^$\n]+?)\$([^$]|$)/g, (match, before, formula, after) => {
                        if (before !== '$' && after !== '$') {
                            try {
                                return before + katex.renderToString(formula.trim(), { displayMode: false, throwOnError: false }) + after;
                            } catch (e) {
                                return match;
                            }
                        }
                        return match;
                    });
                });
            } catch (err) {
                console.warn('Manual math formula rendering failed:', err);
            }
        }
    }, 0);
}

// ===== v2: Per-message actions (Copy / Regenerate) =====

function copyMessageContent(messageEl) {
    try {
        const txt = messageEl.querySelector('.message-content')?.innerText || '';
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(txt).then(() => flashAction(messageEl, 'Copied'));
        } else {
            const textarea = document.createElement('textarea');
            textarea.value = txt;
            document.body.appendChild(textarea);
            textarea.select();
            document.execCommand('copy');
            document.body.removeChild(textarea);
            flashAction(messageEl, 'Copied');
        }
    } catch (err) {
        console.error('Copy failed:', err);
        alert('Copy failed');
    }
}

function flashAction(messageEl, text) {
    const btn = messageEl.querySelector('.msg-action-btn[data-action="copy"]');
    if (!btn) return;
    const span = btn.querySelector('span');
    const original = span ? span.textContent : '';
    if (span) span.textContent = text;
    setTimeout(() => {
        if (span) span.textContent = original;
    }, 1500);
}

async function regenerateMessage(messageId, role) {
    if (!confirm('Regenerate this message? This will discard the current response.')) return;
    if (!socket || !socket.connected) {
        alert('Not connected to server');
        return;
    }
    socket.emit('regenerate_message', { message_id: messageId, role: role });
}

async function requestRollback(messageId) {
    if (!confirm('Rollback to this point? All subsequent messages will be removed.')) return;
    if (!socket || !socket.connected) {
        alert('Not connected to server');
        return;
    }
    socket.emit('rollback_request', { message_id: messageId });
}

function getRoleClass(role) {
    const r = (role || '').toLowerCase();
    if (r.includes('manager') || r.includes('moderator')) return 'moderator';
    if (r.includes('expert') || r.includes('domain')) return 'expert';
    if (r.includes('architect')) return 'architect';
    if (r.includes('writer') || r.includes('paper')) return 'architect';
    if (r.includes('critic') || r.includes('analyst')) return 'analyst';
    if (r.includes('you') || r.includes('user')) return 'user';
    return '';
}

function clearMessages() {
    const container = document.getElementById('messages');
    if (!container) return;
    container.innerHTML = `
        <div class="rounded-lg border border-dashed border-border bg-muted/40 p-6 text-xs text-muted-foreground">
            <h3 class="text-sm font-medium text-foreground mb-2">Welcome to AI Meeting System</h3>
            <p>Please open "Meeting Settings" to configure the topic and click "Start Meeting". The system will organize multiple AI agents for collaborative discussion.</p>
        </div>
    `;
    document.getElementById('reportSection')?.classList.add('hidden');
}

function displayReport(report) {
    const section = document.getElementById('reportSection');
    const content = document.getElementById('reportContent');
    if (!section || !content) return;
    content.innerHTML = renderMarkdown(report || 'No report generated.');
    section.classList.remove('hidden');
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function exportMeeting() {
    const messages = Array.from(document.querySelectorAll('#messages .message'));
    if (messages.length === 0) {
        alert('No messages to export');
        return;
    }
    const lines = [];
    lines.push('# Meeting Export\n');
    lines.push(`Topic: ${currentTopic || 'N/A'}\n`);
    lines.push(`Exported: ${new Date().toLocaleString('en-US')}\n`);
    lines.push('---\n');
    messages.forEach(msg => {
        const role = msg.querySelector('.message-role')?.textContent || '';
        const turn = msg.querySelector('.message-turn')?.textContent || '';
        const body = msg.querySelector('.message-content')?.innerText || msg.textContent || '';
        if (role) lines.push(`\n## ${role} ${turn}\n`);
        lines.push(body.trim());
        lines.push('\n');
    });
    const text = lines.join('\n');
    const blob = new Blob([text], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `meeting_${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
}

function updateStatus(status, message) {
    const statusEl = document.getElementById('status');
    if (!statusEl) return;
    statusEl.innerHTML = `<span class="status-badge ${status}">${escapeHtml(message)}</span>`;
}

function addStatusMessage(message) {
    console.log('Status:', message);
    updateStatus('running', message);
}

function updateProgress(percent, current, total) {
    const bar = document.getElementById('progress');
    const text = document.getElementById('progressText');
    if (bar) bar.style.width = Math.max(0, Math.min(100, percent)) + '%';
    if (text) text.textContent = `${current} / ${total} rounds`;
}

function showPhasePrompt(currentPhase, nextPhase) {
    const el = document.getElementById('phasePrompt');
    const tip = document.getElementById('phasePromptTip');
    if (!el) return;
    if (tip) {
        if (nextPhase) {
            tip.textContent = `Advance from ${currentPhase} to ${nextPhase}?`;
        } else {
            tip.textContent = `End ${currentPhase} and complete the meeting?`;
        }
    }
    el.classList.remove('hidden');
    document.getElementById('phaseAdvanceBtn')?.classList.remove('hidden');
    document.getElementById('phaseContinueBtn')?.classList.remove('hidden');
}

function hidePhasePrompt() {
    const el = document.getElementById('phasePrompt');
    if (el) el.classList.add('hidden');
    document.getElementById('phaseAdvanceBtn')?.classList.add('hidden');
    document.getElementById('phaseContinueBtn')?.classList.add('hidden');
    currentPhaseDecision = null;  // Reset after handling
}

function hidePhaseFeatures() {
    // Hide feature panels that appear in certain phases
}

function showPhaseFeatures() {
    // Show feature panels that appear in certain phases
}

// ===== v2: DAG visualization =====

let dagInitialized = false;

function ensureDagPanel() {
    // The panel is part of the page shell. Keeping it in the markup makes the
    // view available immediately, instead of relying on a brittle CSS selector
    // to find a particular Tailwind sidebar width.
    const panel = document.getElementById('meetingDagContainer')
        || document.querySelector('[data-dag-panel]');
    if (!panel) return null;

    if (panel.dataset.dagReady !== 'true') {
        const body = panel.querySelector('#meetingDagBody');
        const toggle = panel.querySelector('#dagToggleBtn');
        const header = panel.querySelector('.dag-header');
        if (toggle) {
            toggle.addEventListener('click', (event) => {
                event.stopPropagation();
                toggleDagPanel();
            });
        }
        if (header && toggle) {
            header.addEventListener('click', () => toggleDagPanel());
        }
        syncDagPanelState();
        panel.dataset.dagReady = 'true';
    }
    dagInitialized = true;
    return panel;
}

function syncDagPanelState() {
    const panel = document.getElementById('meetingDagContainer')
        || document.querySelector('[data-dag-panel]');
    if (!panel) return;
    const body = panel.querySelector('#meetingDagBody');
    const toggle = panel.querySelector('#dagToggleBtn');
    const collapsed = panel.classList.contains('collapsed');
    if (body) body.classList.toggle('hidden', collapsed);
    if (toggle) {
        toggle.textContent = collapsed ? 'Expand' : 'Collapse';
        toggle.setAttribute('aria-expanded', String(!collapsed));
    }
}

function toggleDagPanel() {
    const panel = document.getElementById('meetingDagContainer')
        || document.querySelector('[data-dag-panel]');
    if (!panel) return;
    panel.classList.toggle('collapsed');
    syncDagPanelState();
}

function indexAllMessages() {
    const map = new Map();
    document.querySelectorAll('#messages .message').forEach((el) => {
        const mid = el.getAttribute('data-message-id');
        if (!mid) return;
        map.set(mid, {
            id: mid,
            role: el.getAttribute('data-message-role') || el.querySelector('.message-role')?.textContent || '?',
            parentId: el.getAttribute('data-parent-id') || null,
            branchId: el.getAttribute('data-branch-id') || null,
            modelId: el.getAttribute('data-model-id') || null,
            provider: el.getAttribute('data-provider') || null,
            preview: (el.querySelector('.message-content')?.innerText || '').trim().slice(0, 48),
            el,
        });
    });
    return map;
}

function renderDag() {
    ensureDagPanel();
    const body = document.getElementById('meetingDagBody');
    if (!body) return;

    const map = indexAllMessages();
    if (map.size === 0) {
        body.innerHTML = '<div class="dag-empty">No messages yet</div>';
        return;
    }

    // Flatten DAG: every node rendered at the same indent level (no hierarchy).
    // We still surface parent relationships via a small visual indicator so
    // users can see the DAG structure without nested indentation.
    const children = new Map();
    map.forEach((node) => {
        if (node.parentId && map.has(node.parentId) && map.get(node.parentId).branchId === node.branchId) {
            if (!children.has(node.parentId)) children.set(node.parentId, []);
            children.get(node.parentId).push(node);
        }
    });

    // Flatten via DFS, but track depth only for visual cues (no margin-indent).
    const flat = [];
    const roots = [];
    map.forEach((node) => {
        if (!node.parentId || !map.has(node.parentId) || map.get(node.parentId).branchId !== node.branchId) {
            roots.push(node);
        }
    });
    const visit = (node, depth) => {
        flat.push({ node, depth });
        const kids = children.get(node.id) || [];
        kids.forEach((child) => visit(child, depth + 1));
    };
    roots.forEach((root) => visit(root, 0));

    body.innerHTML = flat
        .map(({ node, depth }) => renderDagNode(node, depth, children.get(node.id)?.length || 0))
        .join('') || '<div class="dag-empty">No messages yet</div>';

    body.querySelectorAll('[data-dag-mid]').forEach((el) => {
        el.addEventListener('click', () => {
            const mid = el.getAttribute('data-dag-mid');
            const target = map.get(mid);
            if (target?.el) {
                target.el.scrollIntoView({ behavior: 'smooth', block: 'center' });
                target.el.style.outline = '2px solid #6366f1';
                setTimeout(() => { target.el.style.outline = ''; }, 1200);
            }
        });
    });
}

function renderDagNode(node, depth, childCount) {
    // Flat layout: no margin-left indentation. We surface DAG relationships
    // through a non-breaking visual cue so the tree structure remains visible.
    const branchTag = node.branchId
        ? `<span class="dag-branch" title="Branch: ${escapeHtml(node.branchId)}">·${escapeHtml(node.branchId.slice(0, 4))}</span>`
        : '';
    const modelTag = node.modelId
        ? `<span class="dag-model" title="Model: ${escapeHtml(node.modelId)}">${escapeHtml(node.modelId)}</span>`
        : '';
    const parentCue = depth > 0
        ? `<span class="dag-depth-cue" title="Depth ${depth}">↳ d${depth}</span>`
        : `<span class="dag-depth-cue dag-depth-root" title="Root node">●</span>`;
    const kidCount = childCount > 0
        ? ` <span class="dag-child-count">[${childCount}]</span>`
        : '';
    return `<div class="dag-node dag-node-flat" data-dag-mid="${escapeAttr(node.id)}">
        ${parentCue}
        <span class="dag-role">${escapeHtml(node.role)}</span>
        ${modelTag}${branchTag}
        <span class="dag-preview">${escapeHtml(node.preview || '(empty)')}</span>${kidCount}
    </div>`;
}

function scrollToMessage(messageId) {
    const msg = document.querySelector(`[data-message-id="${messageId}"]`);
    if (msg) {
        msg.scrollIntoView({ behavior: 'smooth', block: 'center' });
        msg.style.outline = '2px solid #6366f1';
        setTimeout(() => { msg.style.outline = ''; }, 2000);
    }
}

// ===== v2: Branch comparison =====

let compareData = null;

function openCompareDialog() {
    const dlg = document.getElementById('compareDialog');
    if (dlg) dlg.classList.remove('hidden');
}

function closeCompareDialog() {
    const dlg = document.getElementById('compareDialog');
    if (dlg) dlg.classList.add('hidden');
}

function handleCompareStarted(data) {
    compareData = data;
    const dlg = document.getElementById('compareDialog');
    const results = document.getElementById('compareResults');
    if (!dlg || !results) return;
    results.innerHTML = '<div class="text-sm text-muted-foreground">Comparing models...</div>';
    dlg.classList.remove('hidden');
}

function handleBranchResult(data) {
    if (!compareData) return;
    compareData.results = compareData.results || [];
    compareData.results.push(data);
    renderCompareResults();
}

function handleBranchesComplete(data) {
    if (!compareData) return;
    compareData.results = data.results || [];
    renderCompareResults();
    closeCompareDialog();
}

function renderCompareResults() {
    if (!compareData || !compareData.results) return;
    const results = document.getElementById('compareResults');
    if (!results) return;
    results.innerHTML = compareData.results.map(r => `
        <div class="compare-result-item">
            <div class="compare-result-header">
                <span class="compare-model">${escapeHtml(r.model || 'Unknown')}</span>
            </div>
            <div class="compare-result-preview">${escapeHtml((r.preview || '').slice(0, 200))}</div>
            <button class="compare-use-btn" onclick="useCompareResult('${r.branch_id}')">Use This</button>
        </div>
    `).join('');
}

function useCompareResult(branchId) {
    if (!socket || !socket.connected) return;
    socket.emit('use_branch', { branch_id: branchId });
    closeCompareDialog();
}

// ===== Rollback handlers =====

function handleRollback(data) {
    const msg = document.querySelector(`[data-message-id="${data.message_id}"]`);
    if (msg) {
        msg.style.outline = '2px solid #dc2626';
    }
    addStatusMessage(`Rolling back to message ${data.message_id.slice(0, 8)}...`);
}

function handleRollbackDone(data) {
    // Remove all messages after the rolled-back point
    const messages = document.querySelectorAll('#messages .message');
    let foundTarget = false;
    messages.forEach(msg => {
        const id = msg.getAttribute('data-message-id');
        if (id === data.rolled_back_to) {
            foundTarget = true;
        } else if (foundTarget) {
            msg.remove();
        }
    });
    addStatusMessage('Rollback complete');
    renderDag();
}

// ===== UI helpers =====

function updateUIForMeetingStart() {
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    if (startBtn) startBtn.style.display = 'none';
    if (stopBtn) stopBtn.style.display = 'flex';
}

function updateUIForMeetingEnd() {
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    if (startBtn) startBtn.style.display = 'flex';
    if (stopBtn) stopBtn.style.display = 'none';
    const input = document.getElementById('topicMainInput');
    if (input) {
        input.placeholder = 'Enter meeting topic or instruction here, press Enter to start/send';
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text || '';
    return div.innerHTML;
}

function escapeAttr(text) {
    return (text || '').replace(/'/g, '&#39;').replace(/"/g, '&quot;');
}

// ===== Vision Model Selection =====
let visionModelsCache = null;

async function initVisionModels() {
    const statusEl = document.getElementById('visionStatus');
    const providerSelect = document.getElementById('visionProviderSelect');
    const modelSelect = document.getElementById('visionModelSelect');
    const modelContainer = document.getElementById('visionModelContainer');
    
    if (!providerSelect) return;
    
    try {
        // Fetch vision model info from backend
        const resp = await fetch('/api/vision/models');
        const result = await resp.json();
        
        if (resp.ok && result.success) {
            visionModelsCache = result;
            
            // Update status
            if (statusEl) {
                let status = [];
                if (result.providers?.openai_compatible?.available) {
                    status.push('OpenAI ✓');
                } else {
                    status.push('OpenAI ✗');
                }
                if (result.providers?.huggingface?.available) {
                    status.push('HF ✓');
                } else {
                    status.push('HF ✗');
                }
                statusEl.textContent = status.join(' | ');
            }
            
            // Load saved selection
            try {
                const saved = localStorage.getItem('vision_model');
                if (saved) {
                    const parsed = JSON.parse(saved);
                    if (parsed.provider) providerSelect.value = parsed.provider;
                    if (parsed.model && modelSelect) modelSelect.value = parsed.model;
                }
            } catch (e) { /* ignore */ }
            
            // Update model list based on provider
            updateVisionModelList();
        }
    } catch (e) {
        console.warn('Failed to load vision models:', e);
        if (statusEl) statusEl.textContent = 'Load failed';
    }
}

function onVisionProviderChange() {
    updateVisionModelList();
    saveVisionModelSelection();
}

function updateVisionModelList() {
    const providerSelect = document.getElementById('visionProviderSelect');
    const modelSelect = document.getElementById('visionModelSelect');
    const modelContainer = document.getElementById('visionModelContainer');
    
    if (!providerSelect || !modelSelect) return;
    
    const provider = providerSelect.value;
    
    // Show/hide model container based on provider
    if (provider === 'auto') {
        modelContainer?.classList.add('hidden');
        return;
    } else {
        modelContainer?.classList.remove('hidden');
    }
    
    if (!visionModelsCache) return;
    
    // Clear options
    modelSelect.innerHTML = '<option value="">Default Model</option>';
    
    if (provider === 'openai') {
        // OpenAI models
        const openaiModels = visionModelsCache.providers?.openai_compatible?.models || [];
        openaiModels.forEach(model => {
            const opt = document.createElement('option');
            opt.value = model;
            opt.textContent = model;
            modelSelect.appendChild(opt);
        });
    } else if (provider === 'huggingface') {
        // Hugging Face models
        const popularModels = visionModelsCache.popular_hf_models || [];
        popularModels.forEach(model => {
            const opt = document.createElement('option');
            opt.value = model.id;
            opt.textContent = `${model.name} (${model.category})`;
            opt.title = model.description;
            modelSelect.appendChild(opt);
        });
    }
    
    // Restore saved model selection
    try {
        const saved = localStorage.getItem('vision_model');
        if (saved) {
            const parsed = JSON.parse(saved);
            if (parsed.model) modelSelect.value = parsed.model;
        }
    } catch (e) { /* ignore */ }
}

function saveVisionModelSelection() {
    const providerSelect = document.getElementById('visionProviderSelect');
    const modelSelect = document.getElementById('visionModelSelect');
    
    const provider = providerSelect?.value || 'auto';
    const model = modelSelect?.value || '';
    
    try {
        localStorage.setItem('vision_model', JSON.stringify({ provider, model }));
    } catch (e) { /* ignore */ }
}

// Listen for model select changes
document.addEventListener('DOMContentLoaded', function() {
    const modelSelect = document.getElementById('visionModelSelect');
    if (modelSelect) {
        modelSelect.addEventListener('change', saveVisionModelSelection);
    }
    // Wire up the DAG panel toggle button immediately so it is clickable even
    // before the first message arrives (renderDag is not called yet).
    ensureDagPanel();
});

// Expose functions to window for inline onclick handlers
window.startMeeting = startMeeting;
window.stopMeeting = stopMeeting;
window.clearMessages = clearMessages;
window.exportMeeting = exportMeeting;
window.showSaveDialog = showSaveDialog;
window.hideSaveDialog = hideSaveDialog;
window.confirmSave = confirmSave;
window.showLoadDialog = showLoadDialog;
window.hideLoadDialog = hideLoadDialog;
window.loadConversation = loadConversation;
window.deleteConversation = deleteConversation;
window.handleFileUpload = handleFileUpload;
window.removeUploadedFile = removeUploadedFile;
window.toggleRoleModels = toggleRoleModels;
window.saveRoleSelections = saveRoleSelections;
window.toggleTeamTemplates = toggleTeamTemplates;
window.onTemplateChanged = onTemplateChanged;
window.toggleTemplateEditor = toggleTemplateEditor;
window.addTemplateSlot = addTemplateSlot;
window.removeTemplateSlot = removeTemplateSlot;
window.saveTemplate = saveTemplate;
window.deleteSelectedTemplate = deleteSelectedTemplate;
window.getSelectedTemplateId = getSelectedTemplateId;
window.getSelectedSlotModelOverrides = getSelectedSlotModelOverrides;
window.initTeamTemplates = initTeamTemplates;
window.switchMode = switchMode;
window.copyMessageContent = copyMessageContent;
window.regenerateMessage = regenerateMessage;
window.requestRollback = requestRollback;
window.scrollToMessage = scrollToMessage;
window.toggleDagPanel = toggleDagPanel;
window.openCompareDialog = openCompareDialog;
window.closeCompareDialog = closeCompareDialog;
window.useCompareResult = useCompareResult;
window.renderDag = renderDag;
window.initVisionModels = initVisionModels;
window.onVisionProviderChange = onVisionProviderChange;
