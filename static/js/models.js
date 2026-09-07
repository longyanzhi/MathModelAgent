// Model Management Page
let builtinModels = [];
let userModels = [];
let editingModelName = null;

async function loadAll() {
    try {
        const resp = await fetch('/api/models/list');
        const result = await resp.json();
        if (resp.ok && result.success) {
            builtinModels = result.builtin_models || [];
            userModels = result.user_models || [];
            renderBuiltin();
            renderUser();
            populateRoleSelections();
        } else {
            console.warn('Failed to load models', result);
        }
    } catch (e) {
        console.error('Failed to load models:', e);
    }
    loadSavedSelections();
}

function renderBuiltin() {
    const listEl = document.getElementById('builtinModelsList');
    if (!listEl) return;
    if (builtinModels.length === 0) {
        listEl.innerHTML = '<div class="text-sm text-muted-foreground">No built-in models</div>';
        return;
    }
    listEl.innerHTML = builtinModels.map(m => `
        <div class="rounded-md border border-border bg-background p-3 space-y-1">
            <div class="flex items-center justify-between">
                <span class="font-medium">${escapeHtml(m.name || m.model)}</span>
                <span class="text-xs px-2 py-0.5 rounded-full bg-muted text-muted-foreground">${escapeHtml(m.category || '')}</span>
            </div>
            <div class="text-xs text-muted-foreground">provider: ${escapeHtml(m.provider || 'openai')}</div>
        </div>
    `).join('');
}

function renderUser() {
    const listEl = document.getElementById('userModelsList');
    if (!listEl) return;
    if (userModels.length === 0) {
        listEl.innerHTML = '<div class="text-sm text-muted-foreground">No custom models added yet.</div>';
        return;
    }
    listEl.innerHTML = userModels.map(m => `
        <div class="rounded-md border border-border bg-background p-3">
            <div class="flex items-center justify-between">
                <div>
                    <div class="font-medium">${escapeHtml(m.name)} ${m.enabled ? '' : '<span class="text-xs text-muted-foreground">(disabled)</span>'}</div>
                    <div class="text-xs text-muted-foreground">
                        ${escapeHtml(m.provider)} · ${escapeHtml(m.model)} · ${escapeHtml(m.category)}${m.vision_capable ? ' · vision' : ''}
                    </div>
                    ${m.description ? `<div class="text-xs text-muted-foreground mt-1">${escapeHtml(m.description)}</div>` : ''}
                </div>
                <div class="flex gap-2 flex-shrink-0">
                    <button class="px-3 h-8 rounded-md border border-input bg-background text-xs font-medium hover:bg-accent" onclick="toggleModel('${escapeAttr(m.name)}')">${m.enabled ? 'Disable' : 'Enable'}</button>
                    <button class="px-3 h-8 rounded-md border border-input bg-background text-xs font-medium hover:bg-accent" onclick="editModel('${escapeAttr(m.name)}')">Edit</button>
                    <button class="px-3 h-8 rounded-md bg-destructive text-destructive-foreground text-xs font-medium hover:opacity-90" onclick="deleteModel('${escapeAttr(m.name)}')">Delete</button>
                </div>
            </div>
        </div>
    `).join('');
}

function populateRoleSelections() {
    const all = [
        ...builtinModels.map(m => ({ name: m.name || m.model, label: `[built-in] ${m.name || m.model}` })),
        ...userModels.filter(m => m.enabled).map(m => ({ name: m.name, label: `[custom] ${m.name}` })),
    ];
    const roleKeys = [
        { key: 'Project Manager', el: 'roleSelectionProjectManager' },
        { key: 'Domain Expert', el: 'roleSelectionDomainExpert' },
        { key: 'Critic', el: 'roleSelectionCritic' },
        { key: 'Model Architect', el: 'roleSelectionArchitect' },
        { key: 'Paper Writer', el: 'roleSelectionPaperWriter' },
    ];
    for (const { el } of roleKeys) {
        const sel = document.getElementById(el);
        if (!sel) continue;
        sel.innerHTML = '<option value="">-- Use default --</option>' + all.map(o => `<option value="${escapeAttr(o.name)}">${escapeHtml(o.label)}</option>`).join('');
    }
}

function loadSavedSelections() {
    try {
        const saved = localStorage.getItem('meeting_models');
        if (!saved) return;
        const parsed = JSON.parse(saved);
        const roleKeys = [
            { key: 'Project Manager', el: 'roleSelectionProjectManager', category: 'text_dialogue' },
            { key: 'Domain Expert', el: 'roleSelectionDomainExpert', category: 'deep_thinking' },
            { key: 'Critic', el: 'roleSelectionCritic', category: 'deep_thinking' },
            { key: 'Model Architect', el: 'roleSelectionArchitect', category: 'text_dialogue' },
            { key: 'Paper Writer', el: 'roleSelectionPaperWriter', category: 'academic_writing' },
        ];
        for (const r of roleKeys) {
            const sel = document.getElementById(r.el);
            if (!sel) continue;
            const found = parsed.find(p => p.role === r.key);
            if (found) sel.value = found.model_name;
        }
    } catch (e) {
        console.warn('Failed to load saved selections', e);
    }
}

function saveSelections() {
    const roleKeys = [
        { key: 'Project Manager', el: 'roleSelectionProjectManager' },
        { key: 'Domain Expert', el: 'roleSelectionDomainExpert' },
        { key: 'Critic', el: 'roleSelectionCritic' },
        { key: 'Model Architect', el: 'roleSelectionArchitect' },
        { key: 'Paper Writer', el: 'roleSelectionPaperWriter' },
    ];
    const out = [];
    for (const r of roleKeys) {
        const sel = document.getElementById(r.el);
        if (sel && sel.value) {
            out.push({ role: r.key, model_name: sel.value });
        }
    }
    localStorage.setItem('meeting_models', JSON.stringify(out));
    alert('Selection saved. Return to main page to start meeting.');
}

async function deleteModel(name) {
    if (!confirm(`Delete custom model "${name}"?`)) return;
    try {
        const resp = await fetch(`/api/models/remove/${encodeURIComponent(name)}`, { method: 'DELETE' });
        const result = await resp.json();
        if (resp.ok && result.success) {
            await loadAll();
        } else {
            alert('Delete failed: ' + (result.error || 'unknown'));
        }
    } catch (e) {
        alert('Delete failed: ' + e.message);
    }
}

async function toggleModel(name) {
    try {
        const resp = await fetch(`/api/models/toggle/${encodeURIComponent(name)}`, { method: 'POST' });
        const result = await resp.json();
        if (resp.ok && result.success) {
            await loadAll();
        } else {
            alert('Toggle failed: ' + (result.error || 'unknown'));
        }
    } catch (e) {
        alert('Toggle failed: ' + e.message);
    }
}

function editModel(name) {
    const m = userModels.find(x => x.name === name);
    if (!m) return;
    editingModelName = name;
    document.getElementById('modelDialogTitle').textContent = `Edit Model: ${name}`;
    document.getElementById('fieldName').value = m.name || '';
    document.getElementById('fieldProvider').value = m.provider || 'openai';
    document.getElementById('fieldModel').value = m.model || '';
    document.getElementById('fieldCategory').value = m.category || 'text_dialogue';
    document.getElementById('fieldBaseUrl').value = m.base_url || '';
    document.getElementById('fieldApiKey').value = m.api_key || '';
    document.getElementById('fieldMaxTokens').value = m.max_tokens || '';
    document.getElementById('fieldMaxRetries').value = m.api_max_retries || 0;
    document.getElementById('fieldDescription').value = m.description || '';
    document.getElementById('fieldVisionCapable').checked = !!m.vision_capable;
    document.getElementById('modelDialog').classList.remove('hidden');
}

function openAddDialog() {
    editingModelName = null;
    document.getElementById('modelDialogTitle').textContent = 'Add Custom Model';
    document.getElementById('fieldName').value = '';
    document.getElementById('fieldProvider').value = 'openai';
    document.getElementById('fieldModel').value = '';
    document.getElementById('fieldCategory').value = 'text_dialogue';
    document.getElementById('fieldBaseUrl').value = '';
    document.getElementById('fieldApiKey').value = '';
    document.getElementById('fieldMaxTokens').value = '';
    document.getElementById('fieldMaxRetries').value = 0;
    document.getElementById('fieldDescription').value = '';
    document.getElementById('fieldVisionCapable').checked = false;
    document.getElementById('modelDialog').classList.remove('hidden');
}

function closeAddDialog() {
    document.getElementById('modelDialog').classList.add('hidden');
}

async function confirmAddDialog() {
    const name = document.getElementById('fieldName').value.trim();
    if (!name) {
        alert('Display name is required');
        return;
    }
    const payload = {
        name: name,
        provider: document.getElementById('fieldProvider').value,
        model: (document.getElementById('fieldModel').value || '').trim() || name,
        category: document.getElementById('fieldCategory').value,
        base_url: document.getElementById('fieldBaseUrl').value.trim() || null,
        api_key: document.getElementById('fieldApiKey').value || null,
        max_tokens: parseInt(document.getElementById('fieldMaxTokens').value) || null,
        api_max_retries: parseInt(document.getElementById('fieldMaxRetries').value) || 0,
        vision_capable: document.getElementById('fieldVisionCapable').checked,
        description: document.getElementById('fieldDescription').value.trim(),
        enabled: true,
    };
    try {
        const resp = await fetch('/api/models/add', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const result = await resp.json();
        if (resp.ok && result.success) {
            closeAddDialog();
            await loadAll();
        } else {
            alert('Save failed: ' + (result.error || 'unknown'));
        }
    } catch (e) {
        alert('Save failed: ' + e.message);
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

document.addEventListener('DOMContentLoaded', function() {
    loadAll();
    document.getElementById('addModelBtn').addEventListener('click', openAddDialog);
    document.getElementById('modelDialogCancel').addEventListener('click', closeAddDialog);
    document.getElementById('modelDialogConfirm').addEventListener('click', confirmAddDialog);
    document.getElementById('saveSelectionBtn').addEventListener('click', saveSelections);
});

// Expose handlers to inline onclick
window.deleteModel = deleteModel;
window.toggleModel = toggleModel;
window.editModel = editModel;