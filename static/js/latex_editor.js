// LaTeX Editor Page
let currentFilename = null;

function setStatus(msg, kind = 'info') {
    const bar = document.getElementById('statusBar');
    bar.textContent = msg;
    bar.classList.remove('hidden');
    bar.classList.remove('border-destructive', 'border-primary', 'text-destructive', 'text-primary');
    if (kind === 'error') bar.classList.add('border-destructive', 'text-destructive');
    else if (kind === 'success') bar.classList.add('border-primary', 'text-primary');
}

async function refreshMiktexStatus() {
    try {
        const resp = await fetch('/api/latex/info');
        const data = await resp.json();
        if (data.success) {
            const el = document.getElementById('miktexStatus');
            if (data.miktex_found) {
                el.textContent = '✓ MiKTeX detected';
                el.classList.remove('text-destructive');
            } else {
                el.textContent = '✗ MiKTeX not found - install and configure PATH';
                el.classList.add('text-destructive');
            }
        }
    } catch (e) {
        console.warn('Failed to query MiKTeX info', e);
    }
}

async function refreshFileList() {
    try {
        const resp = await fetch('/api/latex/files');
        const data = await resp.json();
        if (resp.ok && data.success) {
            const sel = document.getElementById('fileSelect');
            const items = (data.files || []).filter(f => f.name.endsWith('.tex'));
            sel.innerHTML = '<option value="">-- Select file --</option>' + items.map(f => `<option value="${escapeAttr(f.name)}">${escapeHtml(f.name)}</option>`).join('');
        }
    } catch (e) {
        console.error('Failed to list files:', e);
    }
}

async function loadSelectedFile() {
    const sel = document.getElementById('fileSelect');
    const filename = sel.value;
    if (!filename) {
        alert('Please select a file');
        return;
    }
    try {
        const resp = await fetch(`/api/latex/read/${encodeURIComponent(filename)}`);
        const data = await resp.json();
        if (resp.ok && data.success) {
            document.getElementById('editor').value = data.content || '';
            currentFilename = filename;
            document.getElementById('currentFilename').textContent = filename;
            setStatus(`Loaded ${filename}`, 'success');
        } else {
            setStatus('Load failed: ' + (data.error || 'unknown'), 'error');
        }
    } catch (e) {
        setStatus('Load error: ' + e.message, 'error');
    }
}

function newFile() {
    const nameInput = document.getElementById('newFilename');
    const name = (nameInput.value || '').trim();
    if (!name) {
        alert('Please enter a filename');
        return;
    }
    const safe = name.endsWith('.tex') ? name : name + '.tex';
    currentFilename = safe;
    document.getElementById('editor').value = '% New LaTeX document\n\\documentclass{article}\n\\usepackage{amsmath}\n\\begin{document}\n\n\\section{Introduction}\nHello world.\n\n\\end{document}\n';
    document.getElementById('currentFilename').textContent = safe + ' (unsaved)';
    setStatus('New file created in editor. Click Save to persist.', 'info');
}

async function saveFile() {
    const content = document.getElementById('editor').value;
    const filename = currentFilename;
    if (!filename) {
        alert('Please create or load a file first');
        return;
    }
    try {
        const resp = await fetch('/api/latex/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename, content })
        });
        const data = await resp.json();
        if (resp.ok && data.success) {
            setStatus(`Saved ${filename}`, 'success');
            await refreshFileList();
            document.getElementById('currentFilename').textContent = filename;
        } else {
            setStatus('Save failed: ' + (data.error || 'unknown'), 'error');
        }
    } catch (e) {
        setStatus('Save error: ' + e.message, 'error');
    }
}

async function compile() {
    const content = document.getElementById('editor').value;
    if (!content.trim()) {
        alert('Editor is empty');
        return;
    }
    setStatus('Compiling... this may take a while', 'info');
    try {
        // Auto-save first
        if (currentFilename) {
            await fetch('/api/latex/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename: currentFilename, content })
            });
        }
        const resp = await fetch('/api/latex/compile', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content, job_name: currentFilename ? currentFilename.replace(/\.tex$/, '') : null })
        });
        const data = await resp.json();
        if (resp.ok && data.success) {
            const url = data.pdf_url;
            document.getElementById('pdfFrame').src = url + '?t=' + Date.now();
            document.getElementById('downloadLink').href = url;
            document.getElementById('downloadLink').classList.remove('hidden');
            document.getElementById('downloadLink').textContent = 'Download PDF';
            setStatus('✓ PDF compiled successfully', 'success');
            // Show log preview if any
            if (data.log_url) {
                try {
                    const logResp = await fetch(data.log_url);
                    const logText = await logResp.text();
                    document.getElementById('logContent').textContent = logText.slice(-3000);
                    document.getElementById('logSection').classList.remove('hidden');
                } catch (_) { /* ignore log fetch errors */ }
            }
        } else {
            setStatus('✗ Compile failed: ' + (data.error || 'unknown'), 'error');
            if (data.log) {
                document.getElementById('logContent').textContent = data.log;
                document.getElementById('logSection').classList.remove('hidden');
            }
        }
    } catch (e) {
        setStatus('Compile error: ' + e.message, 'error');
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
    refreshMiktexStatus();
    refreshFileList();
    document.getElementById('loadFileBtn').addEventListener('click', loadSelectedFile);
    document.getElementById('newFileBtn').addEventListener('click', newFile);
    document.getElementById('saveBtn').addEventListener('click', saveFile);
    document.getElementById('compileBtn').addEventListener('click', compile);
    // Ctrl/Cmd+S to save
    document.getElementById('editor').addEventListener('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 's') {
            e.preventDefault();
            saveFile();
        }
    });
});