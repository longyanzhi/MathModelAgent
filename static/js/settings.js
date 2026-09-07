// Independent settings page script
let uploadedFiles = [];

function loadSettings() {
    try {
        const saved = localStorage.getItem('meeting_settings');
        if (!saved) return;
        const parsed = JSON.parse(saved);
        if (parsed.maxTurns) document.getElementById('maxTurns').value = parsed.maxTurns;
        document.getElementById('enableSearch').checked = !!parsed.enableSearch;

        // Load phase rounds configuration
        if (parsed.phaseTurns) {
            if (parsed.phaseTurns['Problem Analysis']) document.getElementById('phaseTurnsProblemAnalysis').value = parsed.phaseTurns['Problem Analysis'];
            if (parsed.phaseTurns['Model Design']) document.getElementById('phaseTurnsModelDesign').value = parsed.phaseTurns['Model Design'];
            if (parsed.phaseTurns['Model Building']) document.getElementById('phaseTurnsModelBuilding').value = parsed.phaseTurns['Model Building'];
            if (parsed.phaseTurns['Paper Writing']) document.getElementById('phaseTurnsPaperWriting').value = parsed.phaseTurns['Paper Writing'];
        }

        if (Array.isArray(parsed.contextFiles)) {
            uploadedFiles = parsed.contextFiles.map(path => ({ name: path.split('/').pop() || path, serverPath: path, size: 0 }));
            updateUploadedFilesList();
        }
    } catch (e) {
        console.warn('Failed to load settings', e);
    }
}

function saveSettings() {
    const maxTurns = parseInt(document.getElementById('maxTurns').value) || 5;
    const enableSearch = !!document.getElementById('enableSearch').checked;
    const contextFiles = uploadedFiles.map(f => f.serverPath);

    // Save phase rounds configuration
    const phaseTurns = {
        'Problem Analysis': parseInt(document.getElementById('phaseTurnsProblemAnalysis').value) || 20,
        'Model Design': parseInt(document.getElementById('phaseTurnsModelDesign').value) || 20,
        'Model Building': parseInt(document.getElementById('phaseTurnsModelBuilding').value) || 20,
        'Paper Writing': parseInt(document.getElementById('phaseTurnsPaperWriting').value) || 30
    };

    const payload = {
        maxTurns,
        enableSearch,
        phaseTurns,
        contextFiles
    };
    localStorage.setItem('meeting_settings', JSON.stringify(payload));
    const hint = document.getElementById('saveHint');
    if (hint) {
        hint.textContent = 'Saved, return to main page to start meeting';
        setTimeout(() => { hint.textContent = ''; }, 2000);
    }
}

function bindUpload() {
    const fileUploadBtn = document.getElementById('fileUploadBtn');
    const fileUploadInput = document.getElementById('fileUpload');
    if (fileUploadBtn && fileUploadInput) {
        fileUploadBtn.addEventListener('click', () => fileUploadInput.click());
        fileUploadInput.addEventListener('change', handleFileUpload);
    }
}

async function handleFileUpload(event) {
    const files = Array.from(event.target.files);
    if (files.length === 0) return;

    const maxFiles = 6;
    if (uploadedFiles.length + files.length > maxFiles) {
        alert(`Maximum ${maxFiles} files can be uploaded.`);
        event.target.value = '';
        return;
    }

    const formData = new FormData();
    files.forEach(f => formData.append('files', f));

    const uploadBtn = document.getElementById('fileUploadBtn');
    const originalText = uploadBtn ? uploadBtn.textContent : '📁 Select Files to Upload';

    try {
        if (uploadBtn) {
            uploadBtn.textContent = 'Uploading...';
            uploadBtn.disabled = true;
        }

        let uploadUrl = '/api/upload/files';
        const response = await fetch(uploadUrl, { method: 'POST', body: formData });
        const result = await response.json();

        if (response.ok && result.success) {
            const newFiles = result.files.map(f => ({ name: f.name, serverPath: f.path, size: f.size }));
            uploadedFiles.push(...newFiles);
            updateUploadedFilesList();
            saveSettings(); // Sync save
        } else {
            alert(result.error || 'File upload failed');
        }
    } catch (err) {
        console.error('File upload error:', err);
        alert('File upload failed: ' + err.message);
    } finally {
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
    const headerHtml = `<div class="font-semibold mb-2 text-foreground">Uploaded Files (${fileCount}/${maxFiles})</div>`;
    const html = uploadedFiles.map((file, index) => {
        return `
            <div class="flex items-center justify-between py-2 border-b border-border last:border-b-0">
                <span class="flex-1 overflow-hidden text-ellipsis whitespace-nowrap mr-2" title="${file.name}">
                    📄 ${file.name}
                </span>
                <button type="button" onclick="removeUploadedFile(${index})" class="bg-destructive text-destructive-foreground border-none rounded px-2 py-1 cursor-pointer text-xs whitespace-nowrap hover:opacity-90 transition-opacity">Delete</button>
            </div>
        `;
    }).join('');
    listEl.innerHTML = headerHtml + html;
}

async function removeUploadedFile(index) {
    if (index < 0 || index >= uploadedFiles.length) return;
    const file = uploadedFiles[index];
    try {
        const resp = await fetch(`/api/upload/files/${encodeURIComponent(file.serverPath)}`, { method: 'DELETE' });
        if (resp.ok) {
            uploadedFiles.splice(index, 1);
            updateUploadedFilesList();
            saveSettings();
        }
    } catch (e) {
        console.warn('Failed to delete file', e);
        uploadedFiles.splice(index, 1);
        updateUploadedFilesList();
        saveSettings();
    }
}

document.addEventListener('DOMContentLoaded', function() {
    bindUpload();
    loadSettings();
    document.getElementById('saveSettingsBtn').addEventListener('click', saveSettings);
});
