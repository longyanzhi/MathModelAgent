// Python Execution Page
let currentRunId = null;

const SNIPPETS = {
    version: `# Python version and environment
import sys, platform
print('Python', sys.version.split()[0])
print('Executable:', sys.executable)
print('Platform:', platform.platform())
print('Working dir:', __import__('os').getcwd())`,
    packages: `# List installed packages
import importlib.metadata as md
pkgs = sorted({d.name: d.version for d in md.distributions()}.items())
print(f'{len(pkgs)} packages installed')
for name, ver in pkgs[:30]:
    print(f'  {name}=={ver}')
print('...' if len(pkgs) > 30 else '')`,
    numpy: `import numpy as np
print('NumPy', np.__version__)
a = np.arange(12).reshape(3, 4)
print('Array:')
print(a)
print('Sum:', a.sum(), '· Mean:', a.mean())`,
    matplotlib: `import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2*np.pi, 100)
plt.figure(figsize=(6, 3))
plt.plot(x, np.sin(x), label='sin(x)')
plt.plot(x, np.cos(x), label='cos(x)')
plt.legend()
plt.title('matplotlib works!')
out = 'python_workspace/matplotlib_test.png'
import os
os.makedirs('python_workspace', exist_ok=True)
plt.savefig(out, dpi=100, bbox_inches='tight')
print(f'Saved plot to: {out}')`,
    sympy: `import sympy as sp
x, y = sp.symbols('x y')
expr = sp.sin(x)**2 + sp.cos(x)**2
print('sin^2(x) + cos^2(x) simplifies to:', sp.simplify(expr))
sol = sp.solve(x**2 - 4, x)
print('Roots of x^2 - 4 = 0:', sol)`,
};

async function loadPythonInfo() {
    const infoEl = document.getElementById('pythonInfo');
    const statusEl = document.getElementById('pythonStatus');
    if (!infoEl) return;
    try {
        const resp = await fetch('/api/python/info');
        const data = await resp.json();
        if (resp.ok && data.success) {
            infoEl.innerHTML = `
                <div>🐍 Python <span class="font-semibold">${escapeHtml(data.version)}</span> · ${escapeHtml(data.platform)}</div>
                <div class="truncate" title="${escapeHtml(data.executable)}">📁 ${escapeHtml(data.executable)}</div>
                <div>📂 Prefix: ${escapeHtml(data.prefix)}</div>
            `;
            if (statusEl) statusEl.textContent = `Connected · ${data.version}`;
        } else {
            infoEl.innerHTML = '<div class="text-red-500">Failed to load Python info</div>';
        }
    } catch (e) {
        infoEl.innerHTML = '<div class="text-red-500">Failed to load Python info</div>';
    }
}

function loadSnippet(name) {
    const code = SNIPPETS[name];
    if (!code) return;
    const editor = document.getElementById('codeEditor');
    editor.value = code;
    editor.focus();
}

function appendOutput(text, cls = '') {
    const out = document.getElementById('output');
    if (!out) return;
    const placeholder = out.querySelector('.text-zinc-500');
    if (placeholder) out.innerHTML = '';
    const line = document.createElement('div');
    if (cls) line.className = cls;
    line.textContent = text;
    out.appendChild(line);
    out.scrollTop = out.scrollHeight;
}

function clearOutput() {
    const out = document.getElementById('output');
    if (out) out.innerHTML = '<div class="text-zinc-500">No output yet. Click Run to execute.</div>';
}

async function runCode() {
    const editor = document.getElementById('codeEditor');
    const timeoutEl = document.getElementById('timeoutInput');
    const code = editor ? editor.value : '';
    if (!code.trim()) {
        alert('Please enter some Python code');
        return;
    }
    const timeout = timeoutEl ? parseInt(timeoutEl.value) || 60 : 60;
    appendOutput(`\n=== Run @ ${new Date().toLocaleTimeString()} (timeout=${timeout}s) ===`, 'text-zinc-400');
    try {
        const resp = await fetch('/api/python/execute', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code: code, timeout: timeout }),
        });
        const data = await resp.json();
        currentRunId = data.run_id || null;
        if (data.stdout) appendOutput(data.stdout, 'text-zinc-100');
        if (data.stderr) appendOutput(data.stderr, 'text-red-400');
        const status = data.success ? '✓ Done' : `✗ Failed (exit ${data.exit_code ?? 'n/a'})`;
        const statusClass = data.success ? 'text-green-400' : 'text-red-400';
        const elapsed = data.duration_ms ? ` · ${data.duration_ms}ms` : '';
        appendOutput(`${status}${elapsed}`, statusClass);
    } catch (e) {
        appendOutput(`Error: ${e.message}`, 'text-red-400');
    }
}

function stopExecution() {
    // Note: backend kills subprocess on timeout; client can't cancel mid-run
    appendOutput('Stop is enforced via timeout. Wait for timeout to expire.', 'text-yellow-400');
}

async function installPackage() {
    const pkgEl = document.getElementById('packageInput');
    const out = document.getElementById('packageOutput');
    const pkg = pkgEl ? pkgEl.value.trim() : '';
    if (!pkg) {
        alert('Please enter a package name');
        return;
    }
    out.style.display = 'block';
    out.textContent = `Installing ${pkg}...`;
    try {
        const resp = await fetch('/api/python/install', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ package: pkg }),
        });
        const data = await resp.json();
        let msg = data.success ? `✓ Installed ${pkg}\n` : `✗ Failed: ${data.error || 'unknown'}\n`;
        if (data.stdout) msg += data.stdout + '\n';
        if (data.stderr) msg += data.stderr;
        out.textContent = msg;
        if (data.success && pkgEl) pkgEl.value = '';
        loadPythonInfo();
    } catch (e) {
        out.textContent = `Error: ${e.message}`;
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text ?? '';
    return div.innerHTML;
}

// Auto-resize textarea
function autoResize() {
    const el = document.getElementById('codeEditor');
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 600) + 'px';
}

document.addEventListener('DOMContentLoaded', function() {
    const editor = document.getElementById('codeEditor');
    if (editor) {
        editor.addEventListener('input', autoResize);
        editor.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                runCode();
            }
        });
        autoResize();
    }
    loadPythonInfo();
});
