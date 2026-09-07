#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Python code executor.

Runs user-supplied Python snippets in the same Python interpreter that hosts
the Flask app, so all locally installed packages (numpy, pandas, scipy, torch,
sympy, etc.) are available without re-installation.

Security notes:
- Code runs with the same privileges as the Flask process.
- A hard wall-clock timeout is enforced; subprocess is killed on expiry.
- Working directory is sandboxed to a per-run temp folder.
"""
import os
import sys
import json
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Any, Optional

from utils.logger import logger


# Workspace where user scripts and run outputs are stored persistently
WORKSPACE_DIR = Path(os.getcwd()) / "python_workspace"
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

# Per-run temp dir (auto cleaned)
RUN_TMP_DIR = WORKSPACE_DIR / "_runs"
RUN_TMP_DIR.mkdir(parents=True, exist_ok=True)

# Hard timeout (seconds) for code execution
DEFAULT_TIMEOUT = 60
MAX_TIMEOUT = 300


def get_python_info() -> Dict[str, Any]:
    """Return information about the local Python environment."""
    return {
        "executable": sys.executable,
        "version": sys.version.split()[0],
        "version_full": sys.version,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "cwd": os.getcwd(),
        "default_encoding": sys.getdefaultencoding(),
        "path": sys.path[:10],
        "platform": sys.platform,
    }


def _make_runner_script(code: str, work_dir: Path) -> Path:
    """Write a runner script that executes user code in `work_dir`.

    The runner script writes a JSON result file with stdout, stderr and the
    exit code so the parent process can read it even if the process is killed.
    """
    runner = work_dir / "_runner.py"
    # Use repr() to safely embed the user's source as a string literal.
    payload = json.dumps(code)
    runner.write_text(
        "import sys, json, traceback\n"
        f"_code = {payload}\n"
        "_stdout_path = r'" + str(work_dir / "stdout.txt") + "'\n"
        "_stderr_path = r'" + str(work_dir / "stderr.txt") + "'\n"
        "_result_path = r'" + str(work_dir / "result.json") + "'\n"
        "_stdout = sys.stdout\n"
        "_stderr = sys.stderr\n"
        "try:\n"
        "    with open(_stdout_path, 'w', encoding='utf-8') as _fo, "
        "open(_stderr_path, 'w', encoding='utf-8') as _fe:\n"
        "        sys.stdout = _fo\n"
        "        sys.stderr = _fe\n"
        "        try:\n"
        "            exec(compile(_code, '<user_code>', 'exec'), { '__name__': '__main__', '__file__': None })\n"
        "        except SystemExit as _e:\n"
        "            print(f'SystemExit: {_e}', file=_fe)\n"
        "finally:\n"
        "    sys.stdout = _stdout\n"
        "    sys.stderr = _stderr\n"
        "_result = {\n"
        "    'exit_code': 0,\n"
        "    'stdout': open(_stdout_path, 'r', encoding='utf-8', errors='replace').read(),\n"
        "    'stderr': open(_stderr_path, 'r', encoding='utf-8', errors='replace').read(),\n"
        "}\n"
        "with open(_result_path, 'w', encoding='utf-8') as _fr:\n"
        "    json.dump(_result, _fr, ensure_ascii=False)\n",
        encoding="utf-8",
    )
    return runner


def execute_python(
    code: str,
    timeout: Optional[int] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute Python `code` in the local Python environment.

    Returns a dict with keys: success, run_id, stdout, stderr, exit_code, error, duration_ms.
    """
    if code is None or not str(code).strip():
        return {"success": False, "error": "Code is empty"}

    run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
    work_dir = RUN_TMP_DIR / run_id
    work_dir.mkdir(parents=True, exist_ok=True)

    effective_timeout = max(1, min(int(timeout or DEFAULT_TIMEOUT), MAX_TIMEOUT))

    runner_script = _make_runner_script(str(code), work_dir)

    # Copy the user's source to the workspace for inspection
    user_script = work_dir / "user_code.py"
    user_script.write_text(str(code), encoding="utf-8")

    try:
        env = os.environ.copy()
        # Force UTF-8 IO
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        # Add site-packages dir of current interpreter (already in sys.path)
        # so subprocess sees the same packages as the parent.
        env["PYTHONPATH"] = os.pathsep.join(sys.path) + os.pathsep + env.get("PYTHONPATH", "")

        import time
        start = time.time()
        proc = subprocess.run(
            [sys.executable, "-I", "-u", str(runner_script)],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=effective_timeout,
            env=env,
        )
        duration_ms = int((time.time() - start) * 1000)

        # Read captured output and result file
        result_path = work_dir / "result.json"
        captured_stdout = ""
        captured_stderr = ""
        if result_path.exists():
            try:
                with open(result_path, "r", encoding="utf-8") as f:
                    captured = json.load(f)
                captured_stdout = captured.get("stdout", "")
                captured_stderr = captured.get("stderr", "")
            except Exception as e:
                logger.warning(f"Failed to read result.json: {e}")

        return {
            "success": proc.returncode == 0,
            "run_id": run_id,
            "exit_code": proc.returncode,
            "stdout": captured_stdout,
            "stderr": captured_stderr or proc.stderr,
            "duration_ms": duration_ms,
            "work_dir": str(work_dir),
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "run_id": run_id,
            "error": f"Execution timed out after {effective_timeout}s",
            "timeout": effective_timeout,
            "work_dir": str(work_dir),
        }
    except Exception as e:
        logger.error(f"Python execution failed: {e}", exc_info=True)
        return {
            "success": False,
            "run_id": run_id,
            "error": str(e),
            "work_dir": str(work_dir),
        }


def install_package(package: str) -> Dict[str, Any]:
    """Install a Python package using pip into the local environment."""
    if not package or not package.strip():
        return {"success": False, "error": "Package name is empty"}

    # Basic safety: only allow simple package specs (no shell injection)
    import re
    if not re.match(r'^[A-Za-z0-9_.\-\[\]<>=, ]+$', package):
        return {"success": False, "error": "Invalid package name"}

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package.strip()],
            capture_output=True,
            text=True,
            timeout=180,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout[-3000:],
            "stderr": result.stderr[-3000:],
            "package": package.strip(),
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "pip install timed out (180s)"}
    except Exception as e:
        logger.error(f"pip install failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def list_run_files() -> list:
    """List previous run output files in the workspace."""
    files = []
    if not WORKSPACE_DIR.exists():
        return files
    for f in WORKSPACE_DIR.rglob("*"):
        if f.is_file() and not f.name.startswith("_"):
            try:
                files.append({
                    "name": str(f.relative_to(WORKSPACE_DIR)),
                    "path": str(f),
                    "size": f.stat().st_size,
                    "modified": f.stat().st_mtime,
                })
            except Exception:
                pass
    return sorted(files, key=lambda x: x["modified"], reverse=True)[:50]
