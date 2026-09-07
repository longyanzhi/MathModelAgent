#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LaTeX compiler backend
Integrates MiKTeX (pdflatex) for compiling LaTeX to PDF
"""
import os
import subprocess
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from utils.logger import logger

# Default MiKTeX install path
DEFAULT_MIKTEX_PATHS = [
    r"C:\Program Files\MiKTeX",
    r"C:\Program Files (x86)\MiKTeX",
    os.path.expanduser(r"~\AppData\Local\Programs\MiKTeX"),
]


def find_miktex_bin() -> Optional[str]:
    """Find pdflatex.exe in MiKTeX install"""
    candidates = os.environ.get("LATEX_BIN_PATH")
    if candidates and Path(candidates).exists():
        return candidates
    for base in DEFAULT_MIKTEX_PATHS:
        for sub in ["miktex/bin/x64", "bin/x64", "bin"]:
            exe_dir = Path(base) / sub
            if exe_dir.exists():
                exe = exe_dir / ("pdflatex.exe" if os.name == "nt" else "pdflatex")
                if exe.exists():
                    return str(exe_dir)
    # Try system PATH
    sys_pdflatex = shutil.which("pdflatex")
    if sys_pdflatex:
        return str(Path(sys_pdflatex).parent)
    return None


# Working dir for compiled output
COMPILED_DIR = Path(os.getcwd()) / "latex_workspace"
COMPILED_DIR.mkdir(exist_ok=True)


def compile_latex(content: str, job_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Compile LaTeX content to PDF using pdflatex.
    Returns {success, pdf_path, log, ...}
    """
    bin_dir = find_miktex_bin()
    if not bin_dir:
        return {
            "success": False,
            "error": "MiKTeX (pdflatex) not found. Please ensure MiKTeX is installed and accessible.",
        }

    job_name = job_name or f"latex_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    work_dir = COMPILED_DIR / job_name
    work_dir.mkdir(parents=True, exist_ok=True)

    tex_path = work_dir / f"{job_name}.tex"
    tex_path.write_text(content, encoding="utf-8")

    try:
        # First pass
        result1 = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "PATH": bin_dir + os.pathsep + os.environ.get("PATH", "")},
        )
        # Second pass for refs
        result2 = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "PATH": bin_dir + os.pathsep + os.environ.get("PATH", "")},
        )
        pdf_path = work_dir / f"{job_name}.pdf"
        log_path = work_dir / f"{job_name}.log"

        if pdf_path.exists():
            # Move PDF to a stable URL
            stable_pdf = COMPILED_DIR / f"{job_name}.pdf"
            shutil.copy(str(pdf_path), str(stable_pdf))
            return {
                "success": True,
                "pdf_url": f"/latex/pdf/{job_name}.pdf",
                "log_url": f"/latex/log/{job_name}.log",
                "tex_path": str(tex_path),
            }
        else:
            log_content = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
            return {
                "success": False,
                "error": "PDF compilation failed",
                "log": log_content[-3000:],  # last 3KB
                "stdout": result1.stdout[-1000:] + result2.stdout[-1000:],
                "stderr": result1.stderr[-1000:] + result2.stderr[-1000:],
            }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "LaTeX compilation timeout (120s)"}
    except Exception as e:
        logger.error(f"LaTeX compilation error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def list_tex_files() -> List[Dict[str, Any]]:
    """List all available .tex files and PDFs in the workspace"""
    files = []
    for f in COMPILED_DIR.glob("*.tex"):
        files.append({
            "name": f.name,
            "path": str(f),
            "modified": f.stat().st_mtime,
        })
    for f in COMPILED_DIR.glob("*.pdf"):
        files.append({
            "name": f.name,
            "path": str(f),
            "modified": f.stat().st_mtime,
        })
    return sorted(files, key=lambda x: x["modified"], reverse=True)


def read_tex_file(filename: str) -> Optional[str]:
    """Read a .tex file content"""
    safe_name = Path(filename).name  # prevent path traversal
    path = COMPILED_DIR / safe_name
    if path.exists() and path.suffix == ".tex":
        return path.read_text(encoding="utf-8")
    return None


def save_tex_file(filename: str, content: str) -> Dict[str, Any]:
    """Save a .tex file"""
    safe_name = Path(filename).name
    if not safe_name.endswith(".tex"):
        safe_name += ".tex"
    path = COMPILED_DIR / safe_name
    path.write_text(content, encoding="utf-8")
    return {"success": True, "path": str(path)}
