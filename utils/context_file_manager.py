#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-meeting Document File Manager
Handles loading, reading, and formatting of pre-meeting document files
"""

import os
import json
import re
import mimetypes
from typing import List, Dict, Any, Optional, Tuple

from config import Config
from utils.logger import logger

# Try to import chardet, fallback to alternative if not available
try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False
    logger.warning("chardet not installed, will use basic encoding detection")


class ContextFileManager:
    """Manages loading and processing of pre-meeting document files"""
    
    CONTEXT_FILE_MAX_CHARS = 8000
    MAX_CONTEXT_FILES = 6
    
    def __init__(self, workspace_root: str, vector_memory=None):
        self.workspace_root = workspace_root
        self.vector_memory = vector_memory
        self.context_files_info: List[Dict[str, Any]] = []
        self.context_files_overview: str = ""
        self._context_files_announced: bool = False
        self._context_files_event_sent: bool = False
    
    def set_context_files(self, file_specs: List[Any]):
        """Allow external setting of pre-meeting document file list"""
        if not file_specs:
            self.context_files_info = []
            self.context_files_overview = ""
            self._context_files_announced = False
            self._context_files_event_sent = False
            return
        normalized = self._normalize_context_file_specs(file_specs)
        if not normalized:
            logger.warning("Pre-meeting document setup failed: no valid file paths provided.")
            self.context_files_info = []
            self.context_files_overview = ""
            self._context_files_announced = False
            self._context_files_event_sent = False
            return
        self._ingest_context_files(normalized)
    
    def load_default_context_files(self):
        """Load default pre-meeting document files"""
        raw = getattr(Config, "CONTEXT_FILES", "")
        if not raw:
            return
        try:
            if raw.strip().startswith('['):
                data = json.loads(raw)
            else:
                data = [item.strip() for item in re.split(r'[\n;,]+', raw) if item.strip()]
        except Exception as e:
            logger.warning(f"Failed to parse default pre-meeting documents: {e}")
            return
        normalized = self._normalize_context_file_specs(data)
        if normalized:
            self._ingest_context_files(normalized)
    
    def _normalize_context_file_specs(self, file_specs: List[Any]) -> List[Dict[str, Optional[str]]]:
        """Normalize file specifications"""
        normalized = []
        for item in file_specs:
            path = None
            alias = None
            if isinstance(item, dict):
                path = item.get('path') or item.get('file') or item.get('filepath')
                alias = item.get('alias') or item.get('name') or item.get('label')
            elif isinstance(item, str):
                path = item
            else:
                continue
            if not path:
                continue
            cleaned_path = path.strip()
            if not cleaned_path:
                continue
            normalized.append({
                'path': cleaned_path,
                'alias': alias.strip() if isinstance(alias, str) else None
            })
            if len(normalized) >= self.MAX_CONTEXT_FILES:
                break
        return normalized
    
    def _ingest_context_files(self, file_specs: List[Dict[str, Optional[str]]]):
        """Process file specification list"""
        if not file_specs:
            return
        base_candidates = []
        base_dir = getattr(Config, "CONTEXT_FILES_BASE_DIR", "") or ""
        if base_dir:
            base_candidates.append(os.path.abspath(base_dir))
        base_candidates.append(self.workspace_root)
        collected = []
        for spec in file_specs[:self.MAX_CONTEXT_FILES]:
            info = self._read_context_file(spec, base_candidates)
            collected.append(info)
        self.context_files_info = collected
        self.context_files_overview = self._format_context_files_overview(self.context_files_info)
        self._context_files_announced = False
        self._context_files_event_sent = False
        if self.context_files_overview and self.vector_memory:
            try:
                self.vector_memory.add_conversation(
                    "System Documents",
                    f"Pre-meeting Document Overview:\n{self.context_files_overview}",
                    0,
                    "context_file"
                )
            except Exception as e:
                logger.warning(f"Failed to record pre-meeting documents to vector memory: {e}")
    
    def _read_context_file(self, spec: Dict[str, Optional[str]], base_candidates: List[str]) -> Dict[str, Any]:
        """Read a single context file"""
        raw_path = spec.get('path')
        info = {
            "path": raw_path,
            "name": spec.get('alias') or (os.path.basename(raw_path) if raw_path else "") or "Unnamed File"
        }
        if not raw_path:
            info["error"] = "No path provided"
            return info
        
        resolved = self._resolve_context_path(raw_path, base_candidates)
        info["resolved_path"] = resolved
        
        # Detailed check for file existence
        if not resolved:
            info["error"] = "Path resolution failed"
            logger.warning(f"File path resolution failed: original path='{raw_path}'")
            return info
        
        if not os.path.exists(resolved):
            # Provide more detailed error information
            error_msg = f"File does not exist: {resolved}"
            if os.path.isabs(raw_path):
                error_msg += f" (original absolute path: {raw_path})"
            else:
                error_msg += f" (original path: {raw_path}, tried base directories: {base_candidates})"
            info["error"] = error_msg
            logger.warning(f"File does not exist: original path='{raw_path}', resolved path='{resolved}'")
            
            # Check if it's a path format issue (Windows path)
            if os.name == 'nt':  # Windows system
                # Try different path formats
                cleaned_raw = raw_path.strip().strip('"').strip("'")
                alt_paths = []
                if '/' in cleaned_raw:
                    alt_paths.append(cleaned_raw.replace('/', '\\'))
                if '\\' in cleaned_raw:
                    alt_paths.append(cleaned_raw.replace('\\', '/'))
                
                for alt_path in alt_paths:
                    if alt_path != cleaned_raw:
                        alt_resolved = os.path.abspath(alt_path) if os.path.isabs(alt_path) else alt_path
                        if os.path.exists(alt_resolved):
                            logger.info(f"Found alternative path format: '{alt_path}' -> '{alt_resolved}' exists")
                            info["suggestion"] = f"Try using path: {alt_resolved}"
                            break
            return info
        if os.path.isdir(resolved):
            info["error"] = "Path points to directory"
            return info
        
        try:
            info["size"] = os.path.getsize(resolved)
            
            # Detect file type
            file_ext = os.path.splitext(resolved)[1].lower()
            info["file_type"] = file_ext or "Unknown"
            
            # Check if binary file
            if self._is_binary_file(resolved, file_ext):
                info["error"] = f"Unsupported file type ({file_ext or 'binary file'}), only text files are supported"
                return info
            
            # Try multiple encodings to read file
            content, encoding = self._read_text_file_with_encoding(resolved)
            if content is None:
                info["error"] = "Unable to recognize file encoding, please ensure file is UTF-8, GBK, or GB2312 encoded text"
                return info
            
            info["encoding"] = encoding
            info["summary"] = self._summarize_file_content(content)
            info["preview"] = content[:400].strip()
            info["content"] = content  # Store full content for agents to use
            
        except Exception as e:
            info["error"] = f"Read failed: {e}"
            logger.error(f"Failed to read file {resolved}: {e}")
        
        return info
    
    def _is_binary_file(self, file_path: str, file_ext: str) -> bool:
        """Detect if file is binary"""
        # Common binary file extensions
        binary_extensions = {
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.zip', '.rar', '.7z', '.tar', '.gz',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico', '.svg',
            '.mp3', '.mp4', '.avi', '.mov', '.wmv',
            '.exe', '.dll', '.so', '.dylib',
            '.bin', '.dat', '.db', '.sqlite'
        }
        
        if file_ext in binary_extensions:
            return True
        
        # Detect by file content (read first 512 bytes)
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(512)
                # Check for NULL bytes (characteristic of binary files)
                if b'\x00' in chunk:
                    return True
                # Check if it's text (most bytes are printable ASCII or common UTF-8)
                text_ratio = sum(1 for b in chunk if 32 <= b < 127 or b in (9, 10, 13)) / len(chunk) if chunk else 0
                if text_ratio < 0.7:  # If less than 70% printable characters, may be binary
                    return True
        except Exception:
            pass
        
        return False
    
    def _read_text_file_with_encoding(self, file_path: str) -> Tuple[Optional[str], Optional[str]]:
        """Try multiple encodings to read text file"""
        # Encoding attempt order: UTF-8 -> GBK -> GB2312 -> Auto detect
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
        
        # First try chardet auto-detection (if available)
        if HAS_CHARDET:
            try:
                with open(file_path, 'rb') as f:
                    raw_data = f.read(min(10000, self.CONTEXT_FILE_MAX_CHARS * 2))  # Read more bytes for detection
                    if raw_data:
                        detected = chardet.detect(raw_data)
                        if detected and detected.get('confidence', 0) > 0.7:
                            detected_encoding = detected['encoding']
                            if detected_encoding and detected_encoding.lower() not in [e.lower() for e in encodings]:
                                encodings.insert(0, detected_encoding)
            except Exception:
                pass
        
        # Try various encodings
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='strict') as f:
                    content = f.read(self.CONTEXT_FILE_MAX_CHARS)
                    # Verify content is reasonable (contains some printable characters)
                    if content and any(c.isprintable() or c in '\n\r\t' for c in content[:100]):
                        return content, encoding
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"Failed to read file with encoding {encoding}: {e}")
                continue
        
        # If all encodings fail, try with errors='ignore'
        for encoding in encodings[:2]:  # Only try UTF-8 and GBK
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    content = f.read(self.CONTEXT_FILE_MAX_CHARS)
                    if content.strip():  # At least some content
                        return content, encoding
            except Exception:
                continue
        
        return None, None
    
    def _resolve_context_path(self, raw_path: str, base_candidates: List[str]) -> str:
        """Resolve file path"""
        # Clean path: remove quotes, leading/trailing spaces
        cleaned_path = raw_path.strip().strip('"').strip("'")
        
        # Log original input
        logger.debug(f"Path resolution started: raw_path='{raw_path}', cleaned='{cleaned_path}'")
        
        # Handle absolute path
        if os.path.isabs(cleaned_path):
            resolved = os.path.abspath(cleaned_path)
            # Normalize path (unify to system path separator)
            resolved = os.path.normpath(resolved)
            logger.info(f"Absolute path resolved: '{raw_path}' -> '{resolved}' (exists: {os.path.exists(resolved)})")
            return resolved
        
        # Handle relative path: try multiple base directories
        for base in base_candidates:
            if not base:
                continue
            candidate = os.path.abspath(os.path.join(base, cleaned_path))
            candidate = os.path.normpath(candidate)  # Normalize path
            if os.path.exists(candidate):
                logger.info(f"Relative path resolved: '{raw_path}' (base: '{base}') -> '{candidate}'")
                return candidate
            else:
                logger.debug(f"Relative path attempt failed: '{candidate}' (base: '{base}')")
        
        # If not found in any base directory, return last candidate (for error reporting)
        final_candidate = os.path.abspath(os.path.join(self.workspace_root, cleaned_path))
        final_candidate = os.path.normpath(final_candidate)  # Normalize path
        logger.warning(f"Path resolution failed: '{raw_path}' -> '{final_candidate}' (file not found)")
        return final_candidate
    
    def _summarize_file_content(self, content: str) -> str:
        """Summarize file content"""
        if not content:
            return "(File is empty or content cannot be read)"
        stripped = content.strip()
        if not stripped:
            return "(File contains only whitespace)"
        if stripped[0] in ['{', '[']:
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict):
                    keys = list(parsed.keys())[:6]
                    return f"JSON object, containing fields: {', '.join(keys)}"
                if isinstance(parsed, list):
                    return f"JSON array, {len(parsed)} items, example: {str(parsed[:2])[:120]}..."
            except Exception:
                pass
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            return "(No available text content)"
        summary = " / ".join(lines[:3])
        if len(summary) > 360:
            summary = summary[:360].rstrip() + "..."
        return summary
    
    def _format_context_files_overview(self, infos: List[Dict[str, Any]]) -> str:
        """Format pre-meeting document overview"""
        if not infos:
            return ""
        lines = ["## Pre-meeting Documents"]
        for idx, info in enumerate(infos, 1):
            name = info.get("name") or f"File {idx}"
            path = info.get("path") or ""
            if info.get("error"):
                error_msg = info['error']
                resolved_path = info.get("resolved_path", "")
                if resolved_path and resolved_path != path:
                    error_msg += f" (attempted access: {resolved_path})"
                lines.append(f"{idx}. {name} ({path}) - Read failed: {error_msg}")
                continue
            size = info.get("size")
            size_label = ""
            if isinstance(size, (int, float)) and size >= 0:
                size_label = f"{size/1024:.1f}KB"
            file_type = info.get("file_type", "")
            encoding = info.get("encoding", "")
            summary = info.get("summary") or "Failed to generate summary"
            detail = f"{idx}. {name} ({path}"
            if size_label:
                detail += f", {size_label}"
            if file_type and file_type != "Unknown":
                detail += f", type: {file_type}"
            if encoding:
                detail += f", encoding: {encoding}"
            detail += ")"
            lines.append(detail)
            lines.append(f"   Summary: {summary}")
        return "\n".join(lines)
    
    def format_context_files_content(self, infos: List[Dict[str, Any]] = None) -> str:
        """Format full content of pre-meeting document files for agents to use directly"""
        if infos is None:
            infos = self.context_files_info
        if not infos:
            return ""
        lines = ["## Pre-meeting Document File Content"]
        for idx, info in enumerate(infos, 1):
            name = info.get("name") or f"File {idx}"
            path = info.get("path") or ""
            if info.get("error"):
                lines.append(f"\n### {idx}. {name} ({path})")
                lines.append(f"**Read failed**: {info['error']}\n")
                continue
            content = info.get("content", "")
            if not content:
                lines.append(f"\n### {idx}. {name} ({path})")
                lines.append("**File content is empty**\n")
                continue
            lines.append(f"\n### {idx}. {name} ({path})")
            lines.append("```")
            lines.append(content)
            lines.append("```\n")
        return "\n".join(lines)
    
    def emit_context_files_event(self, message_callback=None):
        """Send pre-meeting document file event"""
        if self._context_files_event_sent or not self.context_files_info or not message_callback:
            return
        items = []
        for info in self.context_files_info:
            items.append({
                "title": info.get("name"),
                "summary": info.get("summary") or info.get("error") or "",
                "published": info.get("size") and f"{info['size']/1024:.1f}KB",
                "url": None,
                "raw": {"path": info.get("path")}
            })
        message_callback({
            'type': 'external_resource',
            'source': 'context_files',
            'items': items
        })
        self._context_files_event_sent = True
