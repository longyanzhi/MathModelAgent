# MathModelAgent — Environment Configuration Guide

This guide explains how to configure the `.env` file for **MathModelAgent**. The project uses [python-dotenv](https://pypi.org/project/python-dotenv/) to load environment variables from a `.env` file located in the project root.

---

## 1. Create the `.env` file

In the project root directory (the same folder that contains `app.py` and `config.py`), create a file named `.env`:

```bash
# On Windows PowerShell
New-Item -Path . -Name ".env" -ItemType File
```

> `.env` is automatically loaded by `python-dotenv` via `load_dotenv()` in `config.py`. You don't need to change anything else.

---

## 2. Required Variables

These **must** be set, otherwise the app will fail to start (validation error at boot).

```env
# OpenAI-compatible LLM API key (REQUIRED)
OPENAI_API_KEY=sk-your-real-api-key-here

# Flask secret key (REQUIRED for production — change from the default!)
SECRET_KEY=replace-with-a-long-random-string
```

- `OPENAI_API_KEY` — the main LLM key used by agents. The app will refuse to start if it is empty.
- `SECRET_KEY` — Flask session key. The default value is unsafe; replace it before deploying.

---

## 3. LLM / API Settings

```env
# OpenAI-compatible API endpoint
OPENAI_BASE_URL=https://www.dmxapi.cn/v1

# Request timeouts (seconds)
API_TIMEOUT=600
API_CONNECT_TIMEOUT=30
API_READ_TIMEOUT=600

# Retry policy
API_MAX_RETRIES=8
API_RETRY_DELAY=2.0
```

`OPENAI_BASE_URL` is the OpenAI-compatible gateway (default points to dmxapi). Point this to any compatible endpoint you use.

---

## 4. Hugging Face (Vision / Image Captioning)

Used by the **vision handler** to recognize uploaded images.

```env
# Hugging Face endpoint (use the mirror in mainland China)
HF_ENDPOINT=https://hf-mirror.com

# Your HF token (get one at https://huggingface.co/settings/tokens)
HF_API_KEY=hf_your_token_here

# Optional: dedicated inference endpoint URL
HF_INFERENCE_ENDPOINT=

# Default vision model
HF_VISION_MODEL=Salesforce/blip-image-captioning-large
```

---

## 5. Server Settings

```env
HOST=0.0.0.0
PORT=5000

DEBUG=True
# Disable Werkzeug auto-reloader to avoid [WinError 10038] on Windows
WERKZEUG_RELOAD=false
```

- `DEBUG` — set to `False` in production.
- `WERKZEUG_RELOAD` — leave `false` on Windows unless you know what you're doing.

---

## 6. Meeting Settings

```env
# Bounds on the number of meeting turns (validated client-side)
MIN_TURNS=1
MAX_TURNS=20
DEFAULT_TURNS=5

# How many meetings can run at the same time
MAX_CONCURRENT_MEETINGS=10
```

---

## 7. File Upload Limits

```env
# Per-file size cap (default 100 MB)
MAX_FILE_SIZE=104857600

# Max number of files per upload
MAX_FILES=6

# Comma-separated pre-meeting context file paths
MEETING_CONTEXT_FILES=
MEETING_CONTEXT_BASE=
```

---

## 8. Logging

```env
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
LOGS_DIR=logs
```

---

## 9. Complete Example `.env`

```env
# ===== Required =====
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
SECRET_KEY=please-change-me-to-a-random-secret

# ===== LLM =====
OPENAI_BASE_URL=https://www.dmxapi.cn/v1
API_TIMEOUT=600
API_CONNECT_TIMEOUT=30
API_READ_TIMEOUT=600
API_MAX_RETRIES=8
API_RETRY_DELAY=2.0

# ===== Hugging Face (vision) =====
HF_ENDPOINT=https://hf-mirror.com
HF_API_KEY=hf_xxxxxxxxxxxxxxxxxxxxxxxx
HF_INFERENCE_ENDPOINT=
HF_VISION_MODEL=Salesforce/blip-image-captioning-large

# ===== Server =====
HOST=0.0.0.0
PORT=5000
DEBUG=True
WERKZEUG_RELOAD=false

# ===== Meeting =====
MIN_TURNS=1
MAX_TURNS=20
DEFAULT_TURNS=5
MAX_CONCURRENT_MEETINGS=10

# ===== Uploads =====
MAX_FILE_SIZE=104857600
MAX_FILES=6
MEETING_CONTEXT_FILES=
MEETING_CONTEXT_BASE=

# ===== Logging =====
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
LOGS_DIR=logs
```

---

## 10. Validation

When the app starts, `Config.validate()` runs automatically and prints errors / warnings, e.g.:

```
==================================================
Configuration validation failed!
==================================================
OPENAI_API_KEY is not set
```

If you see this, go back to step 2 and fill in the required keys.

---

## 11. Tips

- **Never commit `.env` to git.** Add it to `.gitignore`:

  ```gitignore
  .env
  ```

- Values not provided fall back to the defaults defined in `config.py` — review that file for the full reference.
- Comments (lines starting with `#`) and blank lines in `.env` are ignored.
