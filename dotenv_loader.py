"""Minimal .env loader — no python-dotenv dependency required."""

import os
from pathlib import Path


def load_secrets():
    """Load environment variables from .env file and HF Space secrets."""
    env_paths = [
        Path(__file__).parent / ".env",
        Path.cwd() / ".env",
    ]
    for p in env_paths:
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    # Set correct defaults for Xiaomi MiMo API
    # Only https://api.xiaomimimo.com/v1 — no broken fallback URLs
    os.environ.setdefault("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
    os.environ.setdefault("MIMO_TTS_MODEL", "mimo-v2-tts")
    os.environ.setdefault("PLANNER_MODEL", "mimo-v2-flash")
    os.environ.setdefault("CS_VIDEO_DIR", "/tmp/content-studio/videos")
    os.environ.setdefault("CS_AUDIO_DIR", "/tmp/content-studio/audio")
    os.environ.setdefault("CS_IMAGE_DIR", "/tmp/content-studio/images")
    os.environ.setdefault("CS_PLAN_DIR", "/tmp/content-studio/plans")
    os.environ.setdefault("CS_OUTPUT_DIR", "/tmp/content-studio")
