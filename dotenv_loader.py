"""Minimal .env loader — no python-dotenv dependency required."""

import os
from pathlib import Path


def load_secrets():
    """Load environment variables from .env file and HF Space secrets."""
    # HF Spaces injects secrets as environment variables automatically.
    # This loader handles the local .env file case for development.
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

    # Set defaults for missing config
    os.environ.setdefault("MIMO_BASE_URL", "https://api.mymimo.ai")
    os.environ.setdefault("MIMO_TTS_MODEL", "mimo-tts-1")
    os.environ.setdefault("PLANNER_MODEL", "gpt-4o-mini")
    os.environ.setdefault("BH_VIDEO_DIR", "/tmp/content-studio/videos")
    os.environ.setdefault("BH_PLAN_DIR", "/tmp/content-studio/plans")
