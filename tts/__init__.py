"""Text-to-speech integration for Content Automation Studio.

Supports Mimo TTS API (OpenAI-compatible endpoint).
All API keys are read from environment variables — never hard-coded.
Only uses https://api.xiaomimimo.com/v1 — no broken fallback URLs.
"""
from .mimo_tts import MimoTTS, generate_speech, list_voices
