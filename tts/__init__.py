"""Text-to-speech integration for browser-harness videos.

Supports Mimo TTS API (OpenAI-compatible endpoint) and other TTS providers.
All API keys are read from environment variables — never hard-coded.
"""
from .mimo_tts import MimoTTS, generate_speech, list_voices
