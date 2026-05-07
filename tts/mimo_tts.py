"""Mimo TTS client — Xiaomi MiMo text-to-speech API.

API documentation: https://www.mimo-v2.com/docs/usage-guide/tts
OpenAI-compatible endpoint at https://api.mimo-v2.com/v1

Reads MIMO_API_KEY from environment variables (or .env file).
Never hard-code API keys.

Usage with browser-harness -c:
    browser-harness -c '
    from tts import generate_speech
    path = generate_speech("Welcome to my channel!", voice="alloy")
    print(f"Audio saved: {path}")
    '
"""

import json
import os
import subprocess
import tempfile
import time
import urllib.request
import urllib.error
from pathlib import Path

# Configuration — all from environment variables, never hard-coded
MIMO_API_KEY = os.environ.get("MIMO_API_KEY", "")
MIMO_BASE_URL = os.environ.get("MIMO_BASE_URL", "https://api.mimo-v2.com/v1")
MIMO_TTS_MODEL = os.environ.get("MIMO_TTS_MODEL", "mimo-v2-tts")

# Output directory
TTS_OUTPUT_DIR = Path(os.environ.get("BH_VIDEO_DIR", Path.home() / "browser-harness-videos"))
TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class MimoTTS:
    """Client for the Xiaomi MiMo TTS API (OpenAI-compatible endpoint).

    API docs: https://www.mimo-v2.com/docs/usage-guide/tts

    Endpoint: POST https://api.mimo-v2.com/v1/audio/speech
    Model: mimo-v2-tts
    Auth: api-key header or Authorization: Bearer

    All credentials come from environment variables:
      - MIMO_API_KEY: API key for the TTS service
      - MIMO_BASE_URL: Base URL (default: https://api.mimo-v2.com/v1)
      - MIMO_TTS_MODEL: Model name (default: mimo-v2-tts)
    """

    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = api_key or MIMO_API_KEY
        self.base_url = (base_url or MIMO_BASE_URL).rstrip("/")
        self.model = model or MIMO_TTS_MODEL

        if not self.api_key:
            raise ValueError(
                "MIMO_API_KEY not set. Set it in .env or as an environment variable. "
                "Never commit API keys to the repository."
            )

    def generate(self, text, voice="alloy", output_path=None, response_format="mp3",
                 speed=1.0):
        """Generate speech from text.

        Args:
            text: Text to convert to speech.
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer, etc.).
            output_path: Path for the output audio file. Auto-generated if None.
            response_format: Audio format ("mp3", "opus", "aac", "flac", "wav", "pcm").
            speed: Speech speed (0.25 to 4.0, default 1.0).

        Returns:
            Path to the saved audio file.
        """
        if len(text) > 4096:
            return self._generate_long_text(text, voice, output_path, response_format, speed)

        # Build URL — base_url already includes /v1
        url = f"{self.base_url}/audio/speech"

        body = {
            "model": self.model,
            "input": text,
            "voice": voice,
            "response_format": response_format,
            "speed": speed,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = str(TTS_OUTPUT_DIR / f"tts_{timestamp}.{response_format}")

        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                audio_data = response.read()
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else "unknown"
            raise RuntimeError(
                f"Mimo TTS API error (HTTP {e.code}): {error_body}. "
                f"Check your MIMO_API_KEY and MIMO_BASE_URL settings."
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot reach Mimo API at {url}: {e.reason}. "
                f"Check your internet connection and MIMO_BASE_URL setting. "
                f"Current base URL: {self.base_url}"
            ) from e

        if len(audio_data) < 100:
            raise RuntimeError(
                f"Mimo TTS returned suspiciously small response ({len(audio_data)} bytes). "
                f"The API key may be invalid or the service may be down."
            )

        with open(output_path, "wb") as f:
            f.write(audio_data)

        return output_path

    def _generate_long_text(self, text, voice, output_path, response_format, speed):
        """Generate speech for text longer than 4096 characters by splitting into chunks."""
        chunks = self._split_text(text, max_length=3800)
        chunk_paths = []

        for i, chunk in enumerate(chunks):
            chunk_path = tempfile.mktemp(suffix=f".{response_format}")
            path = self.generate(
                chunk, voice=voice, output_path=chunk_path,
                response_format=response_format, speed=speed,
            )
            chunk_paths.append(path)

        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = str(TTS_OUTPUT_DIR / f"tts_{timestamp}.{response_format}")

        return self._concatenate_audio(chunk_paths, output_path)

    def _split_text(self, text, max_length=3800):
        """Split text into chunks at sentence boundaries."""
        sentences = text.replace(". ", ".\n").replace("? ", "?\n").replace("! ", "!\n").split("\n")
        chunks = []
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) + 1 > max_length:
                if current:
                    chunks.append(current.strip())
                current = sentence
            else:
                current = (current + " " + sentence).strip()

        if current:
            chunks.append(current.strip())

        return chunks

    def _concatenate_audio(self, paths, output_path):
        """Concatenate multiple audio files using ffmpeg."""
        concat_file = tempfile.mktemp(suffix=".txt")
        with open(concat_file, "w") as f:
            for p in paths:
                f.write(f"file '{p}'\n")

        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_file, "-c", "copy", output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Audio concatenation failed: {result.stderr[-500:]}")

        for p in paths:
            try:
                os.unlink(p)
            except Exception:
                pass
        try:
            os.unlink(concat_file)
        except Exception:
            pass

        return output_path

    def list_voices(self):
        """List available voices. Returns the default Mimo TTS voice list."""
        return [
            {"id": "alloy", "name": "Alloy"},
            {"id": "echo", "name": "Echo"},
            {"id": "fable", "name": "Fable"},
            {"id": "onyx", "name": "Onyx"},
            {"id": "nova", "name": "Nova"},
            {"id": "shimmer", "name": "Shimmer"},
        ]


# --- Module-level convenience functions ---

def generate_speech(text, voice="alloy", output_path=None, speed=1.0):
    """Generate speech from text using Mimo TTS.

    Args:
        text: Text to convert to speech.
        voice: Voice ID.
        output_path: Output file path. Auto-generated if None.
        speed: Speech speed (0.25–4.0).

    Returns:
        Path to the saved audio file.
    """
    client = MimoTTS()
    return client.generate(text, voice=voice, output_path=output_path, speed=speed)


def list_voices():
    """List available TTS voices."""
    client = MimoTTS()
    return client.list_voices()
