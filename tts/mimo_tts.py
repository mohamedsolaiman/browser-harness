"""Xiaomi MiMo TTS client — chat completions-based text-to-speech API.

API documentation: https://platform.xiaomimimo.com/docs/en-US/usage-guide/speech-synthesis

IMPORTANT: MiMo TTS uses the CHAT COMPLETIONS endpoint, NOT /audio/speech.
- Endpoint: POST https://api.xiaomimimo.com/v1/chat/completions
- Model: mimo-v2-tts (or mimo-v2.5-tts)
- The text to speak goes in the ASSISTANT message
- A user message is required (can be a simple instruction)
- Audio config goes in the "audio" parameter: {"format": "wav", "voice": "mimo_default"}
- Audio comes back as base64 in response.choices[0].message.audio.data

Available voices:
  - mimo_default (MiMo-Default)
  - default_zh (MiMo-Chinese Female)
  - default_en (MiMo-English Female)

Available formats: wav, pcm16 (for streaming)

Reads MIMO_API_KEY from environment variables — never hard-coded.
"""

import base64
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

# Configuration
MIMO_API_KEY = os.environ.get("MIMO_API_KEY", "")
MIMO_BASE_URL = os.environ.get("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
MIMO_TTS_MODEL = os.environ.get("MIMO_TTS_MODEL", "mimo-v2-tts")

# Fallback URLs to try if primary fails
FALLBACK_BASE_URLS = [
    "https://api.xiaomimimo.com/v1",
    "https://api.mimo-v2.com/v1",
]

# Output directory
TTS_OUTPUT_DIR = Path(os.environ.get("BH_VIDEO_DIR", Path.home() / "browser-harness-videos"))
TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _tts_api_request(base_url, api_key, payload, timeout=60):
    """Make a TTS API request using the best available HTTP library.

    Tries: openai SDK → requests → urllib (in that order)
    Returns the parsed JSON response dict.
    """
    # Method 1: Try requests library (best for TTS since openai SDK may not handle audio response)
    try:
        import requests as req_lib
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        resp = req_lib.post(url, json=payload, headers=headers, timeout=timeout)
        if resp.status_code >= 400:
            raise RuntimeError(f"TTS API error (HTTP {resp.status_code}): {resp.text[:300]}")
        return resp.json()
    except ImportError:
        pass
    except Exception as e:
        print(f"  [tts] requests failed ({e}), trying urllib...")

    # Method 2: Fallback to urllib
    import urllib.request
    import urllib.error
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else "unknown"
        raise RuntimeError(
            f"MiMo TTS API error (HTTP {e.code}): {error_body}. "
            f"Check MIMO_API_KEY and MIMO_BASE_URL. URL: {url}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach MiMo API at {url}: {e.reason}. "
            f"Check MIMO_BASE_URL. Current: {base_url}"
        ) from e


class MimoTTS:
    """Client for the Xiaomi MiMo TTS API (chat completions-based).

    The MiMo TTS API uses the chat completions endpoint with an "audio" parameter.
    Text to speak goes in the ASSISTANT message, and a USER message is required.
    Audio comes back as base64 in the response message's audio.data field.

    Auth: Authorization: Bearer header
    Endpoint: POST {base_url}/chat/completions
    Model: mimo-v2-tts
    """

    # Available voices
    VOICES = {
        "mimo_default": "MiMo-Default",
        "default_zh": "MiMo-Chinese Female",
        "default_en": "MiMo-English Female",
    }

    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = api_key or MIMO_API_KEY
        self.base_url = (base_url or MIMO_BASE_URL).rstrip("/")
        self.model = model or MIMO_TTS_MODEL

        if not self.api_key:
            raise ValueError(
                "MIMO_API_KEY not set. Set it in .env or as an environment variable. "
                "Never commit API keys to the repository."
            )

    def generate(self, text, voice="mimo_default", output_path=None, response_format="wav",
                 speed=1.0, style=None):
        """Generate speech from text using MiMo TTS via chat completions.

        Args:
            text: Text to convert to speech.
            voice: Voice to use (mimo_default, default_zh, default_en).
            output_path: Path for the output audio file. Auto-generated if None.
            response_format: Audio format ("wav", "pcm16").
            speed: Speech speed hint (used as style if <1 or >1).
            style: Optional style string (e.g. "Happy", "Whisper", "Angry").

        Returns:
            Path to the saved audio file.
        """
        if len(text) > 4096:
            return self._generate_long_text(text, voice, output_path, response_format, speed, style)

        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            ext = "wav" if response_format in ("wav", "pcm16") else response_format
            output_path = str(TTS_OUTPUT_DIR / f"tts_{timestamp}.{ext}")

        # Build the message content with optional style tag
        # The text to speak goes in the ASSISTANT message
        assistant_content = text
        if style:
            assistant_content = f"<style>{style}</style>{text}"
        if speed < 0.8:
            assistant_content = f"<style>Slow down</style>{assistant_content}"
        elif speed > 1.2:
            assistant_content = f"<style>Speed up</style>{assistant_content}"

        body = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": "Please read the following text aloud."},
                {"role": "assistant", "content": assistant_content},
            ],
            "audio": {
                "format": response_format,
                "voice": voice,
            },
        }

        # Try each URL with fallback
        urls_to_try = [self.base_url] + [u for u in FALLBACK_BASE_URLS if u.rstrip("/") != self.base_url]
        last_error = None

        for url in urls_to_try:
            try:
                data = _tts_api_request(url, self.api_key, body, timeout=60)
                # Extract audio from the response
                audio_bytes = self._extract_audio(data)
                print(f"  [tts] TTS succeeded using {url}")

                # Handle PCM16 format — needs conversion to WAV
                if response_format == "pcm16":
                    output_path = self._pcm16_to_wav(audio_bytes, output_path)
                else:
                    with open(output_path, "wb") as f:
                        f.write(audio_bytes)

                return output_path
            except Exception as e:
                last_error = e
                print(f"  [tts] Failed with {url}: {e}")
                continue

        raise RuntimeError(
            f"All TTS API URLs failed. Last error: {last_error}. "
            f"Check MIMO_API_KEY and MIMO_BASE_URL."
        )

    def _extract_audio(self, data):
        """Extract audio bytes from the API response."""
        try:
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError(f"No choices in TTS response. Full response: {json.dumps(data)[:500]}")

            message = choices[0].get("message", {})
            audio_data = message.get("audio", {}).get("data")

            if not audio_data:
                text_content = message.get("content", "")
                raise RuntimeError(
                    f"No audio data in TTS response. Message content: {text_content[:200]}. "
                    f"Full response: {json.dumps(data)[:500]}"
                )

            audio_bytes = base64.b64decode(audio_data)

            if len(audio_bytes) < 100:
                raise RuntimeError(f"Suspiciously small audio response ({len(audio_bytes)} bytes)")

            return audio_bytes

        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected TTS response format: {e}. Response: {json.dumps(data)[:500]}") from e

    def _pcm16_to_wav(self, pcm_data, output_path):
        """Convert raw PCM16 data to a WAV file using ffmpeg."""
        raw_path = output_path.rsplit(".", 1)[0] + ".pcm"
        with open(raw_path, "wb") as f:
            f.write(pcm_data)

        wav_path = output_path.rsplit(".", 1)[0] + ".wav"
        cmd = [
            "ffmpeg", "-y",
            "-f", "s16le", "-ar", "24000", "-ac", "1",
            "-i", raw_path,
            wav_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            with open(output_path, "wb") as f:
                f.write(pcm_data)
        else:
            output_path = wav_path

        try:
            os.unlink(raw_path)
        except Exception:
            pass

        return output_path

    def _generate_long_text(self, text, voice, output_path, response_format, speed, style):
        """Generate speech for text longer than 4096 characters by splitting into chunks."""
        chunks = self._split_text(text, max_length=3800)
        chunk_paths = []

        for i, chunk in enumerate(chunks):
            chunk_path = tempfile.mktemp(suffix=".wav")
            path = self.generate(
                chunk, voice=voice, output_path=chunk_path,
                response_format=response_format, speed=speed, style=style,
            )
            chunk_paths.append(path)

        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = str(TTS_OUTPUT_DIR / f"tts_{timestamp}.wav")

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
        """List available TTS voices."""
        return [
            {"id": "mimo_default", "name": "MiMo-Default"},
            {"id": "default_zh", "name": "MiMo-Chinese Female"},
            {"id": "default_en", "name": "MiMo-English Female"},
        ]


# --- Module-level convenience ---

def generate_speech(text, voice="mimo_default", output_path=None, speed=1.0, style=None):
    """Generate speech from text using MiMo TTS."""
    client = MimoTTS()
    return client.generate(text, voice=voice, output_path=output_path, speed=speed, style=style)


def list_voices():
    """List available TTS voices."""
    return MimoTTS.VOICES
