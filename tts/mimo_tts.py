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

Only uses https://api.xiaomimimo.com/v1 — no broken fallback URLs.
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

# Output directory
TTS_OUTPUT_DIR = Path(os.environ.get("CS_AUDIO_DIR", "/tmp/content-studio/audio"))
TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _tts_api_request(base_url, api_key, payload, timeout=60):
    """Make a TTS API request using the best available HTTP library.

    Tries: requests → urllib (in that order)
    Returns the parsed JSON response dict.

    Args:
        base_url: API base URL.
        api_key: API key.
        payload: Request payload dict.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response dict.

    Raises:
        RuntimeError: If all methods fail.
    """
    last_error = None

    # Method 1: Try requests library (most reliable in containers)
    try:
        import requests as req_lib
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        print(f"  [tts] Trying requests to {url}...")
        resp = req_lib.post(url, json=payload, headers=headers, timeout=timeout)
        if resp.status_code >= 400:
            raise RuntimeError(f"TTS API error (HTTP {resp.status_code}): {resp.text[:300]}")
        print(f"  [tts] requests succeeded (HTTP {resp.status_code})")
        return resp.json()
    except ImportError:
        print("  [tts] requests library not available, trying urllib...")
    except Exception as e:
        last_error = e
        print(f"  [tts] requests failed ({e}), trying urllib...")

    # Method 2: Fallback to urllib
    try:
        import urllib.request
        import urllib.error
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read())
            print(f"  [tts] urllib succeeded")
            return data
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
    except Exception as e:
        last_error = e

    raise RuntimeError(
        f"All TTS API request methods failed. Last error: {last_error}. "
        f"Check MIMO_API_KEY and network connectivity."
    )


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

        Raises:
            RuntimeError: If TTS generation fails.
        """
        # Handle long text by splitting
        if len(text) > 4096:
            return self._generate_long_text(text, voice, output_path, response_format, speed, style)

        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            ext = "wav" if response_format in ("wav", "pcm16") else response_format
            output_path = str(TTS_OUTPUT_DIR / f"tts_{timestamp}.{ext}")

        # Build the message content with optional style tag
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

        try:
            data = _tts_api_request(self.base_url, self.api_key, body, timeout=60)
            # Extract audio from the response
            audio_bytes = self._extract_audio(data)

            # Handle PCM16 format — needs conversion to WAV
            if response_format == "pcm16":
                output_path = self._pcm16_to_wav(audio_bytes, output_path)
            else:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(audio_bytes)

            print(f"  [tts] Audio saved: {output_path} ({len(audio_bytes):,} bytes)")
            return output_path

        except Exception as e:
            raise RuntimeError(
                f"TTS generation failed: {e}. "
                f"Check MIMO_API_KEY and MIMO_BASE_URL settings."
            ) from e

    def _extract_audio(self, data):
        """Extract audio bytes from the API response.

        Args:
            data: Parsed JSON response dict.

        Returns:
            Audio bytes.

        Raises:
            RuntimeError: If no audio data found in response.
        """
        try:
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError(
                    f"No choices in TTS response. Full response: {json.dumps(data)[:500]}"
                )

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
            raise RuntimeError(
                f"Unexpected TTS response format: {e}. "
                f"Response: {json.dumps(data)[:500]}"
            ) from e

    def _pcm16_to_wav(self, pcm_data, output_path):
        """Convert raw PCM16 data to a WAV file using ffmpeg.

        Args:
            pcm_data: Raw PCM16 audio bytes.
            output_path: Target output path.

        Returns:
            Path to the WAV file.
        """
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
        """Generate speech for text longer than 4096 characters by splitting into chunks.

        Args:
            text: Full text to convert.
            voice: TTS voice.
            output_path: Output file path.
            response_format: Audio format.
            speed: Speech speed.
            style: Optional style.

        Returns:
            Path to the concatenated audio file.
        """
        chunks = self._split_text(text, max_length=3800)
        chunk_paths = []

        for i, chunk in enumerate(chunks):
            try:
                chunk_path = tempfile.mktemp(suffix=".wav")
                path = self.generate(
                    chunk, voice=voice, output_path=chunk_path,
                    response_format=response_format, speed=speed, style=style,
                )
                chunk_paths.append(path)
            except Exception as e:
                print(f"  [tts] WARNING: Failed to generate chunk {i + 1}/{len(chunks)}: {e}")
                continue

        if not chunk_paths:
            raise RuntimeError("All TTS chunks failed")

        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = str(TTS_OUTPUT_DIR / f"tts_{timestamp}.wav")

        return self._concatenate_audio(chunk_paths, output_path)

    def _split_text(self, text, max_length=3800):
        """Split text into chunks at sentence boundaries.

        Args:
            text: Text to split.
            max_length: Maximum chunk length.

        Returns:
            List of text chunks.
        """
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
        """Concatenate multiple audio files using ffmpeg.

        Args:
            paths: List of audio file paths.
            output_path: Output path.

        Returns:
            Path to the concatenated audio file.
        """
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
        """List available TTS voices.

        Returns:
            List of voice dicts with id and name.
        """
        return [
            {"id": "mimo_default", "name": "MiMo-Default"},
            {"id": "default_zh", "name": "MiMo-Chinese Female"},
            {"id": "default_en", "name": "MiMo-English Female"},
        ]


# --- Module-level convenience ---

def generate_speech(text, voice="mimo_default", output_path=None, speed=1.0, style=None):
    """Generate speech from text using MiMo TTS.

    Args:
        text: Text to convert to speech.
        voice: Voice to use.
        output_path: Output file path.
        speed: Speech speed.
        style: Optional style.

    Returns:
        Path to the saved audio file.
    """
    client = MimoTTS()
    return client.generate(text, voice=voice, output_path=output_path, speed=speed, style=style)


def list_voices():
    """List available TTS voices.

    Returns:
        Dict of voice names.
    """
    return MimoTTS.VOICES
