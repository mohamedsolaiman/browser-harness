"""Xiaomi MiMo TTS client — chat completions-based text-to-speech API.

API documentation: https://platform.xiaomimimo.com/docs/en-US/usage-guide/speech-synthesis

IMPORTANT: MiMo TTS uses the CHAT COMPLETIONS endpoint, NOT /audio/speech.
- Endpoint: POST https://api.xiaomimimo.com/v1/chat/completions
- Model: mimo-v2-tts
- The text goes in messages (user/assistant role)
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
import urllib.request
import urllib.error
from pathlib import Path

# Configuration
MIMO_API_KEY = os.environ.get("MIMO_API_KEY", "")
MIMO_BASE_URL = os.environ.get("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
MIMO_TTS_MODEL = os.environ.get("MIMO_TTS_MODEL", "mimo-v2-tts")

# Output directory
TTS_OUTPUT_DIR = Path(os.environ.get("BH_VIDEO_DIR", Path.home() / "browser-harness-videos"))
TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class MimoTTS:
    """Client for the Xiaomi MiMo TTS API (chat completions-based).

    The MiMo TTS API uses the chat completions endpoint with an "audio" parameter.
    Text to speak goes in the messages array, and the audio comes back as base64
    in the response message's audio.data field.

    Auth: api-key header OR Authorization: Bearer header
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

        url = f"{self.base_url}/chat/completions"

        # Build the message content with optional style tag
        content = text
        if style:
            content = f"<style>{style}</style>{text}"
        if speed < 0.8:
            content = f"<style>Slow down</style>{content}"
        elif speed > 1.2:
            content = f"<style>Speed up</style>{content}"

        body = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": content},
            ],
            "audio": {
                "format": response_format,
                "voice": voice,
            },
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            ext = "wav" if response_format in ("wav", "pcm16") else response_format
            output_path = str(TTS_OUTPUT_DIR / f"tts_{timestamp}.{ext}")

        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                data = json.loads(response.read())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else "unknown"
            raise RuntimeError(
                f"MiMo TTS API error (HTTP {e.code}): {error_body}. "
                f"Check your MIMO_API_KEY and MIMO_BASE_URL settings. "
                f"URL: {url}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot reach MiMo API at {url}: {e.reason}. "
                f"Check your MIMO_BASE_URL setting. Current: {self.base_url}"
            ) from e

        # Extract audio from the response
        try:
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError(f"No choices in TTS response. Full response: {json.dumps(data)[:500]}")

            message = choices[0].get("message", {})
            audio_data = message.get("audio", {}).get("data")

            if not audio_data:
                # Check if there's text content (might be an error)
                text_content = message.get("content", "")
                raise RuntimeError(
                    f"No audio data in TTS response. Message content: {text_content[:200]}. "
                    f"Full response: {json.dumps(data)[:500]}"
                )

            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_data)

            if len(audio_bytes) < 100:
                raise RuntimeError(f"Suspiciously small audio response ({len(audio_bytes)} bytes)")

        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected TTS response format: {e}. Response: {json.dumps(data)[:500]}") from e

        # Handle PCM16 format — needs conversion to WAV
        if response_format == "pcm16":
            output_path = self._pcm16_to_wav(audio_bytes, output_path)
        else:
            with open(output_path, "wb") as f:
                f.write(audio_bytes)

        return output_path

    def _pcm16_to_wav(self, pcm_data, output_path):
        """Convert raw PCM16 data to a WAV file using ffmpeg."""
        # Write raw PCM first
        raw_path = output_path.rsplit(".", 1)[0] + ".pcm"
        with open(raw_path, "wb") as f:
            f.write(pcm_data)

        # Convert to WAV with ffmpeg (PCM16LE, 24kHz, mono — MiMo TTS spec)
        wav_path = output_path.rsplit(".", 1)[0] + ".wav"
        cmd = [
            "ffmpeg", "-y",
            "-f", "s16le", "-ar", "24000", "-ac", "1",
            "-i", raw_path,
            wav_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            # If ffmpeg fails, save raw PCM and hope for the best
            with open(output_path, "wb") as f:
                f.write(pcm_data)
        else:
            output_path = wav_path

        # Cleanup raw file
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
            chunk_path = tempfile.mktemp(suffix=f".wav")
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
    """Generate speech from text using MiMo TTS.

    Args:
        text: Text to convert to speech.
        voice: Voice ID (mimo_default, default_zh, default_en).
        output_path: Output file path. Auto-generated if None.
        speed: Speech speed hint.
        style: Optional style string (Happy, Whisper, Angry, etc.).

    Returns:
        Path to the saved audio file.
    """
    client = MimoTTS()
    return client.generate(text, voice=voice, output_path=output_path, speed=speed, style=style)


def list_voices():
    """List available TTS voices."""
    return MimoTTS.VOICES
