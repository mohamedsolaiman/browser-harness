"""Video editor for compositing browser recordings with TTS audio, titles, and overlays.

Uses ffmpeg for all video operations — no moviepy dependency required.
This keeps the dependency footprint minimal and leverages the system ffmpeg.

Typical workflow:
    1. Record browser session with SessionRecorder → video.mp4
    2. Generate TTS audio with mimo_tts → audio.mp3
    3. Compose final video with VideoEditor → final.mp4
    4. Upload to YouTube/TikTok/X via domain skills
"""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

VIDEO_OUTPUT_DIR = Path(os.environ.get("BH_VIDEO_DIR", Path.home() / "browser-harness-videos"))
VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class VideoEditor:
    """Composes videos from browser recordings, TTS audio, titles, and overlays.

    All operations use ffmpeg subprocess calls. No Python video libraries needed.
    """

    def __init__(self):
        self._inputs = []
        self._title = None
        self._title_duration = 5
        self._watermark = None
        self._filters = []

    def add_video(self, path, start=0, duration=None):
        """Add a video clip to the composition.

        Args:
            path: Path to the video file (MP4, WebM, etc.).
            start: Start time in seconds for trimming.
            duration: Duration in seconds (None = to end).
        """
        self._inputs.append({"type": "video", "path": str(path), "start": start, "duration": duration})
        return self

    def add_audio(self, path, start=0, duration=None, fade_out=None):
        """Add an audio track to the composition.

        Args:
            path: Path to the audio file (MP3, WAV, AAC, etc.).
            start: Start time in seconds for trimming.
            duration: Duration in seconds (None = to end).
            fade_out: Fade out duration in seconds from the end.
        """
        self._inputs.append({
            "type": "audio", "path": str(path),
            "start": start, "duration": duration, "fade_out": fade_out
        })
        return self

    def set_title(self, text, duration=5, font_size=48, color="white",
                  bg_color="black@0.7", position="center"):
        """Add a title screen before the video.

        Args:
            text: Title text to display.
            duration: Duration of the title screen in seconds.
            font_size: Font size in pixels.
            color: Text color.
            bg_color: Background color (with optional alpha).
            position: Text position ("center", "top", "bottom").
        """
        self._title = {
            "text": text, "duration": duration, "font_size": font_size,
            "color": color, "bg_color": bg_color, "position": position
        }
        return self

    def set_watermark(self, text, font_size=24, color="white@0.5", position="bottom-right"):
        """Add a persistent watermark overlay.

        Args:
            text: Watermark text.
            font_size: Font size in pixels.
            color: Text color with alpha.
            position: Position ("top-left", "top-right", "bottom-left", "bottom-right").
        """
        self._watermark = {
            "text": text, "font_size": font_size, "color": color, "position": position
        }
        return self

    def add_subtitle_track(self, srt_path):
        """Add subtitles from an SRT file.

        Args:
            srt_path: Path to the SRT subtitle file.
        """
        self._inputs.append({"type": "subtitle", "path": str(srt_path)})
        return self

    def render(self, output_path=None, resolution="1080p", fps=30):
        """Render the final composed video.

        Args:
            output_path: Output file path. Auto-generated if None.
            resolution: Output resolution ("720p", "1080p", "4k").
            fps: Output frame rate.

        Returns:
            Path to the rendered video file.
        """
        if not self._inputs:
            raise RuntimeError("No inputs added. Use add_video() and add_audio() first.")

        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = str(VIDEO_OUTPUT_DIR / f"composed_{timestamp}.mp4")

        resolutions = {"720p": (1280, 720), "1080p": (1920, 1080), "4k": (3840, 2160)}
        width, height = resolutions.get(resolution, (1920, 1080))

        # Separate video and audio inputs
        video_inputs = [i for i in self._inputs if i["type"] == "video"]
        audio_inputs = [i for i in self._inputs if i["type"] == "audio"]
        subtitle_inputs = [i for i in self._inputs if i["type"] == "subtitle"]

        if not video_inputs:
            # If only audio, create a video from a black screen + audio
            return self._render_audio_only(audio_inputs, output_path, width, height, fps)

        # Build ffmpeg command
        cmd = ["ffmpeg", "-y"]

        # Add title screen if specified
        if self._title:
            title_path = self._render_title_frame(width, height)
            cmd.extend([
                "-loop", "1",
                "-t", str(self._title["duration"]),
                "-i", title_path,
            ])

        # Add video inputs
        for vi in video_inputs:
            cmd.extend(["-i", vi["path"]])
            # Note: start/duration trimming applied via filter

        # Add audio inputs
        for ai in audio_inputs:
            cmd.extend(["-i", ai["path"]])

        # Build filter complex
        filters = []
        video_streams = []

        # Title stream (if exists)
        title_offset = 0
        if self._title:
            filters.append(f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p[titlev]")
            video_streams.append("titlev")
            title_offset = 1

        # Process video streams
        for idx, vi in enumerate(video_inputs):
            stream_idx = idx + title_offset
            label = f"v{idx}"
            trim_parts = [f"[{stream_idx}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,format=yuv420p"]

            if vi["start"] > 0 or vi["duration"]:
                trim_parts.append(f"trim=start={vi['start']}" + (f":duration={vi['duration']}" if vi["duration"] else ""))
                trim_parts.append("setpts=PTS-STARTPTS")

            filters.append(",".join(trim_parts) + f"[{label}]")
            video_streams.append(label)

        # Concatenate video streams
        if len(video_streams) > 1:
            concat_inputs = "".join(f"[{s}]" for s in video_streams)
            n = len(video_streams)
            filters.append(f"{concat_inputs}concat=n={n}:v=1:a=0[outv]")
        else:
            filters.append(f"[{video_streams[0]}]copy[outv]")

        # Add watermark
        if self._watermark:
            pos = self._watermark["position"]
            positions = {
                "top-left": f"x=10:y=10",
                "top-right": f"x=w-tw-10:y=10",
                "bottom-left": f"x=10:y=h-th-10",
                "bottom-right": f"x=w-tw-10:y=h-th-10",
                "center": f"x=(w-tw)/2:y=(h-th)/2",
            }
            draw_pos = positions.get(pos, positions["bottom-right"])
            wm_text = self._watermark["text"].replace("'", "\\'")
            wm_color = self._watermark["color"]
            wm_size = self._watermark["font_size"]
            filters.append(
                f"[outv]drawtext=text='{wm_text}':fontsize={wm_size}:"
                f"fontcolor={wm_color}:{draw_pos}[finalv]"
            )
            final_video = "finalv"
        else:
            final_video = "outv"

        # Process audio
        audio_labels = []
        for idx, ai in enumerate(audio_inputs):
            stream_idx = len(video_inputs) + title_offset + idx
            label = f"a{idx}"
            audio_filter = f"[{stream_idx}:a]"

            trim_parts = []
            if ai["start"] > 0:
                trim_parts.append(f"atrim=start={ai['start']}")
                trim_parts.append("asetpts=PTS-STARTPTS")
            if ai["duration"]:
                trim_parts.append(f"atrim=duration={ai['duration']}")
                trim_parts.append("asetpts=PTS-STARTPTS")
            if ai.get("fade_out") and ai["fade_out"] > 0:
                trim_parts.append(f"afade=t=out:st={ai['duration'] - ai['fade_out']}:d={ai['fade_out']}")

            if trim_parts:
                filters.append(audio_filter + ",".join(trim_parts) + f"[{label}]")
            else:
                filters.append(audio_filter + f"acopy[{label}]")
            audio_labels.append(label)

        # Audio mixing
        if audio_labels:
            if len(audio_labels) > 1:
                mix_inputs = "".join(f"[{l}]" for l in audio_labels)
                filters.append(f"{mix_inputs}amix=inputs={len(audio_labels)}:duration=first[outa]")
                final_audio = "outa"
            else:
                final_audio = audio_labels[0]
        else:
            final_audio = None

        # Build final command
        filter_complex = ";".join(filters)
        cmd.extend(["-filter_complex", filter_complex])
        cmd.extend(["-map", f"[{final_video}]"])
        if final_audio:
            cmd.extend(["-map", f"[{final_audio}]"])

        cmd.extend([
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-r", str(fps),
            "-pix_fmt", "yuv420p",
        ])

        if final_audio:
            cmd.extend(["-c:a", "aac", "-b:a", "192k"])

        # Add subtitle if present
        for si in subtitle_inputs:
            cmd.extend(["-vf", f"subtitles={si['path']}"])

        cmd.append(output_path)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg render failed: {result.stderr[-1000:]}")

        return output_path

    def _render_title_frame(self, width, height):
        """Render a title frame as PNG using ffmpeg.

        Returns:
            Path to the rendered title frame PNG.
        """
        tmp = tempfile.mktemp(suffix=".png")
        title = self._title

        # Position mappings for drawtext
        positions = {
            "center": f"x=(w-text_w)/2:y=(h-text_h)/2",
            "top": f"x=(w-text_w)/2:y=h*0.1",
            "bottom": f"x=(w-text_w)/2:y=h*0.8",
        }
        draw_pos = positions.get(title["position"], positions["center"])
        text = title["text"].replace("'", "\\'")

        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c={title['bg_color']}:s={width}x{height}:d=1",
            "-frames:v", "1",
            "-vf", f"drawtext=text='{text}':fontsize={title['font_size']}:fontcolor={title['color']}:{draw_pos}",
            tmp
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            # Fallback: create a simple black frame
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:d=1",
                "-frames:v", "1", tmp
            ]
            subprocess.run(cmd, capture_output=True, text=True)

        return tmp

    def _render_audio_only(self, audio_inputs, output_path, width, height, fps):
        """Render a video with a black screen and audio track."""
        if not audio_inputs:
            raise RuntimeError("No audio inputs provided")

        cmd = ["ffmpeg", "-y"]

        # Black screen input
        cmd.extend([
            "-f", "lavfi",
            "-i", f"color=c=black:s={width}x{height}:r={fps}",
        ])

        # Audio inputs
        for ai in audio_inputs:
            cmd.extend(["-i", ai["path"]])

        # Map video
        cmd.extend(["-map", "0:v"])

        # Map first audio
        if audio_inputs:
            cmd.extend(["-map", "1:a"])

        # Shortest to match audio length
        cmd.extend([
            "-c:v", "libx264", "-preset", "medium", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            output_path
        ])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg audio-only render failed: {result.stderr[-500:]}")

        return output_path


def compose_video(video_path=None, audio_path=None, title=None, output_path=None,
                  resolution="1080p", fps=30, watermark=None):
    """Convenience function to compose a video from parts.

    Args:
        video_path: Path to the video recording (from SessionRecorder).
        audio_path: Path to TTS audio file.
        title: Title text to add as a title screen.
        output_path: Output path. Auto-generated if None.
        resolution: Output resolution.
        fps: Output frame rate.
        watermark: Watermark text.

    Returns:
        Path to the composed video file.
    """
    editor = VideoEditor()

    if video_path:
        editor.add_video(video_path)
    if audio_path:
        fade = None
        if video_path:
            # Fade out audio 2 seconds before end
            fade = 2
        editor.add_audio(audio_path, fade_out=fade)
    if title:
        editor.set_title(title)
    if watermark:
        editor.set_watermark(watermark)

    return editor.render(output_path=output_path, resolution=resolution, fps=fps)


def create_slideshow(frames_dir, audio_path=None, output_path=None, fps=2, duration_per_frame=5):
    """Create a slideshow video from a directory of PNG/JPG frames.

    Args:
        frames_dir: Directory containing frame images (sorted alphabetically).
        audio_path: Optional audio to mux in.
        output_path: Output path. Auto-generated if None.
        fps: Output frame rate for the slideshow.
        duration_per_frame: Seconds each frame is displayed.

    Returns:
        Path to the slideshow video.
    """
    if output_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = str(VIDEO_OUTPUT_DIR / f"slideshow_{timestamp}.mp4")

    # List and sort frames
    frames = sorted(Path(frames_dir).glob("*.png")) + sorted(Path(frames_dir).glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"No PNG/JPG frames found in {frames_dir}")

    # Create a concat file for ffmpeg
    concat_file = tempfile.mktemp(suffix=".txt")
    with open(concat_file, "w") as f:
        for frame in frames:
            f.write(f"file '{frame}'\n")
            f.write(f"duration {duration_per_frame}\n")
        # Add last frame again to avoid ffmpeg truncation
        f.write(f"file '{frames[-1]}'\n")

    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file]

    if audio_path:
        cmd.extend(["-i", audio_path])

    cmd.extend([
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-pix_fmt", "yuv420p", "-r", str(fps),
    ])

    if audio_path:
        cmd.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])

    cmd.append(output_path)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Slideshow creation failed: {result.stderr[-500:]}")

    # Cleanup
    try:
        os.unlink(concat_file)
    except Exception:
        pass

    return output_path


def generate_srt(segments, output_path=None):
    """Generate an SRT subtitle file from timed segments.

    Args:
        segments: List of dicts with 'start', 'end' (seconds), and 'text' keys.
        output_path: Output path. Auto-generated if None.

    Returns:
        Path to the SRT file.
    """
    if output_path is None:
        output_path = str(VIDEO_OUTPUT_DIR / f"subtitles_{time.strftime('%Y%m%d_%H%M%S')}.srt")

    def format_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    with open(output_path, "w") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_time(seg['start'])} --> {format_time(seg['end'])}\n")
            f.write(f"{seg['text']}\n\n")

    return output_path
