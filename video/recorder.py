"""Browser session video recorder using CDP screencast frames.

Captures frames from the browser via Page.startScreencast and assembles
them into an MP4 video using ffmpeg. No external recording tools needed.

Usage with browser-harness -c:
    browser-harness -c '
    from video import start_recording, stop_recording
    start_recording()
    goto_url("https://example.com")
    wait(3)
    path = stop_recording()
    print(f"Video saved: {path}")
    '
"""

import base64
import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

# Output directory for recorded videos
VIDEO_OUTPUT_DIR = Path(os.environ.get("BH_VIDEO_DIR", Path.home() / "browser-harness-videos"))
VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class SessionRecorder:
    """Records browser sessions by capturing CDP screencast frames.

    Uses Page.startScreencast to receive frames as they render, saves them
    as PNG files, then assembles into MP4 using ffmpeg.
    """

    def __init__(self, fps=10, quality=80, max_dim=1920):
        """Initialize the recorder.

        Args:
            fps: Target frames per second for the output video.
            quality: JPEG quality 1-100 for screencast frames (lower = faster).
            max_dim: Maximum dimension for frame downscaling.
        """
        self.fps = fps
        self.quality = quality
        self.max_dim = max_dim
        self._frames_dir = None
        self._frame_count = 0
        self._recording = False
        self._frame_lock = threading.Lock()
        self._start_time = None

    def start(self, cdp_fn=None, helpers_module=None):
        """Start recording by enabling CDP screencast.

        Args:
            cdp_fn: The cdp() function from browser_harness.helpers.
                    If None, will import from helpers automatically.
            helpers_module: The helpers module (for capture_screenshot fallback).
        """
        if self._recording:
            raise RuntimeError("Recording already in progress")

        self._frames_dir = tempfile.mkdtemp(prefix="bh_recording_")
        self._frame_count = 0
        self._start_time = time.time()
        self._recording = True
        self._cdp = cdp_fn
        self._helpers = helpers_module

        if self._cdp is None:
            from browser_harness.helpers import cdp
            self._cdp = cdp

        if self._helpers is None:
            try:
                from browser_harness import helpers
                self._helpers = helpers
            except ImportError:
                pass

        # Enable CDP screencast - frames will be delivered as events
        try:
            self._cdp("Page.startScreencast", format="png", quality=self.quality,
                       maxWidth=self.max_dim, maxHeight=self.max_dim)
        except Exception:
            # Screencast not supported; fall back to periodic screenshots
            self._screencast_mode = "screenshot"
            self._screenshot_thread = threading.Thread(target=self._screenshot_loop, daemon=True)
            self._screenshot_thread.start()
        else:
            self._screencast_mode = "screencast"
            # Start a thread that drains events and saves frames
            self._drain_thread = threading.Thread(target=self._drain_screencast_frames, daemon=True)
            self._drain_thread.start()

        return self._frames_dir

    def _drain_screencast_frames(self):
        """Drain CDP screencast frame events and save as PNG files."""
        from browser_harness.helpers import drain_events
        while self._recording:
            try:
                events = drain_events()
                for event in events:
                    if event.get("method") == "Page.screencastFrame":
                        data = event.get("params", {}).get("data")
                        if data:
                            self._save_frame(data)
                        # Acknowledge the frame
                        session_id = event.get("params", {}).get("sessionId")
                        try:
                            self._cdp("Page.screencastFrameAck", sessionId=session_id)
                        except Exception:
                            pass
                time.sleep(0.01)
            except Exception:
                time.sleep(0.05)

    def _screenshot_loop(self):
        """Fallback: capture frames via periodic screenshots."""
        interval = 1.0 / self.fps
        while self._recording:
            try:
                if self._helpers:
                    path = self._helpers.capture_screenshot(
                        str(Path(self._frames_dir) / f"frame_{self._frame_count:06d}.png"),
                        max_dim=self.max_dim
                    )
                    with self._frame_lock:
                        self._frame_count += 1
            except Exception:
                pass
            time.sleep(interval)

    def _save_frame(self, base64_data):
        """Save a single frame from base64-encoded screencast data."""
        frame_path = Path(self._frames_dir) / f"frame_{self._frame_count:06d}.png"
        with open(frame_path, "wb") as f:
            f.write(base64.b64decode(base64_data))
        with self._frame_lock:
            self._frame_count += 1

    def stop(self, output_path=None, audio_path=None):
        """Stop recording and assemble the video.

        Args:
            output_path: Path for the output MP4 file. If None, auto-generates.
            audio_path: Optional path to an audio file to mux into the video.

        Returns:
            Path to the created MP4 file.
        """
        if not self._recording:
            raise RuntimeError("No recording in progress")

        self._recording = False

        # Stop screencast if active
        if self._screencast_mode == "screencast":
            try:
                self._cdp("Page.stopScreencast")
            except Exception:
                pass
            # Wait for drain thread to finish
            time.sleep(0.2)

        # Wait for screenshot thread
        if self._screencast_mode == "screenshot" and hasattr(self, "_screenshot_thread"):
            self._screenshot_thread.join(timeout=2.0)

        if self._frame_count == 0:
            raise RuntimeError("No frames captured during recording")

        # Generate output path
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = str(VIDEO_OUTPUT_DIR / f"recording_{timestamp}.mp4")

        output_path = str(Path(output_path).with_suffix(".mp4"))

        # Assemble video with ffmpeg
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(self.fps),
            "-i", str(Path(self._frames_dir) / "frame_%06d.png"),
        ]

        if audio_path:
            cmd.extend(["-i", audio_path])

        cmd.extend([
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "medium",
            "-crf", "23",
        ])

        if audio_path:
            cmd.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])

        cmd.append(output_path)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr[-500:]}")

        # Cleanup frame files
        import shutil
        try:
            shutil.rmtree(self._frames_dir)
        except Exception:
            pass

        return output_path

    @property
    def duration(self):
        """Elapsed recording time in seconds."""
        if self._start_time is None:
            return 0
        end = time.time() if self._recording else self._start_time
        return end - self._start_time

    @property
    def frame_count(self):
        """Number of frames captured so far."""
        with self._frame_lock:
            return self._frame_count


# --- Module-level convenience functions ---

_active_recorder = None


def start_recording(fps=10, quality=80, max_dim=1920):
    """Start recording the current browser session.

    Args:
        fps: Target frames per second.
        quality: Frame quality 1-100.
        max_dim: Maximum frame dimension.

    Returns:
        The SessionRecorder instance.
    """
    global _active_recorder
    if _active_recorder is not None and _active_recorder._recording:
        raise RuntimeError("A recording is already in progress. Call stop_recording() first.")

    _active_recorder = SessionRecorder(fps=fps, quality=quality, max_dim=max_dim)
    _active_recorder.start()
    return _active_recorder


def stop_recording(output_path=None, audio_path=None):
    """Stop recording and save the video.

    Args:
        output_path: Path for output MP4. Auto-generated if None.
        audio_path: Optional audio file to mux in.

    Returns:
        Path to the saved MP4 file.
    """
    global _active_recorder
    if _active_recorder is None or not _active_recorder._recording:
        raise RuntimeError("No recording in progress. Call start_recording() first.")

    path = _active_recorder.stop(output_path=output_path, audio_path=audio_path)
    _active_recorder = None
    return path


def capture_frame(output_path=None):
    """Capture a single frame from the browser as PNG.

    Useful for creating slideshow-style videos from specific browser states.

    Args:
        output_path: Path for the PNG. Auto-generated in VIDEO_OUTPUT_DIR if None.

    Returns:
        Path to the saved PNG.
    """
    from browser_harness.helpers import capture_screenshot
    if output_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S_%f")
        output_path = str(VIDEO_OUTPUT_DIR / f"frame_{timestamp}.png")
    return capture_screenshot(output_path)
