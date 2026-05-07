"""Video recording and editing for browser-harness.

Provides browser session recording (CDP screencast frames → MP4)
and video editing/compositing (add TTS audio, titles, overlays).
"""
from .recorder import SessionRecorder, start_recording, stop_recording, capture_frame
from .editor import VideoEditor, compose_video
