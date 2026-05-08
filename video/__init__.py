"""Video recording and editing for Content Automation Studio.

Provides dynamic video composition with Ken Burns effects, crossfade transitions,
text overlays, and subtitle generation — all using ffmpeg.
"""
from .editor import (
    compose_video, create_dynamic_video, create_slideshow,
    generate_srt, create_gradient_placeholder,
)
