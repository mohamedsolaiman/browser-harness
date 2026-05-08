"""Dynamic video editor for Content Automation Studio.

Creates professional videos with:
- Stock video clip composition (primary mode)
- Ken Burns effect (zoom/pan) on AI-generated images (fallback)
- Crossfade transitions between scenes
- Animated text overlays (narration subtitles)
- Professional title cards
- SRT subtitle generation

All operations use ffmpeg - no moviepy dependency required.

Usage:
    from video.editor import create_dynamic_video, create_stock_video, generate_srt
    video = create_dynamic_video(images, audio_path, scenes, output_path)
    video = create_stock_video(video_clips, audio_path, scenes, output_path)
"""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

VIDEO_OUTPUT_DIR = Path(os.environ.get("CS_VIDEO_DIR", "/tmp/content-studio/videos"))
VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Resolution presets
RESOLUTIONS = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}


def _run_ffmpeg(cmd, description="ffmpeg", timeout=600):
    """Run an ffmpeg command with error handling."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            stderr = result.stderr[-1000:] if result.stderr else "No stderr"
            raise RuntimeError(f"{description} failed: {stderr}")
        return result
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{description} timed out after {timeout}s")


def _escape_text(text):
    """Escape text for ffmpeg drawtext filter (handle special characters)."""
    text = text.replace("'", "'\\''")
    text = text.replace(":", "\\:")
    text = text.replace("%", "%%")
    text = text.replace("\\n", " ")
    text = text.replace("\n", " ")
    text = text[:120]
    return text


def _get_audio_duration(path):
    """Get audio file duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return float(result.stdout.strip())
    except Exception:
        return 10.0


def _get_video_duration(path):
    """Get video file duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return float(result.stdout.strip())
    except Exception:
        return 10.0


# ──────────────────────────────────────────────────────────────
# Stock Video Composition (PRIMARY - creates dynamic video)
# ──────────────────────────────────────────────────────────────

def create_stock_video(
    video_paths: list,
    audio_path: str,
    scenes: list,
    output_path: str = None,
    resolution: str = "720p",
    fps: int = 24,
    title: str = None,
    visual_style: str = "cinematic",
    transition_duration: float = 0.8,
) -> str:
    """Create a video from stock footage clips with voiceover.

    This is the PRIMARY composition method. It:
    1. Trims each stock video clip to match scene duration
    2. Adds narration text overlays
    3. Applies crossfade transitions between clips
    4. Muxes with the audio/voiceover track
    5. Optionally adds a title card at the beginning

    Args:
        video_paths: List of stock video file paths (one per scene, may contain None).
        audio_path: Path to combined audio/narration file.
        scenes: List of scene dicts.
        output_path: Output video path.
        resolution: Output resolution preset.
        fps: Output frame rate.
        title: Title text for opening title card.
        visual_style: Visual style name.
        transition_duration: Crossfade duration in seconds.

    Returns:
        Path to the final video file.
    """
    if output_path is None:
        ts = int(time.time())
        output_path = str(VIDEO_OUTPUT_DIR / f"final_{ts}.mp4")

    width, height = RESOLUTIONS.get(resolution, (1280, 720))

    # Calculate total audio duration and per-scene durations
    total_duration = _get_audio_duration(audio_path) if audio_path and os.path.exists(audio_path) else None
    scene_durations = _calculate_scene_durations(scenes, total_duration)

    # Step 1: Prepare scene clips (trim stock videos to match scene durations)
    print(f"  [editor] Preparing {len(scenes)} scene clips from stock footage...")
    scene_clips = []

    for i, scene in enumerate(scenes):
        duration = scene_durations[i] if i < len(scene_durations) else 10.0
        narration = scene.get("narration", "")
        subtitle_text = narration[:100] + "..." if len(narration) > 100 else narration

        # Check if we have a stock video for this scene
        video_path = video_paths[i] if i < len(video_paths) else None

        if video_path and os.path.exists(video_path):
            try:
                clip_path = _prepare_stock_clip(
                    video_path=video_path,
                    duration_seconds=duration,
                    text_overlay=subtitle_text,
                    output_path=str(VIDEO_OUTPUT_DIR / f"scene_clip_{i:03d}.mp4"),
                    width=width,
                    height=height,
                    fps=fps,
                    scene_number=i,
                )
                scene_clips.append(clip_path)
                print(f"  [editor] Scene {i+1}: Stock clip prepared ({duration:.1f}s)")
            except Exception as e:
                print(f"  [editor] WARNING: Stock clip {i+1} failed: {e}, trying image fallback")
                # Try to use AI image for this scene if available
                scene_clips.append(None)
        else:
            scene_clips.append(None)
            print(f"  [editor] Scene {i+1}: No stock video, will use image fallback")

    # Step 2: Fill in missing clips with gradient placeholders that match the duration
    for i in range(len(scene_clips)):
        if scene_clips[i] is None:
            duration = scene_durations[i] if i < len(scene_durations) else 10.0
            scene = scenes[i] if i < len(scenes) else {}
            narration = scene.get("narration", f"Scene {i+1}")
            subtitle_text = narration[:100] + "..." if len(narration) > 100 else narration

            try:
                placeholder_path = _create_colored_clip(
                    text=subtitle_text[:80] if subtitle_text else f"Scene {i+1}",
                    duration=duration,
                    output_path=str(VIDEO_OUTPUT_DIR / f"placeholder_clip_{i:03d}.mp4"),
                    width=width,
                    height=height,
                    fps=fps,
                    scene_number=i,
                )
                scene_clips[i] = placeholder_path
                print(f"  [editor] Scene {i+1}: Created placeholder clip ({duration:.1f}s)")
            except Exception as e:
                print(f"  [editor] WARNING: Placeholder clip {i+1} also failed: {e}")

    # Filter out any remaining None values
    valid_clips = [c for c in scene_clips if c is not None and os.path.exists(c)]

    if not valid_clips:
        print("  [editor] No valid clips, creating audio-only video")
        return _create_audio_only_video(audio_path, output_path, width, height, fps, title, scenes)

    # Step 3: Optionally add title card at the beginning
    if title:
        try:
            title_clip = _create_title_clip(
                title=title,
                duration=4.0,
                output_path=str(VIDEO_OUTPUT_DIR / "title_clip.mp4"),
                width=width,
                height=height,
                fps=fps,
                style=visual_style,
            )
            valid_clips.insert(0, title_clip)
        except Exception as e:
            print(f"  [editor] WARNING: Title clip failed: {e}")

    # Step 4: Apply crossfade transitions and concatenate
    print(f"  [editor] Applying crossfade transitions to {len(valid_clips)} clips...")
    try:
        merged_video = apply_crossfade(
            valid_clips,
            transition_duration=transition_duration,
            output_path=str(VIDEO_OUTPUT_DIR / "merged_no_audio.mp4"),
            width=width,
            height=height,
            fps=fps,
        )
    except Exception as e:
        print(f"  [editor] WARNING: Crossfade failed ({e}), using simple concat")
        try:
            merged_video = _simple_concat(valid_clips, str(VIDEO_OUTPUT_DIR / "merged_no_audio.mp4"))
        except Exception as e2:
            print(f"  [editor] WARNING: Simple concat also failed: {e2}")
            merged_video = valid_clips[0]

    # Step 5: Mux with audio
    print(f"  [editor] Muxing video with audio...")
    try:
        _mux_audio_video(merged_video, audio_path, output_path)
    except Exception as e:
        print(f"  [editor] WARNING: Audio mux failed: {e}")
        try:
            import shutil
            shutil.copy2(merged_video, output_path)
        except Exception:
            output_path = merged_video

    # Cleanup temp clips
    for clip in scene_clips:
        try:
            if clip and os.path.exists(clip) and "final_" not in clip:
                os.unlink(clip)
        except Exception:
            pass

    print(f"  [editor] Final video: {output_path}")
    return output_path


def _prepare_stock_clip(
    video_path: str,
    duration_seconds: float,
    text_overlay: str = "",
    output_path: str = None,
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
    scene_number: int = 0,
) -> str:
    """Prepare a stock video clip: trim to duration, scale, and add text overlay.

    Args:
        video_path: Path to the source stock video.
        duration_seconds: Target duration in seconds.
        text_overlay: Text to display at the bottom.
        output_path: Output path for the prepared clip.
        width: Output width.
        height: Output height.
        fps: Output frame rate.
        scene_number: Scene number for variety.

    Returns:
        Path to the prepared clip.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if output_path is None:
        output_path = str(VIDEO_OUTPUT_DIR / f"stock_clip_{scene_number:03d}.mp4")

    # Get source video duration
    source_duration = _get_video_duration(video_path)

    # Build video filter
    # Scale to fit, pad with black bars if needed (like a smart TV)
    vf_parts = [
        f"scale={width}:{height}:force_original_aspect_ratio=decrease",
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black",
    ]

    # Add slow zoom effect for visual interest (Ken Burns on video)
    total_frames = int(duration_seconds * fps)
    if scene_number % 3 == 0:
        # Slow zoom in
        vf_parts.append(f"zoompan=z='min(zoom+0.0008,1.3)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={total_frames}:s={width}x{height}:fps={fps}")
    elif scene_number % 3 == 1:
        # Slow pan
        vf_parts.append(f"zoompan=z='1.15':x='iw*(1/zoom)/2+(iw/zoom)*(on/{total_frames}-0.5)':y='ih/2-(ih/zoom/2)':d={total_frames}:s={width}x{height}:fps={fps}")
    else:
        # Slow zoom out
        vf_parts.append(f"zoompan=z='if(lte(zoom,1.0),1.3,max(zoom-0.0008,1.0))':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={total_frames}:s={width}x{height}:fps={fps}")

    # Add text overlay at bottom
    if text_overlay:
        safe_text = _escape_text(text_overlay)
        vf_parts.append(
            f"drawbox=x=0:y=h-80:w=iw:h=80:color=black@0.6:t=fill"
        )
        vf_parts.append(
            f"drawtext=text='{safe_text}':"
            f"fontsize=24:fontcolor=white:"
            f"borderw=1:bordercolor=black@0.3:"
            f"x=(w-text_w)/2:y=h-55:"
            f"enable='gte(t,0.5)'"
        )

    vf = ",".join(vf_parts)

    # Determine if we need to trim or loop
    if source_duration >= duration_seconds:
        # Trim to target duration
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "26",
            "-pix_fmt", "yuv420p",
            "-t", str(duration_seconds),
            "-an",
            output_path,
        ]
    else:
        # Loop the video to fill the duration
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1",
            "-i", video_path,
            "-vf", vf,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "26",
            "-pix_fmt", "yuv420p",
            "-t", str(duration_seconds),
            "-an",
            output_path,
        ]

    _run_ffmpeg(cmd, description=f"Stock clip {scene_number}", timeout=120)
    return output_path


def _create_colored_clip(
    text: str,
    duration: float,
    output_path: str = None,
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
    scene_number: int = 0,
) -> str:
    """Create a styled colored clip with text overlay (fallback when no stock video).

    Args:
        text: Text to display.
        duration: Duration in seconds.
        output_path: Output path.
        width: Video width.
        height: Video height.
        fps: Frame rate.
        scene_number: Scene number for color variation.

    Returns:
        Path to the created clip.
    """
    if output_path is None:
        output_path = str(VIDEO_OUTPUT_DIR / f"color_clip_{scene_number:03d}.mp4")

    # Color palettes cycling through different styles
    palettes = [
        ("0x0a0a2e", "0x16213e", "0x4a9eff"),
        ("0x1a0a2a", "0x2d1b69", "0xbb86fc"),
        ("0x0a1a0a", "0x0d2818", "0x66bb6a"),
        ("0x2a0a0a", "0x3d0c02", "0xff7043"),
        ("0x1a1a2a", "0x222e3a", "0x80cbc4"),
    ]
    c1, c2, accent = palettes[scene_number % len(palettes)]

    safe_text = _escape_text(text[:80])
    total_frames = int(duration * fps)

    # Create a gradient + text clip with fade in/out
    vf = (
        f"gradients=s={width}x{height}:c0={c1}:c1={c2}:d={duration}:r={fps},"
        f"drawtext=text='{safe_text}':fontsize=32:"
        f"fontcolor={accent}:x=(w-text_w)/2:y=(h-text_h)/2:"
        f"borderw=2:bordercolor=black@0.5:"
        f"enable='gte(t,0.5)',"
        f"fade=t=in:st=0:d=0.8,fade=t=out:st={max(0, duration-0.8):.1f}:d=0.8"
    )

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", vf,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-pix_fmt", "yuv420p",
        "-t", str(duration),
        "-an",
        output_path,
    ]

    try:
        _run_ffmpeg(cmd, description=f"Color clip {scene_number}", timeout=30)
        return output_path
    except Exception:
        # Simpler fallback
        vf_simple = f"color=c={c1}:s={width}x{height}:d={duration}:r={fps}"
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", vf_simple,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-pix_fmt", "yuv420p",
            "-t", str(duration),
            "-an",
            output_path,
        ]
        _run_ffmpeg(cmd, description=f"Simple color clip {scene_number}", timeout=20)
        return output_path


def _calculate_scene_durations(scenes: list, total_duration: float = None) -> list:
    """Calculate the duration for each scene.

    If total_duration is available, divides evenly.
    Otherwise uses each scene's own duration_seconds field.

    Args:
        scenes: List of scene dicts.
        total_duration: Total audio duration in seconds.

    Returns:
        List of durations in seconds.
    """
    num_scenes = len(scenes)
    if num_scenes == 0:
        return []

    durations = []
    if total_duration and total_duration > 0:
        # Subtract 4 seconds for title card if we'll add one
        usable_duration = total_duration - 4.0 if total_duration > 12 else total_duration
        per_scene = usable_duration / num_scenes
        per_scene = max(per_scene, 3.0)  # Minimum 3 seconds per scene
        durations = [per_scene] * num_scenes
    else:
        for scene in scenes:
            d = scene.get("duration_seconds", 0)
            durations.append(float(d) if d > 0 else 10.0)

    return durations


# ──────────────────────────────────────────────────────────────
# Image-based Video Composition (FALLBACK - Ken Burns on images)
# ──────────────────────────────────────────────────────────────

def create_dynamic_video(
    image_paths: list,
    audio_path: str,
    scenes: list,
    output_path: str = None,
    resolution: str = "720p",
    fps: int = 24,
    title: str = None,
    visual_style: str = "cinematic",
    transition_duration: float = 1.0,
) -> str:
    """Create a dynamic video with Ken Burns effect, crossfades, and text overlays.

    This is the FALLBACK method used when stock videos are not available.
    Uses AI-generated images with zoom/pan effects.

    Args:
        image_paths: List of image file paths (one per scene).
        audio_path: Path to combined audio/narration file.
        scenes: List of scene dicts.
        output_path: Output video path.
        resolution: Output resolution preset.
        fps: Output frame rate.
        title: Title text for opening title card.
        visual_style: Visual style name.
        transition_duration: Crossfade duration in seconds.

    Returns:
        Path to the final video file.
    """
    if output_path is None:
        ts = int(time.time())
        output_path = str(VIDEO_OUTPUT_DIR / f"final_{ts}.mp4")

    width, height = RESOLUTIONS.get(resolution, (1280, 720))

    # Filter out None image paths
    valid_images = [(i, img) for i, img in enumerate(image_paths) if img and os.path.exists(img)]

    if not valid_images:
        print("  [editor] No valid images, creating audio-only video")
        return _create_audio_only_video(audio_path, output_path, width, height, fps, title, scenes)

    # Calculate durations
    total_duration = _get_audio_duration(audio_path) if audio_path and os.path.exists(audio_path) else None
    scene_durations = _calculate_scene_durations(scenes, total_duration)

    # Step 1: Create scene clips with Ken Burns effect
    print(f"  [editor] Creating {len(valid_images)} scene clips with Ken Burns effect...")
    scene_clips = []
    for idx, (scene_idx, img_path) in enumerate(valid_images):
        scene = scenes[scene_idx] if scene_idx < len(scenes) else {}
        duration = scene_durations[scene_idx] if scene_idx < len(scene_durations) else 10.0

        narration = scene.get("narration", "")
        subtitle_text = narration[:100] + "..." if len(narration) > 100 else narration

        try:
            clip_path = create_scene_clip(
                image_path=img_path,
                duration_seconds=duration,
                text_overlay=subtitle_text,
                output_path=str(VIDEO_OUTPUT_DIR / f"clip_{scene_idx:03d}.mp4"),
                width=width,
                height=height,
                fps=fps,
                scene_number=scene_idx + 1,
            )
            scene_clips.append(clip_path)
        except Exception as e:
            print(f"  [editor] WARNING: Scene clip {scene_idx + 1} failed: {e}")
            try:
                clip_path = _create_static_clip(
                    img_path, duration, str(VIDEO_OUTPUT_DIR / f"clip_{scene_idx:03d}_static.mp4"),
                    width, height, fps,
                )
                scene_clips.append(clip_path)
            except Exception as e2:
                print(f"  [editor] WARNING: Static clip also failed for scene {scene_idx + 1}: {e2}")

    if not scene_clips:
        print("  [editor] No scene clips created, falling back to audio-only video")
        return _create_audio_only_video(audio_path, output_path, width, height, fps, title, scenes)

    # Step 2: Optionally add title card
    if title:
        try:
            title_clip = _create_title_clip(
                title=title,
                duration=4.0,
                output_path=str(VIDEO_OUTPUT_DIR / "title_clip.mp4"),
                width=width,
                height=height,
                fps=fps,
                style=visual_style,
            )
            scene_clips.insert(0, title_clip)
        except Exception as e:
            print(f"  [editor] WARNING: Title clip failed: {e}")

    # Step 3: Apply crossfade transitions
    print(f"  [editor] Applying crossfade transitions...")
    try:
        merged_video = apply_crossfade(
            scene_clips,
            transition_duration=transition_duration,
            output_path=str(VIDEO_OUTPUT_DIR / "merged_no_audio.mp4"),
            width=width,
            height=height,
            fps=fps,
        )
    except Exception as e:
        print(f"  [editor] WARNING: Crossfade failed ({e}), using simple concat")
        try:
            merged_video = _simple_concat(scene_clips, str(VIDEO_OUTPUT_DIR / "merged_no_audio.mp4"))
        except Exception as e2:
            merged_video = scene_clips[0]

    # Step 4: Mux with audio
    print(f"  [editor] Muxing video with audio...")
    try:
        _mux_audio_video(merged_video, audio_path, output_path)
    except Exception as e:
        print(f"  [editor] WARNING: Audio mux failed: {e}")
        try:
            import shutil
            shutil.copy2(merged_video, output_path)
        except Exception:
            output_path = merged_video

    # Cleanup temp clips
    for clip in scene_clips:
        try:
            if os.path.exists(clip) and "final_" not in clip:
                os.unlink(clip)
        except Exception:
            pass

    print(f"  [editor] Final video: {output_path}")
    return output_path


def create_scene_clip(
    image_path: str,
    duration_seconds: float,
    text_overlay: str = "",
    output_path: str = None,
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
    scene_number: int = 0,
) -> str:
    """Create a single scene clip with Ken Burns (zoom/pan) effect and optional text overlay."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if output_path is None:
        output_path = str(VIDEO_OUTPUT_DIR / f"scene_clip_{int(time.time())}_{scene_number}.mp4")

    total_frames = int(duration_seconds * fps)

    # Alternate zoom direction for visual variety
    if scene_number % 3 == 0:
        zoom_filter = (
            f"zoompan=z='min(zoom+0.0012,1.4)':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={total_frames}:s={width}x{height}:fps={fps}"
        )
    elif scene_number % 3 == 1:
        zoom_filter = (
            f"zoompan=z='1.2':"
            f"x='iw*(1/zoom)/2 + (iw/zoom)*(on/{total_frames} - 0.5)':"
            f"y='ih/2-(ih/zoom/2)':"
            f"d={total_frames}:s={width}x{height}:fps={fps}"
        )
    else:
        zoom_filter = (
            f"zoompan=z='if(lte(zoom,1.0),1.4,max(zoom-0.0012,1.0))':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={total_frames}:s={width}x{height}:fps={fps}"
        )

    vf_parts = [zoom_filter]

    if text_overlay:
        safe_text = _escape_text(text_overlay)
        vf_parts.append(f"drawbox=x=0:y=h-80:w=iw:h=80:color=black@0.6:t=fill")
        vf_parts.append(
            f"drawtext=text='{safe_text}':"
            f"fontsize=24:fontcolor=white:"
            f"borderw=1:bordercolor=black@0.3:"
            f"x=(w-text_w)/2:y=h-55:"
            f"enable='gte(t,0.5)'"
        )

    vf = ",".join(vf_parts)

    cmd = [
        "ffmpeg", "-y",
        "-i", image_path,
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "28",
        "-pix_fmt", "yuv420p",
        "-t", str(duration_seconds),
        "-an",
        output_path,
    ]

    _run_ffmpeg(cmd, description=f"Scene clip {scene_number}")
    return output_path


# ──────────────────────────────────────────────────────────────
# Shared composition utilities
# ──────────────────────────────────────────────────────────────

def apply_crossfade(
    clips: list,
    transition_duration: float = 1.0,
    output_path: str = None,
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
) -> str:
    """Apply crossfade transitions between video clips."""
    if not clips:
        raise RuntimeError("No clips to merge")

    if output_path is None:
        output_path = str(VIDEO_OUTPUT_DIR / f"merged_{int(time.time())}.mp4")

    if len(clips) == 1:
        cmd = [
            "ffmpeg", "-y",
            "-i", clips[0],
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-pix_fmt", "yuv420p",
            "-an",
            output_path,
        ]
        _run_ffmpeg(cmd, description="Single clip re-encode")
        return output_path

    current_clip = clips[0]

    for i in range(1, len(clips)):
        intermediate_path = str(VIDEO_OUTPUT_DIR / f"xfade_{i}_{int(time.time())}.mp4")
        current_duration = _get_video_duration(current_clip)
        offset = max(0, current_duration - transition_duration)

        transitions = ["fade", "fadeblack", "fadewhite", "slideright", "slideleft", "dissolve"]
        transition = transitions[i % len(transitions)]

        cmd = [
            "ffmpeg", "-y",
            "-i", current_clip,
            "-i", clips[i],
            "-filter_complex",
            f"[0:v][1:v]xfade=transition={transition}:offset={offset:.2f}:duration={transition_duration:.2f}[v]",
            "-map", "[v]",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-pix_fmt", "yuv420p",
            "-an",
            intermediate_path,
        ]

        try:
            _run_ffmpeg(cmd, description=f"Crossfade clip {i}")
        except RuntimeError as e:
            print(f"  [editor] WARNING: xfade failed for clip {i}: {e}, trying concat")
            try:
                intermediate_path = _simple_concat([current_clip, clips[i]], intermediate_path)
            except Exception:
                intermediate_path = current_clip
                break

        if i > 1 and os.path.exists(current_clip) and "xfade_" in current_clip:
            try:
                os.unlink(current_clip)
            except Exception:
                pass

        current_clip = intermediate_path

    if current_clip != output_path:
        import shutil
        shutil.move(current_clip, output_path)

    return output_path


def _create_title_clip(
    title: str,
    duration: float = 4.0,
    output_path: str = None,
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
    style: str = "cinematic",
) -> str:
    """Create an animated title card clip."""
    if output_path is None:
        output_path = str(VIDEO_OUTPUT_DIR / f"title_{int(time.time())}.mp4")

    safe_title = _escape_text(title)

    bg_colors = {
        "cinematic": "#0a0a1a",
        "corporate": "#1a2a3a",
        "tech_code": "#0a1628",
        "nature": "#0a1a0a",
        "abstract": "#1a0a2a",
    }
    bg = bg_colors.get(style, "#0a0a1a")

    vf = (
        f"color=c={bg}:s={width}x{height}:d={duration}:r={fps},"
        f"drawtext=text='{safe_title}':"
        f"fontsize=52:fontcolor=white:"
        f"borderw=3:bordercolor=black@0.5:"
        f"x=(w-text_w)/2:y=(h-text_h)/2:"
        f"enable='gte(t,0.5)',"
        f"fade=t=in:st=0:d=1.5,fade=t=out:st={duration - 1}:d=1"
    )

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", vf,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-pix_fmt", "yuv420p",
        "-t", str(duration),
        "-an",
        output_path,
    ]

    _run_ffmpeg(cmd, description="Title clip")
    return output_path


def _create_static_clip(
    image_path: str,
    duration: float,
    output_path: str,
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
) -> str:
    """Create a simple static clip from an image (no Ken Burns, fallback)."""
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", image_path,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-pix_fmt", "yuv420p",
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        "-t", str(duration),
        "-r", str(fps),
        "-an",
        output_path,
    ]

    _run_ffmpeg(cmd, description="Static clip")
    return output_path


def _simple_concat(clips: list, output_path: str) -> str:
    """Concatenate clips without transitions (fallback)."""
    concat_file = tempfile.mktemp(suffix=".txt")
    with open(concat_file, "w") as f:
        for clip in clips:
            f.write(f"file '{clip}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_file,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-pix_fmt", "yuv420p",
        "-an",
        output_path,
    ]

    try:
        _run_ffmpeg(cmd, description="Simple concat")
    finally:
        try:
            os.unlink(concat_file)
        except Exception:
            pass

    return output_path


def _mux_audio_video(video_path: str, audio_path: str, output_path: str) -> str:
    """Multiplex audio and video streams."""
    if not audio_path or not os.path.exists(audio_path):
        import shutil
        shutil.copy2(video_path, output_path)
        return output_path

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_path,
    ]

    _run_ffmpeg(cmd, description="Audio-video mux")
    return output_path


def _create_audio_only_video(
    audio_path: str,
    output_path: str,
    width: int,
    height: int,
    fps: int,
    title: str = None,
    scenes: list = None,
) -> str:
    """Create a video with black screen + audio (fallback)."""
    vf = f"color=c=black:s={width}x{height}:r={fps}"
    if title:
        safe_title = _escape_text(title)
        vf += f",drawtext=text='{safe_title}':fontsize=48:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2"

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", vf,
        "-i", audio_path,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_path,
    ]

    _run_ffmpeg(cmd, description="Audio-only video")
    return output_path


def create_gradient_placeholder(
    text: str,
    output_path: str,
    width: int = 1280,
    height: int = 720,
    scene_number: int = 0,
) -> str:
    """Create a gradient placeholder image using ffmpeg."""
    gradients = [
        ("0x1a1a2e", "0x16213e"),
        ("0x2d1b69", "0x11001c"),
        ("0x0d2818", "0x04471c"),
        ("0x3d0c02", "0x5f0e02"),
        ("0x1b262c", "0x222e3a"),
    ]
    c1, c2 = gradients[scene_number % len(gradients)]

    safe_text = _escape_text(text[:60])

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i",
        f"gradients=s={width}x{height}:c0={c1}:c1={c2}:d=1",
        "-frames:v", "1",
        "-vf",
        f"drawtext=text='{safe_text}':fontsize=36:"
        f"fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2:"
        f"borderw=2:bordercolor=black@0.5",
        output_path,
    ]

    try:
        _run_ffmpeg(cmd, description="Gradient placeholder", timeout=15)
        return output_path
    except Exception:
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=#1a1a2e:s={width}x{height}:d=1",
            "-frames:v", "1",
            output_path,
        ]
        try:
            _run_ffmpeg(cmd, description="Simple placeholder", timeout=10)
        except Exception:
            pass
        return output_path


def compose_video(video_path=None, audio_path=None, title=None, output_path=None,
                  resolution="1080p", fps=30, watermark=None):
    """Convenience function to compose a video from parts."""
    if output_path is None:
        ts = int(time.time())
        output_path = str(VIDEO_OUTPUT_DIR / f"composed_{ts}.mp4")

    width, height = RESOLUTIONS.get(resolution, (1920, 1080))

    if video_path and audio_path:
        return _mux_audio_video(video_path, audio_path, output_path)
    elif video_path:
        import shutil
        shutil.copy2(video_path, output_path)
        return output_path
    elif audio_path:
        return _create_audio_only_video(audio_path, output_path, width, height, fps, title)
    else:
        raise RuntimeError("No video or audio inputs provided")


def create_slideshow(frames_dir, audio_path=None, output_path=None, fps=2, duration_per_frame=5):
    """Create a slideshow video from a directory of PNG/JPG frames."""
    if output_path is None:
        ts = int(time.time())
        output_path = str(VIDEO_OUTPUT_DIR / f"slideshow_{ts}.mp4")

    frames = sorted(Path(frames_dir).glob("*.png")) + sorted(Path(frames_dir).glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"No PNG/JPG frames found in {frames_dir}")

    concat_file = tempfile.mktemp(suffix=".txt")
    with open(concat_file, "w") as f:
        for frame in frames:
            f.write(f"file '{frame}'\n")
            f.write(f"duration {duration_per_frame}\n")
        f.write(f"file '{frames[-1]}'\n")

    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file]
    if audio_path:
        cmd.extend(["-i", audio_path])
    cmd.extend([
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-pix_fmt", "yuv420p", "-r", str(fps),
    ])
    if audio_path:
        cmd.extend(["-c:a", "aac", "-b:a", "192k", "-shortest"])
    cmd.append(output_path)

    try:
        _run_ffmpeg(cmd, description="Slideshow creation")
    finally:
        try:
            os.unlink(concat_file)
        except Exception:
            pass

    return output_path


def generate_srt(segments, output_path=None):
    """Generate an SRT subtitle file from timed segments."""
    if output_path is None:
        output_path = str(VIDEO_OUTPUT_DIR / f"subtitles_{int(time.time())}.srt")

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
            text = seg.get("text", "")
            words = text.split()
            lines = []
            current = ""
            for w in words:
                if len(current) + len(w) + 1 > 42:
                    lines.append(current)
                    current = w
                else:
                    current = (current + " " + w).strip()
            if current:
                lines.append(current)
            f.write("\n".join(lines[:3]) + "\n\n")

    return output_path
