"""Plan executor — orchestrates video creation with AI visuals, TTS, and dynamic composition.

Takes a content plan (from planner.py) and executes it step by step:
1. Generate TTS audio for each scene
2. Generate AI images for each scene (using Pollinations.ai)
2b. Optionally download stock videos (if PEXELS_API_KEY is set)
3. Compose dynamic video with Ken Burns + transitions
4. Publish to YouTube, TikTok, and X (when enabled)

All credentials from environment variables — never hard-coded.
"""

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

# Import sibling modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tts.mimo_tts import MimoTTS, generate_speech
from video.editor import (
    create_dynamic_video, generate_srt, create_gradient_placeholder, compose_video
)
from visuals.image_gen import generate_scene_images, generate_image, enhance_prompt
from visuals.stock_video import get_videos_for_topic

VIDEO_OUTPUT_DIR = Path(os.environ.get("CS_VIDEO_DIR", "/tmp/content-studio/videos"))
AUDIO_OUTPUT_DIR = Path(os.environ.get("CS_AUDIO_DIR", "/tmp/content-studio/audio"))
IMAGE_OUTPUT_DIR = Path(os.environ.get("CS_IMAGE_DIR", "/tmp/content-studio/images"))


class PlanExecutor:
    """Executes a content plan: generates TTS, AI images, composes dynamic video, and publishes.

    The executor is designed to run step by step, with each step producing
    artifacts that the next step consumes. All steps have robust error handling
    and fallbacks — the executor should never crash.
    """

    def __init__(self, plan=None, plan_path=None, tts_voice="mimo_default",
                 video_resolution="720p", visual_style="cinematic",
                 visual_mode="ai_images"):
        """Initialize the executor.

        Args:
            plan: Content plan dict (from ContentPlanner.plan()).
            plan_path: Path to a saved plan JSON file (alternative to plan dict).
            tts_voice: Default TTS voice to use.
            video_resolution: Output video resolution.
            visual_style: Visual style for image generation.
            visual_mode: Visual source mode ("ai_images", "stock_videos", "ai_plus_stock").
        """
        if plan is not None:
            self.plan = plan
        elif plan_path is not None:
            with open(plan_path) as f:
                self.plan = json.load(f)
        else:
            raise ValueError("Provide either plan or plan_path")

        self.tts_voice = tts_voice
        self.video_resolution = video_resolution
        self.visual_style = visual_style
        self.visual_mode = visual_mode
        self.artifacts = {
            "audio_files": [],
            "image_files": [],
            "stock_videos": [],
            "combined_audio": None,
            "final_video": None,
            "subtitle_file": None,
            "published_to": [],
        }
        self._tts_client = None

    def execute(self, steps=None, publish=True, headless=True):
        """Execute the content plan.

        Args:
            steps: List of step names to execute. Default: all steps.
                   Options: "tts", "visuals", "compose", "publish"
            publish: Whether to publish after composing.
            headless: Always True now (no browser needed).

        Returns:
            Dict with execution results and artifact paths.
        """
        if steps is None:
            steps = ["tts", "visuals", "compose"]
            if publish:
                steps.append("publish")

        step_handlers = {
            "tts": self._step_tts,
            "visuals": self._step_visuals,
            "compose": self._step_compose,
            "publish": self._step_publish,
        }

        for step in steps:
            if step not in step_handlers:
                raise ValueError(f"Unknown step: {step}. Options: {list(step_handlers.keys())}")
            print(f"[executor] Running step: {step}")
            try:
                step_handlers[step]()
            except Exception as e:
                print(f"[executor] Step {step} failed: {e}")
                if step == "compose":
                    # Try fallback composition
                    self._fallback_compose()
            print(f"[executor] Step {step} complete")

        self.plan["status"] = "executed"
        return {
            "plan_title": self.plan.get("title"),
            "artifacts": self.artifacts,
            "status": self.plan["status"],
        }

    def _get_tts_client(self):
        """Get or create the TTS client."""
        if self._tts_client is None:
            self._tts_client = MimoTTS()
        return self._tts_client

    def _step_tts(self, **kwargs):
        """Generate TTS audio for all scenes.

        If TTS fails for a scene, estimates duration and continues.
        """
        scenes = self.plan.get("scenes", [])
        if not scenes:
            raise RuntimeError("Plan has no scenes")

        tts = self._get_tts_client()
        audio_files = []
        subtitle_segments = []
        current_time = 0

        for i, scene in enumerate(scenes):
            narration = scene.get("narration", "")
            if not narration:
                continue

            output_path = str(AUDIO_OUTPUT_DIR / f"scene_{i:03d}.wav")

            print(f"  [tts] Generating audio for scene {i+1}/{len(scenes)}: {narration[:50]}...")
            try:
                path = tts.generate(narration, voice=self.tts_voice, output_path=output_path)
                audio_files.append(path)

                # Get audio duration for subtitle timing
                duration = self._get_audio_duration(path)
                subtitle_segments.append({
                    "start": current_time,
                    "end": current_time + duration,
                    "text": narration,
                })
                current_time += duration

            except Exception as e:
                print(f"  [tts] WARNING: Failed to generate audio for scene {i+1}: {e}")
                # Estimate duration from text length (~150 words/min)
                words = len(narration.split())
                estimated_duration = max((words / 150) * 60, 5.0)
                subtitle_segments.append({
                    "start": current_time,
                    "end": current_time + estimated_duration,
                    "text": narration,
                })
                current_time += estimated_duration

        # Concatenate all audio files into one
        if audio_files:
            combined_audio = str(AUDIO_OUTPUT_DIR / "full_narration.wav")
            try:
                self._concatenate_audio_files(audio_files, combined_audio)
                self.artifacts["audio_files"] = audio_files
                self.artifacts["combined_audio"] = combined_audio
            except Exception as e:
                print(f"  [tts] WARNING: Audio concatenation failed: {e}")
                # Use first audio file if concatenation fails
                if audio_files:
                    self.artifacts["combined_audio"] = audio_files[0]
                    self.artifacts["audio_files"] = audio_files
        else:
            print("  [tts] WARNING: No audio files generated")
            self.artifacts["combined_audio"] = None

        # Generate SRT subtitles
        if subtitle_segments:
            try:
                srt_path = generate_srt(subtitle_segments)
                self.artifacts["subtitle_file"] = srt_path
            except Exception as e:
                print(f"  [tts] WARNING: SRT generation failed: {e}")

    def _step_visuals(self, **kwargs):
        """Generate AI images for each scene and optionally download stock videos.

        Visual modes:
        - "ai_images": Generate images using Pollinations.ai only
        - "stock_videos": Download stock videos from Pexels only
        - "ai_plus_stock": Both AI images and stock videos
        """
        scenes = self.plan.get("scenes", [])
        if not scenes:
            raise RuntimeError("Plan has no scenes")

        # Step 2a: Generate AI images
        if self.visual_mode in ("ai_images", "ai_plus_stock"):
            print(f"  [visuals] Generating AI images for {len(scenes)} scenes (style: {self.visual_style})...")
            try:
                image_paths = generate_scene_images(
                    scenes=scenes,
                    output_dir=str(IMAGE_OUTPUT_DIR),
                    style=self.visual_style,
                )
                self.artifacts["image_files"] = image_paths
                success = sum(1 for p in image_paths if p is not None)
                print(f"  [visuals] Generated {success}/{len(scenes)} AI images")
            except Exception as e:
                print(f"  [visuals] WARNING: AI image generation failed: {e}")
                # Create gradient placeholders for all scenes
                self.artifacts["image_files"] = self._create_all_placeholders(scenes)

        # Step 2b: Optionally download stock videos
        if self.visual_mode in ("stock_videos", "ai_plus_stock"):
            pexels_key = os.environ.get("PEXELS_API_KEY", "")
            if pexels_key:
                topic = self.plan.get("topic", self.plan.get("title", ""))
                print(f"  [visuals] Downloading stock videos for topic: {topic}")
                try:
                    stock_paths = get_videos_for_topic(
                        topic=topic,
                        scenes=scenes,
                        api_key=pexels_key,
                    )
                    self.artifacts["stock_videos"] = stock_paths
                except Exception as e:
                    print(f"  [visuals] WARNING: Stock video download failed: {e}")
            else:
                print("  [visuals] No PEXELS_API_KEY set, skipping stock videos")
                if self.visual_mode == "stock_videos":
                    # Stock-only mode but no API key — fall back to AI images
                    print("  [visuals] Falling back to AI image generation...")
                    try:
                        image_paths = generate_scene_images(
                            scenes=scenes,
                            output_dir=str(IMAGE_OUTPUT_DIR),
                            style=self.visual_style,
                        )
                        self.artifacts["image_files"] = image_paths
                    except Exception:
                        self.artifacts["image_files"] = self._create_all_placeholders(scenes)

        # Ensure we have at least some visual assets
        if not self.artifacts["image_files"] and not self.artifacts["stock_videos"]:
            print("  [visuals] No visual assets generated, creating placeholders...")
            self.artifacts["image_files"] = self._create_all_placeholders(scenes)

    def _step_compose(self, **kwargs):
        """Compose the final video with Ken Burns effects, transitions, and text overlays."""
        scenes = self.plan.get("scenes", [])
        title = self.plan.get("title", "Untitled")
        audio_path = self.artifacts.get("combined_audio")

        # Determine visual source: prefer stock videos if available, else AI images
        image_paths = self.artifacts.get("image_files", [])

        if not image_paths:
            raise RuntimeError("No visual assets available for video composition")

        # Filter out None paths
        valid_images = [p for p in image_paths if p and os.path.exists(p)]
        if not valid_images:
            raise RuntimeError("No valid image files found for video composition")

        output_path = str(VIDEO_OUTPUT_DIR / "final_video.mp4")

        print(f"  [compose] Creating dynamic video with {len(valid_images)} images...")
        try:
            result = create_dynamic_video(
                image_paths=image_paths,
                audio_path=audio_path,
                scenes=scenes,
                output_path=output_path,
                resolution=self.video_resolution,
                title=title,
                visual_style=self.visual_style,
            )
            self.artifacts["final_video"] = result
            print(f"  [compose] Final video: {result}")
        except Exception as e:
            print(f"  [compose] Dynamic video failed: {e}")
            # Try simpler composition
            raise

    def _fallback_compose(self, **kwargs):
        """Simple fallback video composition if the dynamic method fails."""
        audio_path = self.artifacts.get("combined_audio")
        image_paths = self.artifacts.get("image_files", [])
        title = self.plan.get("title", "Untitled")

        output_path = str(VIDEO_OUTPUT_DIR / "final_video_fallback.mp4")

        # Try creating a simple slideshow from images
        if image_paths:
            valid_images = [p for p in image_paths if p and os.path.exists(p)]
            if valid_images:
                try:
                    # Create a directory with just the valid images
                    slides_dir = VIDEO_OUTPUT_DIR / "fallback_frames"
                    slides_dir.mkdir(exist_ok=True)
                    for i, img in enumerate(valid_images):
                        import shutil
                        ext = Path(img).suffix
                        dest = str(slides_dir / f"frame_{i:03d}{ext}")
                        shutil.copy2(img, dest)

                    from video.editor import create_slideshow
                    video_path = create_slideshow(
                        str(slides_dir),
                        audio_path=audio_path,
                        output_path=str(VIDEO_OUTPUT_DIR / "fallback_slideshow.mp4"),
                        duration_per_frame=10,
                    )
                    self.artifacts["final_video"] = video_path
                    return
                except Exception as e:
                    print(f"  [compose] Fallback slideshow also failed: {e}")

        # Last resort: audio-only video
        if audio_path:
            try:
                from video.editor import compose_video
                result = compose_video(
                    audio_path=audio_path,
                    title=title,
                    output_path=output_path,
                    resolution=self.video_resolution,
                )
                self.artifacts["final_video"] = result
            except Exception as e:
                print(f"  [compose] Even audio-only video failed: {e}")

    def _step_publish(self, **kwargs):
        """Publish the final video to configured platforms."""
        video_path = self.artifacts.get("final_video")
        if not video_path:
            print("  [publish] No final video to publish")
            return

        publishing = self.plan.get("publishing", {})
        published_to = []

        # YouTube
        if "youtube" in publishing and os.environ.get("YOUTUBE_ENABLED"):
            published_to.append({
                "platform": "youtube",
                "error": "Browser automation not available in this deployment"
            })

        # TikTok
        if "tiktok" in publishing and os.environ.get("TIKTOK_ENABLED"):
            published_to.append({
                "platform": "tiktok",
                "error": "Browser automation not available in this deployment"
            })

        # X
        if "x" in publishing and os.environ.get("X_ENABLED"):
            published_to.append({
                "platform": "x",
                "error": "Browser automation not available in this deployment"
            })

        self.artifacts["published_to"] = published_to

    def _create_all_placeholders(self, scenes):
        """Create gradient placeholder images for all scenes.

        Args:
            scenes: List of scene dicts.

        Returns:
            List of placeholder image paths.
        """
        placeholders = []
        for i, scene in enumerate(scenes):
            text = scene.get("visual_instructions", scene.get("narration", f"Scene {i+1}"))[:80]
            output_path = str(IMAGE_OUTPUT_DIR / f"placeholder_{i:03d}.png")
            try:
                path = create_gradient_placeholder(
                    text=text,
                    output_path=output_path,
                    scene_number=i + 1,
                )
                placeholders.append(path)
            except Exception as e:
                print(f"  [visuals] WARNING: Placeholder failed for scene {i+1}: {e}")
                placeholders.append(None)
        return placeholders

    # --- Utility methods ---

    def _get_audio_duration(self, path):
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

    def _concatenate_audio_files(self, paths, output_path):
        """Concatenate multiple audio files into one."""
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

        try:
            os.unlink(concat_file)
        except Exception:
            pass


# --- Module-level convenience ---

def execute_plan(plan=None, plan_path=None, tts_voice="mimo_default", publish=True,
                 video_resolution="720p", visual_style="cinematic",
                 visual_mode="ai_images"):
    """Execute a content plan end-to-end.

    Args:
        plan: Content plan dict (from create_plan).
        plan_path: Path to a saved plan JSON file.
        tts_voice: TTS voice to use.
        publish: Whether to publish after composing.
        video_resolution: Output video resolution.
        visual_style: Visual style for AI image generation.
        visual_mode: Visual source mode ("ai_images", "stock_videos", "ai_plus_stock").

    Returns:
        Dict with execution results.
    """
    executor = PlanExecutor(
        plan=plan, plan_path=plan_path,
        tts_voice=tts_voice, video_resolution=video_resolution,
        visual_style=visual_style, visual_mode=visual_mode,
    )
    return executor.execute(publish=publish)
