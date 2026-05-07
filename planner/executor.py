"""Plan executor — orchestrates video creation, TTS, and publishing.

Takes a content plan (from planner.py) and executes it step by step:
1. Generate TTS audio for each scene
2. Record/capture browser visuals
3. Compose the final video
4. Publish to YouTube, TikTok, and X

All credentials from environment variables — never hard-coded.

Usage:
    from planner import create_plan, execute_plan
    plan = create_plan("Python decorators tutorial")
    result = execute_plan(plan)
    print(f"Published: {result['published_to']}")
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
from video.recorder import SessionRecorder, start_recording, stop_recording, capture_frame
from video.editor import VideoEditor, compose_video, create_slideshow, generate_srt

VIDEO_OUTPUT_DIR = Path(os.environ.get("BH_VIDEO_DIR", Path.home() / "browser-harness-videos"))


class PlanExecutor:
    """Executes a content plan: generates TTS, records video, composes, and publishes.

    The executor is designed to run step by step, with each step producing
    artifacts that the next step consumes. This allows for resumability and
    partial execution.
    """

    def __init__(self, plan=None, plan_path=None, tts_voice="alloy", video_resolution="1080p"):
        """Initialize the executor.

        Args:
            plan: Content plan dict (from ContentPlanner.plan()).
            plan_path: Path to a saved plan JSON file (alternative to plan dict).
            tts_voice: Default TTS voice to use.
            video_resolution: Output video resolution.
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
        self.artifacts = {
            "audio_files": [],
            "frame_dirs": [],
            "video_segments": [],
            "final_video": None,
            "published_to": [],
        }
        self._tts_client = None

    def execute(self, steps=None, publish=True, headless=False):
        """Execute the content plan.

        Args:
            steps: List of step names to execute. Default: all steps.
                   Options: "tts", "record", "compose", "publish"
            publish: Whether to publish after composing.
            headless: If True, skip browser recording (use slideshow mode).

        Returns:
            Dict with execution results and artifact paths.
        """
        if steps is None:
            steps = ["tts", "record", "compose"]
            if publish:
                steps.append("publish")

        step_handlers = {
            "tts": self._step_tts,
            "record": self._step_record,
            "compose": self._step_compose,
            "publish": self._step_publish,
        }

        for step in steps:
            if step not in step_handlers:
                raise ValueError(f"Unknown step: {step}. Options: {list(step_handlers.keys())}")
            print(f"[executor] Running step: {step}")
            step_handlers[step](headless=headless)
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
        """Generate TTS audio for all scenes."""
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

            output_path = str(VIDEO_OUTPUT_DIR / f"scene_{i:03d}.wav")

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
                estimated_duration = (words / 150) * 60
                subtitle_segments.append({
                    "start": current_time,
                    "end": current_time + estimated_duration,
                    "text": narration,
                })
                current_time += estimated_duration

        # Concatenate all audio files into one
        if audio_files:
            combined_audio = str(VIDEO_OUTPUT_DIR / "full_narration.wav")
            self._concatenate_audio_files(audio_files, combined_audio)
            self.artifacts["audio_files"] = audio_files
            self.artifacts["combined_audio"] = combined_audio
        else:
            self.artifacts["combined_audio"] = None

        # Generate SRT subtitles
        if subtitle_segments:
            srt_path = generate_srt(subtitle_segments)
            self.artifacts["subtitle_file"] = srt_path

    def _step_record(self, headless=False, **kwargs):
        """Record browser visuals for each scene.

        Two modes:
        - Browser recording: Navigate to URLs and record the browser session
        - Slideshow: Capture screenshots at each URL and create a slideshow
        """
        scenes = self.plan.get("scenes", [])
        if not scenes:
            raise RuntimeError("Plan has no scenes")

        frames_dir = VIDEO_OUTPUT_DIR / "frames"
        frames_dir.mkdir(exist_ok=True)

        try:
            from browser_harness.helpers import (
                goto_url, new_tab, capture_screenshot, wait_for_load,
                wait, page_info, click_at_xy, scroll
            )
            browser_available = True
        except ImportError:
            browser_available = False
            print("  [record] Browser harness not available, using slideshow mode")

        if headless or not browser_available:
            self._record_slideshow(scenes, frames_dir)
        else:
            self._record_browser(scenes, frames_dir)

    def _record_browser(self, scenes, frames_dir):
        """Record browser session by navigating through scenes."""
        from browser_harness.helpers import (
            goto_url, new_tab, capture_screenshot, wait_for_load,
            wait, page_info, click_at_xy, scroll
        )

        # Start recording
        recorder = start_recording(fps=8, quality=75)

        for i, scene in enumerate(scenes):
            visual_type = scene.get("visual_type", "browser_recording")
            url = scene.get("url")
            instructions = scene.get("visual_instructions", "")

            print(f"  [record] Scene {i+1}: {visual_type} — {instructions[:60]}...")

            if url:
                try:
                    goto_url(url)
                    wait_for_load(timeout=15)
                    wait(2)  # Let page settle
                except Exception as e:
                    print(f"  [record] WARNING: Navigation failed for scene {i+1}: {e}")

            # Execute visual instructions
            if "scroll down" in instructions.lower():
                for _ in range(3):
                    scroll(960, 400, dy=-300)
                    wait(0.5)
            elif "scroll up" in instructions.lower():
                for _ in range(3):
                    scroll(960, 400, dy=300)
                    wait(0.5)

            # Pause for the scene duration
            duration = scene.get("duration_seconds", 10)
            wait(min(duration, 15))

        # Stop recording and save
        try:
            video_path = stop_recording(output_path=str(VIDEO_OUTPUT_DIR / "raw_recording.mp4"))
            self.artifacts["video_segments"].append(video_path)
        except Exception as e:
            print(f"  [record] WARNING: Recording stop failed: {e}")
            # Fallback to screenshots
            self._record_slideshow(scenes, frames_dir)

    def _record_slideshow(self, scenes, frames_dir):
        """Create slideshow frames from URLs."""
        try:
            from browser_harness.helpers import (
                goto_url, new_tab, capture_screenshot, wait_for_load, wait
            )
            browser_available = True
        except ImportError:
            browser_available = False

        for i, scene in enumerate(scenes):
            url = scene.get("url")
            frame_path = str(frames_dir / f"scene_{i:03d}.png")

            print(f"  [slideshow] Capturing scene {i+1}/{len(scenes)}")

            if browser_available and url:
                try:
                    if i == 0:
                        new_tab(url)
                    else:
                        goto_url(url)
                    wait_for_load(timeout=15)
                    wait(2)
                    capture_screenshot(frame_path)
                except Exception as e:
                    print(f"  [slideshow] WARNING: Capture failed for scene {i+1}: {e}")
                    self._create_placeholder_frame(frame_path, scene.get("narration", "")[:50])
            else:
                self._create_placeholder_frame(frame_path, scene.get("narration", "")[:50])

        self.artifacts["frame_dirs"].append(str(frames_dir))

    def _create_placeholder_frame(self, path, text):
        """Create a placeholder frame with text when no browser is available."""
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=#1a1a2e:s=1920x1080:d=1",
            "-frames:v", "1",
            "-vf", f"drawtext=text='{text.replace(chr(39), '')}':fontsize=48:"
                   f"fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2",
            path
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        except Exception:
            pass

    def _step_compose(self, **kwargs):
        """Compose the final video from all artifacts."""
        # Determine video source
        video_source = None
        if self.artifacts["video_segments"]:
            video_source = self.artifacts["video_segments"][0]
        elif self.artifacts["frame_dirs"]:
            # Create slideshow from frames
            frames_dir = self.artifacts["frame_dirs"][0]
            video_source = str(VIDEO_OUTPUT_DIR / "slideshow.mp4")
            try:
                create_slideshow(frames_dir, output_path=video_source, duration_per_frame=8)
            except Exception as e:
                print(f"  [compose] WARNING: Slideshow creation failed: {e}")

        audio_source = self.artifacts.get("combined_audio")
        title = self.plan.get("title", "Untitled")

        if video_source:
            output_path = str(VIDEO_OUTPUT_DIR / "final_video.mp4")
            try:
                result = compose_video(
                    video_path=video_source,
                    audio_path=audio_source,
                    title=title,
                    output_path=output_path,
                    resolution=self.video_resolution,
                )
                self.artifacts["final_video"] = result
            except Exception as e:
                print(f"  [compose] WARNING: Video composition failed: {e}")
                # Try simpler mux
                if video_source and audio_source:
                    self._mux_audio_video(video_source, audio_source, output_path)
                    self.artifacts["final_video"] = output_path
                elif video_source:
                    self.artifacts["final_video"] = video_source
        elif audio_source:
            # Audio-only → create video with black screen + audio
            output_path = str(VIDEO_OUTPUT_DIR / "final_video.mp4")
            editor = VideoEditor()
            editor.add_audio(audio_source)
            result = editor.render(output_path=output_path, resolution=self.video_resolution)
            self.artifacts["final_video"] = result

        if not self.artifacts.get("final_video"):
            raise RuntimeError("Failed to compose any video output")

        print(f"  [compose] Final video: {self.artifacts['final_video']}")

    def _step_publish(self, **kwargs):
        """Publish the final video to configured platforms."""
        video_path = self.artifacts.get("final_video")
        if not video_path:
            raise RuntimeError("No final video to publish")

        publishing = self.plan.get("publishing", {})
        published_to = []

        # YouTube
        if "youtube" in publishing and os.environ.get("YOUTUBE_ENABLED"):
            try:
                result = self._publish_youtube(video_path, publishing["youtube"])
                published_to.append({"platform": "youtube", "result": result})
            except Exception as e:
                print(f"  [publish] YouTube failed: {e}")
                published_to.append({"platform": "youtube", "error": str(e)})

        # TikTok
        if "tiktok" in publishing and os.environ.get("TIKTOK_ENABLED"):
            try:
                result = self._publish_tiktok(video_path, publishing["tiktok"])
                published_to.append({"platform": "tiktok", "result": result})
            except Exception as e:
                print(f"  [publish] TikTok failed: {e}")
                published_to.append({"platform": "tiktok", "error": str(e)})

        # X (Twitter)
        if "x" in publishing and os.environ.get("X_ENABLED"):
            try:
                result = self._publish_x(video_path, publishing["x"])
                published_to.append({"platform": "x", "result": result})
            except Exception as e:
                print(f"  [publish] X/Twitter failed: {e}")
                published_to.append({"platform": "x", "error": str(e)})

        self.artifacts["published_to"] = published_to

    def _publish_youtube(self, video_path, config):
        """Upload video to YouTube Studio via browser automation.

        Requires the user to be logged into YouTube in their Chrome profile.
        See agent-workspace/domain-skills/youtube/upload.md for the full flow.
        """
        from browser_harness.helpers import (
            goto_url, new_tab, wait_for_load, wait, click_at_xy,
            type_text, fill_input, upload_file, page_info, js
        )

        # Navigate to YouTube Studio upload
        new_tab("https://studio.youtube.com/channel/UC/videos/upload?filter=%5B%5D&sort=%7B%22columnType%22%3A%22date%22%2C%22sortOrder%22%3A%22DESCENDING%22%7D")
        wait_for_load(timeout=20)
        wait(3)

        # Click the upload button / select files
        upload_file('input[type="file"]', video_path)
        wait(10)  # Processing time

        # Fill in title
        title = config.get("title", self.plan.get("title", ""))
        try:
            fill_input('#textbox[aria-label="Add a title that describes your video"]', title)
        except Exception:
            # Try alternative selectors
            js(f'document.querySelector("[id=\'textbox\']").textContent = {json.dumps(title)}')

        # Fill in description
        description = config.get("description", "")
        if description:
            try:
                fill_input('#textbox[aria-label="Tell viewers about your video"]', description)
            except Exception:
                pass

        # Set visibility
        visibility = config.get("visibility", "public")
        # Navigate to visibility tab
        try:
            js("document.querySelector('[tab-headers] :nth-child(3)').click()")
            wait(1)
            if visibility == "public":
                js("document.querySelector('[label=\"Public\"] radio').click()")
            elif visibility == "unlisted":
                js("document.querySelector('[label=\"Unlisted\"] radio').click()")
            elif visibility == "private":
                js("document.querySelector('[label=\"Private\"] radio').click()")
        except Exception:
            pass

        # Click publish
        try:
            js("document.querySelector('#publish-button').click()")
        except Exception:
            # Try the done button for scheduled/draft
            js("document.querySelector('#done-button').click()")

        wait(3)
        return {"status": "uploaded", "title": title}

    def _publish_tiktok(self, video_path, config):
        """Upload video to TikTok via browser automation.

        Requires the user to be logged into TikTok in their Chrome profile.
        See agent-workspace/domain-skills/tiktok/upload.md for the full flow.
        """
        from browser_harness.helpers import (
            goto_url, wait_for_load, wait, click_at_xy,
            type_text, upload_file, page_info, js, press_key
        )

        # Navigate to TikTok Studio upload
        goto_url("https://www.tiktok.com/tiktokstudio/upload?from=upload&lang=en")
        wait_for_load(timeout=20)
        wait(3)

        # Dismiss stale draft banner if present
        try:
            js("""
                var btns = document.querySelectorAll('button');
                for (var b of btns) {
                    if (b.textContent.includes('Discard')) { b.click(); break; }
                }
            """)
            wait(1)
        except Exception:
            pass

        # Upload file
        upload_file('input[type="file"]', video_path)
        wait(12)  # Processing time

        # Set caption
        caption = config.get("caption", self.plan.get("title", ""))
        js("document.querySelector('div[contenteditable=\"true\"][role=\"combobox\"]').focus()")
        press_key("End")
        for _ in range(50):
            press_key("Backspace")
        type_text(caption)
        press_key("Escape")

        # Click Post button
        try:
            js("""
                var btns = document.querySelectorAll('button');
                for (var b of btns) {
                    if (b.textContent.trim() === 'Post') { b.click(); break; }
                }
            """)
        except Exception:
            pass

        wait(5)
        return {"status": "uploaded", "caption": caption}

    def _publish_x(self, video_path, config):
        """Post to X (Twitter) via browser automation.

        For video posts on X, this posts a tweet with a link to the YouTube video
        (since X video upload via browser automation is complex and unreliable).
        If the video was published to YouTube, link to it. Otherwise, post text only.
        """
        from browser_harness.helpers import (
            goto_url, wait_for_load, wait, click_at_xy,
            type_text, js, page_info
        )

        tweet_text = config.get("tweet", self.plan.get("title", ""))

        # Add YouTube link if available
        yt_result = next(
            (p for p in self.artifacts.get("published_to", []) if p.get("platform") == "youtube"),
            None
        )
        if yt_result and yt_result.get("result", {}).get("video_url"):
            tweet_text += f" {yt_result['result']['video_url']}"

        # Navigate to X compose
        goto_url("https://x.com/compose/post")
        wait_for_load(timeout=15)
        wait(2)

        # Find compose box and type
        result = js('''
            var el = document.querySelector("[data-testid='tweetTextarea_0']");
            if (!el) return null;
            var r = el.getBoundingClientRect();
            return JSON.stringify({x: Math.round(r.x + r.width/2), y: Math.round(r.y + r.height/2)});
        ''')

        if result:
            import json as _json
            pos = _json.loads(result)
            click_at_xy(pos["x"], pos["y"])
            type_text(tweet_text)

            # Click Post button
            btn = js('''
                var b = document.querySelector("[data-testid='tweetButtonInline']")
                     || document.querySelector("[data-testid='tweetButton']");
                if (!b) return null;
                var r = b.getBoundingClientRect();
                return JSON.stringify({x: Math.round(r.x + r.width/2), y: Math.round(r.y + r.height/2)});
            ''')
            if btn:
                pos = _json.loads(btn)
                click_at_xy(pos["x"], pos["y"])
                wait(2)

        return {"status": "posted", "text": tweet_text}

    # --- Utility methods ---

    def _get_audio_duration(self, path):
        """Get audio file duration in seconds using ffprobe."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            return float(result.stdout.strip())
        except Exception:
            # Estimate from file size (very rough)
            return 10.0

    def _concatenate_audio_files(self, paths, output_path):
        """Concatenate multiple audio files into one."""
        concat_file = tempfile.mktemp(suffix=".txt")
        with open(concat_file, "w") as f:
            for p in paths:
                f.write(f"file '{p}'\n")

        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_file, "-c", "copy", output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Audio concatenation failed: {result.stderr[-500:]}")

        try:
            os.unlink(concat_file)
        except Exception:
            pass

    def _mux_audio_video(self, video_path, audio_path, output_path):
        """Simple audio-video multiplexing with ffmpeg."""
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Mux failed: {result.stderr[-500:]}")


# --- Module-level convenience ---

def execute_plan(plan=None, plan_path=None, tts_voice="alloy", publish=True,
                 video_resolution="1080p", headless=False):
    """Execute a content plan end-to-end.

    Args:
        plan: Content plan dict (from create_plan).
        plan_path: Path to a saved plan JSON file.
        tts_voice: TTS voice to use.
        publish: Whether to publish after composing.
        video_resolution: Output video resolution.
        headless: Skip browser recording, use slideshow mode.

    Returns:
        Dict with execution results.
    """
    executor = PlanExecutor(
        plan=plan, plan_path=plan_path,
        tts_voice=tts_voice, video_resolution=video_resolution
    )
    return executor.execute(publish=publish, headless=headless)
