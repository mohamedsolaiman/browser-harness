"""Browser Harness Content Automator — Web UI & API.

FastAPI application that provides:
- Web UI for creating video content plans and executing them
- REST API for programmatic access
- Background task execution for the content pipeline

All credentials from environment variables — never hard-coded.
"""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, Form, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import jinja2

# ============================================================
# Configuration
# ============================================================

MIMO_API_KEY = os.environ.get("MIMO_API_KEY", "")
MIMO_BASE_URL = os.environ.get("MIMO_BASE_URL", "https://api.mymimo.ai").rstrip("/")
MIMO_TTS_MODEL = os.environ.get("MIMO_TTS_MODEL", "mimo-tts-1")
PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "gpt-4o-mini")

VIDEO_DIR = Path(os.environ.get("BH_VIDEO_DIR", "/app/output/videos"))
PLAN_DIR = Path(os.environ.get("BH_PLAN_DIR", "/app/output/plans"))
AUDIO_DIR = Path(os.environ.get("BH_AUDIO_DIR", "/app/output/audio"))
FRAME_DIR = Path(os.environ.get("BH_FRAME_DIR", "/app/output/frames"))

for d in [VIDEO_DIR, PLAN_DIR, AUDIO_DIR, FRAME_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# Pipeline State
# ============================================================

pipeline_state = {
    "status": "idle",       # idle, planning, generating_tts, composing, publishing, done, error
    "current_topic": None,
    "progress": 0,
    "message": "Ready",
    "plan": None,
    "artifacts": {},
    "history": [],
    "started_at": None,
    "completed_at": None,
}

# ============================================================
# Core Pipeline Functions
# ============================================================

def update_state(status, message="", progress=0, **kwargs):
    """Update the pipeline state."""
    pipeline_state["status"] = status
    pipeline_state["message"] = message
    pipeline_state["progress"] = progress
    pipeline_state.update(kwargs)


def llm_chat(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    """Send a chat completion request to the LLM."""
    url = f"{MIMO_BASE_URL}/v1/chat/completions"
    body = {
        "model": PLANNER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 4096,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MIMO_API_KEY}",
    }
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=120) as response:
        data = json.loads(response.read())
        return data["choices"][0]["message"]["content"]


def generate_tts(text: str, voice: str = "alloy", output_path: str = None) -> str:
    """Generate TTS audio using Mimo API."""
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(AUDIO_DIR / f"tts_{timestamp}.mp3")

    url = f"{MIMO_BASE_URL}/v1/audio/speech"
    body = {
        "model": MIMO_TTS_MODEL,
        "input": text,
        "voice": voice,
        "response_format": "mp3",
        "speed": 1.0,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MIMO_API_KEY}",
    }

    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=60) as response:
        audio_data = response.read()

    with open(output_path, "wb") as f:
        f.write(audio_data)

    return output_path


def get_audio_duration(path: str) -> float:
    """Get audio file duration in seconds."""
    try:
        cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return float(result.stdout.strip())
    except Exception:
        return 10.0


def create_title_frame(text: str, width: int = 1920, height: int = 1080,
                       output_path: str = None) -> str:
    """Create a title card frame using ffmpeg."""
    if not output_path:
        output_path = str(FRAME_DIR / f"title_{int(time.time())}.png")

    safe_text = text.replace("'", "").replace('"', '').replace(":", "")[:80]
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=#0f0f23:s={width}x{height}:d=1",
        "-frames:v", "1",
        "-vf", f"drawtext=text='{safe_text}':fontsize=64:fontcolor=white:"
               f"x=(w-text_w)/2:y=(h-text_h)/2:"
               f"borderw=4:bordercolor=#6c63ff",
        output_path
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    return output_path


def create_scene_frame(scene_num: int, narration: str, width: int = 1920,
                       height: int = 1080, bg_color: str = "#1a1a2e") -> str:
    """Create a scene frame with narration text."""
    output_path = str(FRAME_DIR / f"scene_{scene_num:03d}_{int(time.time())}.png")

    # Word wrap the narration text
    words = narration.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= 60:
            current_line = (current_line + " " + word).strip()
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    # Build drawtext filter with multiple lines
    filters = []
    for i, line in enumerate(lines[:8]):  # Max 8 lines
        safe_line = line.replace("'", "").replace('"', '').replace(":", "").replace(";", "")
        y_offset = (height // 2) - (len(lines[:8]) * 40) // 2 + i * 60
        filters.append(
            f"drawtext=text='{safe_line}':fontsize=40:fontcolor=white:"
            f"x=(w-text_w)/2:y={y_offset}"
        )

    filter_str = ",".join(filters)
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c={bg_color}:s={width}x{height}:d=1",
        "-frames:v", "1",
        "-vf", filter_str,
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        # Fallback: simple frame
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:d=1",
            "-frames:v", "1", output_path
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=10)

    return output_path


def create_slideshow_video(frame_paths: list, audio_path: str = None,
                          output_path: str = None, fps: int = 24,
                          duration_per_frame: float = 8.0) -> str:
    """Create a slideshow video from frame images."""
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(VIDEO_DIR / f"slideshow_{timestamp}.mp4")

    # Create concat file
    concat_file = tempfile.mktemp(suffix=".txt")
    with open(concat_file, "w") as f:
        for fp in frame_paths:
            f.write(f"file '{fp}'\n")
            f.write(f"duration {duration_per_frame}\n")
        if frame_paths:
            f.write(f"file '{frame_paths[-1]}'\n")

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

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    try:
        os.unlink(concat_file)
    except Exception:
        pass

    if result.returncode != 0:
        raise RuntimeError(f"Slideshow creation failed: {result.stderr[-500:]}")

    return output_path


def mux_audio_video(video_path: str, audio_path: str, output_path: str = None) -> str:
    """Mux audio and video together."""
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(VIDEO_DIR / f"final_{timestamp}.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"Mux failed: {result.stderr[-500:]}")

    return output_path


def generate_srt(segments: list, output_path: str = None) -> str:
    """Generate an SRT subtitle file."""
    if not output_path:
        output_path = str(VIDEO_DIR / f"subs_{int(time.time())}.srt")

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


# ============================================================
# Content Plan Creation
# ============================================================

def create_content_plan(topic: str, platforms: list = None,
                       duration_minutes: int = 3, style: str = "tutorial",
                       language: str = "en") -> dict:
    """Create a full content plan using the LLM."""
    if platforms is None:
        platforms = ["youtube"]

    system_prompt = f"""You are an expert content strategist and video producer. Create a detailed, actionable content plan for automated video production.

Output your plan as a JSON object with this exact structure:
{{
  "title": "Compelling video title",
  "description": "Video description for YouTube (2-3 sentences)",
  "style": "{style}",
  "language": "{language}",
  "duration_estimate_seconds": {duration_minutes * 60},
  "scenes": [
    {{
      "scene_number": 1,
      "duration_seconds": 20,
      "narration": "Engaging voiceover text for this scene. Clear, informative, and conversational.",
      "visual_type": "title_card",
      "visual_instructions": "Show the video title with a modern gradient background",
      "url": null
    }},
    {{
      "scene_number": 2,
      "duration_seconds": 30,
      "narration": "Detailed explanation that provides real value to the viewer.",
      "visual_type": "content_card",
      "visual_instructions": "Show key points and code examples",
      "url": null
    }}
  ],
  "publishing": {{
    "youtube": {{
      "title": "YouTube-optimized title",
      "description": "YouTube description with keywords and links",
      "tags": ["tag1", "tag2", "tag3"],
      "category": "Education",
      "visibility": "public"
    }},
    "tiktok": {{
      "caption": "Short TikTok caption #hashtag1 #hashtag2"
    }},
    "x": {{
      "tweet": "Tweet text with link #hashtag"
    }}
  }}
}}

IMPORTANT RULES:
- Return ONLY valid JSON. No markdown fences, no commentary.
- Each scene should be 15-45 seconds long.
- Narration should be engaging, informative, and conversational.
- Include 4-8 scenes for a {duration_minutes}-minute video.
- Make the title SEO-friendly and attention-grabbing.
- Tags should be relevant and include both broad and specific terms."""

    user_prompt = f"""Create a content plan for a {style} video about: {topic}

Target duration: {duration_minutes} minutes
Target platforms: {', '.join(platforms)}
Language: {language}

Generate the complete content plan as JSON."""

    response = llm_chat(system_prompt, user_prompt)

    # Parse JSON from response
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        start = 1
        end = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith("```"):
                end = i
                break
        text = "\n".join(lines[start:end])

    try:
        plan = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            plan = json.loads(text[start:end])
        else:
            raise ValueError("Could not parse LLM response as JSON")

    plan["created_at"] = datetime.now().isoformat()
    plan["topic"] = topic
    plan["target_platforms"] = platforms
    plan["status"] = "draft"

    return plan


# ============================================================
# Full Pipeline Execution
# ============================================================

def run_pipeline(topic: str, platforms: list = None, style: str = "tutorial",
                 duration_minutes: int = 3, tts_voice: str = "alloy",
                 language: str = "en"):
    """Run the full content creation pipeline."""
    try:
        # Step 1: Plan
        update_state("planning", "AI is planning your content...", 10,
                     current_topic=topic, started_at=datetime.now().isoformat())

        plan = create_content_plan(topic, platforms, duration_minutes, style, language)

        # Save plan
        plan_path = str(PLAN_DIR / f"plan_{int(time.time())}.json")
        with open(plan_path, "w") as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)

        update_state("planning", f"Plan created: {plan.get('title', 'Untitled')}", 20,
                     plan=plan)

        # Step 2: Generate TTS for each scene
        update_state("generating_tts", "Generating voiceover audio...", 30)

        audio_files = []
        subtitle_segments = []
        current_time = 0

        scenes = plan.get("scenes", [])
        for i, scene in enumerate(scenes):
            narration = scene.get("narration", "")
            if not narration:
                continue

            progress = 30 + int((i / max(len(scenes), 1)) * 30)
            update_state("generating_tts",
                        f"Generating audio for scene {i+1}/{len(scenes)}...", progress)

            try:
                audio_path = generate_tts(narration, voice=tts_voice,
                                          output_path=str(AUDIO_DIR / f"scene_{i:03d}.mp3"))
                audio_files.append(audio_path)

                duration = get_audio_duration(audio_path)
                subtitle_segments.append({
                    "start": current_time,
                    "end": current_time + duration,
                    "text": narration,
                })
                current_time += duration
            except Exception as e:
                # Estimate duration from text length
                words = len(narration.split())
                estimated = (words / 150) * 60
                subtitle_segments.append({
                    "start": current_time,
                    "end": current_time + estimated,
                    "text": narration,
                })
                current_time += estimated

        # Concatenate all audio
        combined_audio = None
        if audio_files:
            combined_audio = str(AUDIO_DIR / "full_narration.mp3")
            if len(audio_files) > 1:
                concat_file = tempfile.mktemp(suffix=".txt")
                with open(concat_file, "w") as f:
                    for af in audio_files:
                        f.write(f"file '{af}'\n")
                cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                       "-i", concat_file, "-c", "copy", combined_audio]
                subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                try:
                    os.unlink(concat_file)
                except Exception:
                    pass
            else:
                shutil.copy2(audio_files[0], combined_audio)

        # Generate subtitles
        if subtitle_segments:
            srt_path = generate_srt(subtitle_segments)

        # Step 3: Create visual frames for each scene
        update_state("composing", "Creating video visuals...", 65)

        frame_paths = []

        # Title frame
        title = plan.get("title", "Untitled")
        title_frame = create_title_frame(title)
        frame_paths.append(title_frame)

        # Scene frames
        for i, scene in enumerate(scenes):
            narration = scene.get("narration", "")
            visual_type = scene.get("visual_type", "content_card")

            bg_colors = ["#1a1a2e", "#16213e", "#0f3460", "#1b1b2f", "#162447"]
            bg_color = bg_colors[i % len(bg_colors)]

            frame_path = create_scene_frame(i + 1, narration, bg_color=bg_color)
            frame_paths.append(frame_path)

        update_state("composing", "Assembling final video...", 80)

        # Step 4: Compose the video
        if frame_paths and combined_audio:
            # Get total audio duration for calculating frame duration
            total_audio_duration = get_audio_duration(combined_audio)
            duration_per_frame = total_audio_duration / len(frame_paths) if frame_paths else 8.0
            duration_per_frame = max(3.0, min(duration_per_frame, 15.0))

            # Create slideshow video
            slideshow_path = create_slideshow_video(
                frame_paths, combined_audio,
                output_path=str(VIDEO_DIR / "slideshow.mp4"),
                duration_per_frame=duration_per_frame
            )

            # Burn in subtitles if available
            final_video = str(VIDEO_DIR / "final_video.mp4")
            if subtitle_segments:
                try:
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", slideshow_path,
                        "-vf", f"subtitles={srt_path}:force_style='FontSize=22,PrimaryColour=&Hffffff&,OutlineColour=&H40000000&,Outline=2'",
                        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
                        "-pix_fmt", "yuv420p",
                        "-c:a", "copy",
                        final_video
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                    if result.returncode != 0:
                        # Subtitle burn failed, just copy
                        shutil.copy2(slideshow_path, final_video)
                except Exception:
                    shutil.copy2(slideshow_path, final_video)
            else:
                shutil.copy2(slideshow_path, final_video)

        elif combined_audio:
            # Audio only → create video with black screen
            final_video = str(VIDEO_DIR / "final_video.mp4")
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "color=c=black:s=1920x1080:r=24",
                "-i", combined_audio,
                "-c:v", "libx264", "-preset", "medium", "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest", final_video
            ]
            subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        else:
            final_video = None

        # Step 5: Publishing info
        update_state("publishing", "Preparing publishing metadata...", 90)

        artifacts = {
            "final_video": final_video,
            "combined_audio": combined_audio,
            "plan": plan_path,
            "subtitle_file": str(VIDEO_DIR / f"subs_{int(time.time())}.srt") if subtitle_segments else None,
            "frames_count": len(frame_paths),
            "audio_duration": current_time,
        }

        # Note: Actual browser-based publishing requires a logged-in Chrome session
        # On the cloud, we provide the video file and all publishing metadata ready to upload
        published_to = []
        for platform in (platforms or []):
            published_to.append({
                "platform": platform,
                "status": "video_ready",
                "message": f"Video ready for {platform}. Use the download link and upload manually, "
                           f"or configure a logged-in browser session.",
                "metadata": plan.get("publishing", {}).get(platform, {})
            })

        artifacts["published_to"] = published_to

        # Done!
        update_state("done", f"Video created: {plan.get('title', 'Untitled')}", 100,
                     artifacts=artifacts,
                     completed_at=datetime.now().isoformat())

        # Add to history
        pipeline_state["history"].append({
            "topic": topic,
            "title": plan.get("title"),
            "created_at": datetime.now().isoformat(),
            "video_path": final_video,
            "platforms": platforms or [],
            "status": "completed",
        })

        return {
            "success": True,
            "plan": plan,
            "artifacts": artifacts,
        }

    except Exception as e:
        update_state("error", f"Pipeline failed: {str(e)}", 0,
                     completed_at=datetime.now().isoformat())
        return {
            "success": False,
            "error": str(e),
        }


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(title="Browser Harness Content Automator")

# Template engine
template_dir = Path(__file__).parent / "templates"
templates = jinja2.Environment(loader=jinja2.FileSystemLoader(str(template_dir)))


@app.get("/", response_class=HTMLResponse)
async def index():
    """Render the main web UI."""
    template = templates.get_template("index.html")
    return template.render(
        state=pipeline_state,
        mimo_configured=bool(MIMO_API_KEY),
    )


@app.get("/api/status")
async def get_status():
    """Get current pipeline status."""
    return pipeline_state


@app.post("/api/create")
async def create_content(
    background_tasks: BackgroundTasks,
    topic: str = Form(...),
    platforms: str = Form("youtube"),
    style: str = Form("tutorial"),
    duration: int = Form(3),
    tts_voice: str = Form("alloy"),
    language: str = Form("en"),
):
    """Start the content creation pipeline."""
    if pipeline_state["status"] not in ("idle", "done", "error"):
        return JSONResponse(
            {"error": f"Pipeline is busy: {pipeline_state['status']}"},
            status_code=409
        )

    platform_list = [p.strip() for p in platforms.split(",")]

    background_tasks.add_task(
        run_pipeline,
        topic=topic,
        platforms=platform_list,
        style=style,
        duration_minutes=duration,
        tts_voice=tts_voice,
        language=language,
    )

    return {"message": "Pipeline started", "topic": topic, "platforms": platform_list}


@app.post("/api/plan")
async def plan_only(topic: str = Form(...), style: str = Form("tutorial"),
                    duration: int = Form(3), language: str = Form("en")):
    """Create a content plan without executing it."""
    try:
        plan = create_content_plan(topic, style=style,
                                   duration_minutes=duration, language=language)
        return {"success": True, "plan": plan}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/videos")
async def list_videos():
    """List available videos."""
    videos = []
    for f in sorted(VIDEO_DIR.glob("*.mp4"), reverse=True):
        stat = f.stat()
        videos.append({
            "name": f.name,
            "path": str(f),
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        })
    return videos


@app.get("/api/plans")
async def list_plans():
    """List saved content plans."""
    plans = []
    for f in sorted(PLAN_DIR.glob("*.json"), reverse=True):
        try:
            with open(f) as fh:
                plan = json.load(fh)
            plans.append({
                "name": f.name,
                "path": str(f),
                "title": plan.get("title", "Untitled"),
                "topic": plan.get("topic", ""),
                "created_at": plan.get("created_at", ""),
                "scenes_count": len(plan.get("scenes", [])),
            })
        except Exception:
            pass
    return plans


@app.get("/download/video/{filename}")
async def download_video(filename: str):
    """Download a video file."""
    path = VIDEO_DIR / filename
    if not path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(str(path), media_type="video/mp4", filename=filename)


@app.get("/download/plan/{filename}")
async def download_plan(filename: str):
    """Download a plan JSON file."""
    path = PLAN_DIR / filename
    if not path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(str(path), media_type="application/json", filename=filename)


@app.post("/api/reset")
async def reset_pipeline():
    """Reset the pipeline state."""
    pipeline_state.update({
        "status": "idle",
        "current_topic": None,
        "progress": 0,
        "message": "Ready",
        "plan": None,
        "artifacts": {},
        "started_at": None,
        "completed_at": None,
    })
    return {"message": "Pipeline reset"}


@app.get("/api/voices")
async def get_voices():
    """List available TTS voices."""
    return [
        {"id": "alloy", "name": "Alloy", "description": "Balanced, neutral tone"},
        {"id": "echo", "name": "Echo", "description": "Warm, conversational"},
        {"id": "fable", "name": "Fable", "description": "Expressive, storytelling"},
        {"id": "onyx", "name": "Onyx", "description": "Deep, authoritative"},
        {"id": "nova", "name": "Nova", "description": "Energetic, friendly"},
        {"id": "shimmer", "name": "Shimmer", "description": "Clear, professional"},
    ]
