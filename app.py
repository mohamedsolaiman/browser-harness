#!/usr/bin/env python3
"""Content Automation Studio — Gradio Web UI

Deployed on Hugging Face Spaces. Provides a web interface to:
1. Create content plans (AI scripts your video)
2. Generate TTS voiceover
3. Download stock video footage (Pexels/Pixabay) or generate AI images
4. Compose dynamic videos with transitions and voiceover
5. Publish to YouTube, TikTok, X (when enabled)

All API keys stored as HF Space secrets — never exposed.

API: Xiaomi MiMo (OpenAI-compatible)
- Chat: https://api.xiaomimimo.com/v1/chat/completions
- TTS:  https://api.xiaomimimo.com/v1/chat/completions (model: mimo-v2-tts)

Stock Video: Pexels API + Pixabay API (free API keys)
Image Generation: Pollinations.ai (free, no API key — fallback)
"""

import os
import sys
import json
import time
import subprocess
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr

from dotenv_loader import load_secrets
load_secrets()

_modules_loaded = False

def _ensure_modules():
    global _modules_loaded
    if _modules_loaded:
        return
    from planner.planner import ContentPlanner
    from planner.executor import PlanExecutor
    from tts.mimo_tts import MimoTTS, generate_speech
    from video.editor import compose_video, create_dynamic_video, create_stock_video, generate_srt
    from visuals.image_gen import generate_image, generate_scene_images, enhance_prompt
    from visuals.stock_video import search_stock_videos, download_stock_video, get_videos_for_topic
    _modules_loaded = True


OUTPUT_DIR = Path(os.environ.get("CS_OUTPUT_DIR", "/tmp/content-studio"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLANS_DIR = OUTPUT_DIR / "plans"
PLANS_DIR.mkdir(exist_ok=True)
VIDEOS_DIR = OUTPUT_DIR / "videos"
VIDEOS_DIR.mkdir(exist_ok=True)
AUDIO_DIR = OUTPUT_DIR / "audio"
AUDIO_DIR.mkdir(exist_ok=True)
IMAGES_DIR = OUTPUT_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)


def create_content_plan(topic, platforms, duration, style, language, instructions):
    _ensure_modules()
    from planner.planner import ContentPlanner

    if not topic.strip():
        return "❌ Please enter a topic.", None

    platform_list = [p.strip() for p in platforms.split(",") if p.strip()]
    if not platform_list:
        return "❌ Please specify at least one platform.", None

    try:
        planner = ContentPlanner()
        plan = planner.plan(
            topic=topic.strip(),
            platforms=platform_list,
            duration_minutes=int(duration),
            style=style,
            language=language,
            extra_instructions=instructions.strip() or None,
        )

        plan_path = planner.save_plan(plan, output_path=str(PLANS_DIR / f"plan_{int(time.time())}.json"))
        plan["plan_path"] = plan_path

        output = f"""## 📋 Content Plan: {plan.get('title', 'Untitled')}

**Style:** {plan.get('style', 'N/A')} | **Language:** {plan.get('language', 'N/A')} | **Est. Duration:** {plan.get('duration_estimate_seconds', 'N/A')}s

### Scenes ({len(plan.get('scenes', []))})

"""
        for i, scene in enumerate(plan.get("scenes", []), 1):
            output += f"""**Scene {i}** ({scene.get('duration_seconds', '?')}s)
- 🎤 Narration: {scene.get('narration', 'N/A')[:200]}
- 🖼️ Visual: {scene.get('visual_type', 'N/A')} — {scene.get('visual_instructions', 'N/A')[:150]}

"""

        pub = plan.get("publishing", {})
        if pub:
            output += "### 📢 Publishing\n\n"
            for platform, config in pub.items():
                output += f"**{platform.title()}**: "
                if "title" in config:
                    output += f"{config.get('title', '')}"
                if "caption" in config:
                    output += f"{config.get('caption', '')}"
                if "tweet" in config:
                    output += f"{config.get('tweet', '')}"
                output += "\n"

        output += f"\n💾 Plan saved to: `{plan_path}`"
        return output, plan_path

    except Exception as e:
        error_msg = f"❌ Planning failed: {e}"
        tb = traceback.format_exc()
        if len(tb) > 500:
            tb = tb[-500:]
        return f"{error_msg}\n\n<details><summary>Error Details</summary>\n\n```\n{tb}\n```\n</details>", None


def generate_tts_audio(text, voice, speed):
    _ensure_modules()
    from tts.mimo_tts import MimoTTS

    if not text.strip():
        return "❌ Please enter text to convert to speech.", None

    try:
        client = MimoTTS()
        output_path = str(AUDIO_DIR / f"tts_{int(time.time())}.wav")
        path = client.generate(
            text=text.strip(),
            voice=voice,
            output_path=output_path,
            speed=float(speed),
        )
        return f"✅ Audio generated successfully!\n\n📁 File: `{path}`", path
    except Exception as e:
        return f"❌ TTS failed: {e}\n\nMake sure MIMO_API_KEY is set in your Space secrets.", None


def execute_full_pipeline(topic, platforms, duration, style, language, instructions,
                          tts_voice, resolution, visual_style, visual_mode,
                          enable_publish):
    _ensure_modules()
    from planner.planner import ContentPlanner
    from planner.executor import PlanExecutor

    if not topic.strip():
        yield "❌ Please enter a topic.", None, None, None
        return

    platform_list = [p.strip() for p in platforms.split(",") if p.strip()]

    try:
        # Step 1: Plan
        yield (
            "🧠 **Step 1/4: Creating content plan...**\n\n"
            "Connecting to MiMo AI to generate script and visual directions...\n\n"
            "⏳ This may take 30-60 seconds depending on the topic.",
            None, None, None
        )

        planner = ContentPlanner()
        plan = planner.plan(
            topic=topic.strip(),
            platforms=platform_list,
            duration_minutes=int(duration),
            style=style,
            language=language,
            extra_instructions=instructions.strip() or None,
        )

        scenes = plan.get("scenes", [])
        plan_summary = f"**{plan.get('title', 'Untitled')}** — {len(scenes)} scenes, ~{plan.get('duration_estimate_seconds', '?')}s"

        # Save plan
        plan_path = planner.save_plan(plan, output_path=str(PLANS_DIR / f"plan_{int(time.time())}.json"))
        plan["plan_path"] = plan_path

        # Step 2: TTS
        yield (
            f"🎙️ **Step 2/4: Generating voiceover...**\n\n"
            f"Plan: {plan_summary}\n\n"
            f"Generating TTS audio for {len(scenes)} scenes using MiMo TTS...\n\n"
            f"⏳ Each scene takes ~10 seconds.",
            None, None, None
        )

        executor = PlanExecutor(
            plan=plan,
            tts_voice=tts_voice,
            video_resolution=resolution,
            visual_style=visual_style,
            visual_mode=visual_mode,
        )

        # Run TTS step
        try:
            executor._step_tts()
            audio_ok = len(executor.artifacts.get("audio_files", []))
            yield (
                f"✅ TTS complete: {audio_ok}/{len(scenes)} audio files generated\n\n"
                f"Moving to visual generation...",
                None, None, None
            )
        except Exception as e:
            yield (
                f"⚠️ TTS step had issues: {e}\n\n"
                f"Continuing with visual generation...",
                None, None, None
            )

        # Step 3: Generate visuals (stock videos / AI images)
        visual_mode_label = {
            "stock_videos": "Stock Videos (Pexels/Pixabay — dynamic footage)",
            "ai_images": "AI-Generated Images (Pollinations.ai — static with Ken Burns)",
            "ai_plus_stock": "Stock Videos + AI Images (best of both)",
        }.get(visual_mode, visual_mode)

        yield (
            f"🎬 **Step 3/4: Generating visuals...**\n\n"
            f"Plan: {plan_summary}\n\n"
            f"Mode: {visual_mode_label}\n"
            f"Style: {visual_style}\n\n"
            f"{'Downloading stock footage for' if 'stock' in visual_mode else 'Generating AI images for'} {len(scenes)} scenes...\n\n"
            f"⏳ {'Each video takes ~10-20 seconds to download' if 'stock' in visual_mode else 'Each image takes ~15-30 seconds'}.",
            None, None, None
        )

        try:
            executor._step_visuals()
            img_ok = sum(1 for p in executor.artifacts.get("image_files", []) if p is not None)
            vid_ok = sum(1 for p in executor.artifacts.get("stock_videos", []) if p is not None)
            if vid_ok > 0:
                visual_info = f"Stock videos: {vid_ok}/{len(scenes)}"
                if img_ok > 0:
                    visual_info += f" | AI images: {img_ok} (fallback)"
            else:
                visual_info = f"AI images: {img_ok}/{len(scenes)}"
            yield (
                f"✅ Visuals complete: {visual_info}\n\n"
                f"Moving to video composition...",
                None, None, None
            )
        except Exception as e:
            yield (
                f"⚠️ Visual generation had issues: {e}\n\n"
                f"Continuing with available visuals...",
                None, None, None
            )

        # Step 4: Compose
        yield (
            f"🎬 **Step 4/4: Composing final video...**\n\n"
            f"Plan: {plan_summary}\n\n"
            f"Applying Ken Burns effects, crossfade transitions, and text overlays using ffmpeg...\n\n"
            f"⏳ This takes 1-3 minutes depending on video length.",
            None, None, None
        )

        try:
            executor._step_compose()
        except Exception as e:
            yield (
                f"⚠️ Dynamic composition failed: {e}\n\nTrying simpler composition...",
                None, None, None
            )
            try:
                executor._fallback_compose()
            except Exception as e2:
                yield (
                    f"❌ Video composition failed completely: {e2}\n\n"
                    f"```\n{traceback.format_exc()[-800:]}\n```",
                    None, None, None
                )
                return

        # Publishing
        published_info = ""
        if enable_publish:
            yield "🚀 **Publishing...**\n\nUploading to platforms...", None, None, None
            try:
                executor._step_publish()
                published = executor.artifacts.get("published_to", [])
                if published:
                    for p in published:
                        platform = p.get("platform", "unknown")
                        if "error" in p:
                            published_info += f"\n⚠️ {platform}: {p['error']}"
                        else:
                            published_info += f"\n✅ {platform}: Published!"
                else:
                    published_info = "\n⚪ No platforms were enabled for publishing."
            except Exception as e:
                published_info = f"\n⚠️ Publishing failed: {e}"

        # Prepare results
        final_video = executor.artifacts.get("final_video")
        combined_audio = executor.artifacts.get("combined_audio")
        subtitle_file = executor.artifacts.get("subtitle_file")

        result_text = f"""## ✅ Content Pipeline Complete!

### 📋 Plan
{plan_summary}

### 🎬 Output
- **Video**: `{final_video or 'N/A'}`
- **Audio**: `{combined_audio or 'N/A'}`
- **Subtitles**: `{subtitle_file or 'N/A'}`
{published_info}

### 📝 Scenes
"""
        for i, scene in enumerate(scenes, 1):
            result_text += f"{i}. {scene.get('narration', 'N/A')[:80]}...\n"

        yield result_text, final_video, combined_audio, subtitle_file

    except Exception as e:
        error_msg = f"❌ Pipeline failed: {e}"
        tb = traceback.format_exc()
        if len(tb) > 500:
            tb = tb[-500:]
        yield f"{error_msg}\n\n<details><summary>Error Details</summary>\n\n```\n{tb}\n```\n</details>", None, None, None


def load_and_show_plan(plan_path):
    if not plan_path:
        return "❌ No plan file specified."
    try:
        with open(plan_path) as f:
            plan = json.load(f)
        return f"```json\n{json.dumps(plan, indent=2, ensure_ascii=False)[:3000]}\n```"
    except Exception as e:
        return f"❌ Failed to load plan: {e}"


def list_saved_plans():
    plans = sorted(PLANS_DIR.glob("*.json"), reverse=True)
    if not plans:
        return "No saved plans yet."
    output = "### Saved Plans\n\n"
    for p in plans[:20]:
        try:
            with open(p) as f:
                plan = json.load(f)
            title = plan.get("title", "Untitled")
            status = plan.get("status", "draft")
            created = plan.get("created_at", "unknown")
            output += f"- **{title}** ({status}) — {created} — `{p}`\n"
        except Exception:
            output += f"- *(corrupted)* — `{p}`\n"
    return output


def check_status():
    keys = {
        "MIMO_API_KEY": bool(os.environ.get("MIMO_API_KEY")),
        "MIMO_BASE_URL": os.environ.get("MIMO_BASE_URL", "not set (using default)"),
        "PEXELS_API_KEY": bool(os.environ.get("PEXELS_API_KEY")),
        "PIXABAY_API_KEY": bool(os.environ.get("PIXABAY_API_KEY")),
        "YOUTUBE_ENABLED": bool(os.environ.get("YOUTUBE_ENABLED")),
        "TIKTOK_ENABLED": bool(os.environ.get("TIKTOK_ENABLED")),
        "X_ENABLED": bool(os.environ.get("X_ENABLED")),
    }
    output = "### 🔍 System Status\n\n"
    output += f"- **Mimo API Key**: {'✅ Set' if keys['MIMO_API_KEY'] else '❌ Missing — set MIMO_API_KEY in Space Secrets'}\n"
    output += f"- **Mimo Base URL**: `{keys['MIMO_BASE_URL']}`\n"
    output += f"- **Pexels API Key**: {'✅ Set — stock videos available' if keys['PEXELS_API_KEY'] else '⚪ Not set'}\n"
    output += f"- **Pixabay API Key**: {'✅ Set — more stock videos available' if keys['PIXABAY_API_KEY'] else '⚪ Not set'}\n"
    if not keys['PEXELS_API_KEY'] and not keys['PIXABAY_API_KEY']:
        output += "  ⚠️ **Neither key set — will fall back to AI images (static). Get free keys at [Pexels](https://pexels.com/api) and/or [Pixabay](https://pixabay.com/api/docs)**\n"
    output += f"- **YouTube Publishing**: {'✅ Enabled' if keys['YOUTUBE_ENABLED'] else '⚪ Disabled'}\n"
    output += f"- **TikTok Publishing**: {'✅ Enabled' if keys['TIKTOK_ENABLED'] else '⚪ Disabled'}\n"
    output += f"- **X/Twitter Publishing**: {'✅ Enabled' if keys['X_ENABLED'] else '⚪ Disabled'}\n"
    output += f"\n**Output Directory**: `{OUTPUT_DIR}`"

    # Quick connectivity test
    try:
        import requests
        base_url = os.environ.get("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
        api_key = os.environ.get("MIMO_API_KEY", "")
        if api_key:
            test_url = f"{base_url.rstrip('/')}/models"
            resp = requests.get(
                test_url,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("id", "?") for m in data.get("data", [])[:5]]
                output += f"\n\n**API Connection**: ✅ Working! Available models: {', '.join(models)}"
            else:
                output += f"\n\n**API Connection**: ❌ Failed — HTTP {resp.status_code}"
        else:
            output += "\n\n**API Connection**: ⚠️ No API key set, skipping test."
    except Exception as e:
        output += f"\n\n**API Connection**: ❌ Failed — {str(e)[:150]}"

    # Check ffmpeg
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.split("\n")[0]
            output += f"\n\n**FFmpeg**: ✅ {version}"
        else:
            output += "\n\n**FFmpeg**: ❌ Not working"
    except Exception:
        output += "\n\n**FFmpeg**: ❌ Not found"

    # Check Pollinations.ai
    try:
        import requests
        resp = requests.head(
            "https://image.pollinations.ai/prompt/test?width=64&height=64&nologo=true",
            timeout=10,
        )
        output += f"\n\n**Pollinations.ai**: ✅ Reachable (free AI image generation)"
    except Exception:
        output += "\n\n**Pollinations.ai**: ⚠️ Could not verify connectivity"

    return output


# ──────────────────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────────────────

CSS = """
.gradio-container { max-width: 1100px !important; }
.plan-output { min-height: 300px; }
footer { display: none !important; }
.video-preview { max-width: 100%; }
"""

with gr.Blocks(
    title="Content Automation Studio",
    css=CSS,
    theme=gr.themes.Soft(primary_hue="emerald", secondary_hue="slate"),
) as app:

    gr.Markdown("""
    # 🎬 Content Automation Studio
    **AI-powered video creation with stock footage, voiceover, and dynamic composition.**

    > 🎥 **Stock Videos** by [Pexels](https://pexels.com/api) + [Pixabay](https://pixabay.com/api/docs) (free API keys) •
    > 🖼️ **AI Images** by [Pollinations.ai](https://pollinations.ai) (free, fallback) •
    > 🎙️ **Voiceover** by Xiaomi MiMo TTS

    Create stunning videos in 4 steps: **Plan → TTS → Dynamic Visuals → Compose**
    """)

    with gr.Tabs():
        # ── Tab 1: Full Pipeline ──
        with gr.Tab("🚀 Full Pipeline"):
            gr.Markdown("### End-to-end: Plan → TTS → Stock Footage/AI Visuals → Dynamic Video")
            with gr.Row():
                with gr.Column(scale=2):
                    pipe_topic = gr.Textbox(
                        label="📹 Video Topic",
                        placeholder="e.g. Python decorators explained for beginners",
                        lines=2,
                    )
                    pipe_platforms = gr.Textbox(
                        label="Platforms",
                        value="youtube",
                        placeholder="youtube, tiktok, x",
                    )
                    with gr.Row():
                        pipe_duration = gr.Slider(1, 10, value=3, step=1, label="Duration (minutes)")
                        pipe_style = gr.Dropdown(
                            ["tutorial", "explainer", "review", "demo", "news"],
                            value="tutorial", label="Content Style"
                        )
                        pipe_language = gr.Dropdown(
                            ["en", "es", "fr", "de", "ar", "zh", "ja", "ko", "pt", "hi"],
                            value="en", label="Language"
                        )
                    pipe_instructions = gr.Textbox(
                        label="Extra Instructions (optional)",
                        placeholder="e.g. Focus on practical examples, use analogies...",
                        lines=2,
                    )
                    with gr.Row():
                        pipe_voice = gr.Dropdown(
                            ["mimo_default", "default_zh", "default_en"],
                            value="mimo_default", label="TTS Voice"
                        )
                        pipe_resolution = gr.Dropdown(
                            ["720p", "1080p", "4k"],
                            value="720p", label="Resolution"
                        )
                    with gr.Row():
                        pipe_visual_style = gr.Dropdown(
                            ["cinematic", "corporate", "tech_code", "nature", "abstract"],
                            value="cinematic", label="🖼️ Visual Style"
                        )
                        pipe_visual_mode = gr.Dropdown(
                            ["stock_videos", "ai_plus_stock", "ai_images"],
                            value="stock_videos", label="🎥 Visual Source"
                        )
                    pipe_publish = gr.Checkbox(value=False, label="Enable Publishing (requires browser session)")

                    pipe_btn = gr.Button("🚀 Run Full Pipeline", variant="primary", size="lg")

                with gr.Column(scale=3):
                    pipe_output = gr.Markdown(label="Progress & Results", elem_classes=["plan-output"])
                    pipe_video_player = gr.Video(label="📹 Final Video Preview", elem_classes=["video-preview"])
                    with gr.Row():
                        pipe_video = gr.File(label="📹 Download Video")
                        pipe_audio = gr.File(label="🎙️ Audio")
                        pipe_srt = gr.File(label="📝 Subtitles")

            pipe_btn.click(
                fn=execute_full_pipeline,
                inputs=[
                    pipe_topic, pipe_platforms, pipe_duration, pipe_style,
                    pipe_language, pipe_instructions, pipe_voice, pipe_resolution,
                    pipe_visual_style, pipe_visual_mode, pipe_publish,
                ],
                outputs=[pipe_output, pipe_video_player, pipe_audio, pipe_srt],
            )

        # ── Tab 2: Plan Only ──
        with gr.Tab("📋 Plan Content"):
            gr.Markdown("### Create an AI content plan (review before executing)")
            with gr.Row():
                with gr.Column():
                    plan_topic = gr.Textbox(
                        label="Video Topic",
                        placeholder="e.g. How Docker containers work under the hood",
                        lines=2,
                    )
                    plan_platforms = gr.Textbox(label="Platforms", value="youtube")
                    with gr.Row():
                        plan_duration = gr.Slider(1, 10, value=3, step=1, label="Duration (minutes)")
                        plan_style = gr.Dropdown(
                            ["tutorial", "explainer", "review", "demo", "news"],
                            value="tutorial", label="Style"
                        )
                        plan_language = gr.Dropdown(
                            ["en", "es", "fr", "de", "ar", "zh", "ja", "ko", "pt", "hi"],
                            value="en", label="Language"
                        )
                    plan_instructions = gr.Textbox(
                        label="Extra Instructions (optional)", lines=2,
                    )
                    plan_btn = gr.Button("📋 Generate Plan", variant="primary")

                with gr.Column():
                    plan_output = gr.Markdown(elem_classes=["plan-output"])
                    plan_file = gr.File(label="💾 Plan File (JSON)")

            plan_btn.click(
                fn=create_content_plan,
                inputs=[plan_topic, plan_platforms, plan_duration, plan_style, plan_language, plan_instructions],
                outputs=[plan_output, plan_file],
            )

        # ── Tab 3: TTS Only ──
        with gr.Tab("🎙️ Text to Speech"):
            gr.Markdown("### Generate voiceover audio with MiMo TTS\n> Model: `mimo-v2-tts` | Uses chat completions endpoint with audio output")
            with gr.Row():
                with gr.Column():
                    tts_text = gr.Textbox(
                        label="Text to Speak",
                        placeholder="Enter the text you want converted to speech...",
                        lines=6,
                    )
                    with gr.Row():
                        tts_voice = gr.Dropdown(
                            ["mimo_default", "default_zh", "default_en"],
                            value="mimo_default", label="Voice"
                        )
                        tts_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Speed")
                    tts_btn = gr.Button("🎙️ Generate Speech", variant="primary")

                with gr.Column():
                    tts_output = gr.Markdown()
                    tts_file = gr.File(label="🎧 Audio File")

            tts_btn.click(
                fn=generate_tts_audio,
                inputs=[tts_text, tts_voice, tts_speed],
                outputs=[tts_output, tts_file],
            )

        # ── Tab 4: AI Image Preview ──
        with gr.Tab("🖼️ AI Image Preview"):
            gr.Markdown("### Test AI image generation with Pollinations.ai (free, no API key needed)")
            with gr.Row():
                with gr.Column():
                    img_prompt = gr.Textbox(
                        label="Image Prompt",
                        placeholder="e.g. A dramatic sunset over mountain peaks, cinematic lighting",
                        lines=3,
                    )
                    img_style = gr.Dropdown(
                        ["cinematic", "corporate", "tech_code", "nature", "abstract"],
                        value="cinematic", label="Visual Style"
                    )
                    img_btn = gr.Button("🖼️ Generate Preview Image", variant="primary")

                with gr.Column():
                    img_output = gr.Markdown()
                    img_preview = gr.Image(label="Generated Image", type="filepath")

            def preview_image(prompt, style):
                _ensure_modules()
                from visuals.image_gen import generate_image
                if not prompt.strip():
                    return "❌ Please enter a prompt.", None
                try:
                    path = generate_image(
                        prompt=prompt.strip(),
                        style=style,
                        output_path=str(IMAGES_DIR / f"preview_{int(time.time())}.jpg"),
                    )
                    return f"✅ Image generated!\n\n📁 File: `{path}`", path
                except Exception as e:
                    return f"❌ Image generation failed: {e}", None

            img_btn.click(
                fn=preview_image,
                inputs=[img_prompt, img_style],
                outputs=[img_output, img_preview],
            )

        # ── Tab 5: Saved Plans ──
        with gr.Tab("📂 Saved Plans"):
            gr.Markdown("### Browse and manage saved content plans")
            refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
            plans_list = gr.Markdown()
            plan_path_input = gr.Textbox(label="Plan File Path", placeholder="/tmp/content-studio/plans/plan_xxx.json")
            view_btn = gr.Button("👁️ View Plan")
            plan_view = gr.Markdown()

            refresh_btn.click(fn=list_saved_plans, outputs=[plans_list])
            view_btn.click(fn=load_and_show_plan, inputs=[plan_path_input], outputs=[plan_view])

        # ── Tab 6: Settings ──
        with gr.Tab("⚙️ Settings"):
            gr.Markdown("""
            ### Configuration

            API keys are managed through **Hugging Face Space secrets** (Settings → Secrets):

            | Secret | Required | Description |
            |--------|----------|-------------|
            | `MIMO_API_KEY` | ✅ | MiMo API key for TTS and LLM planning |
            | `MIMO_BASE_URL` | No | API base URL (default: `https://api.xiaomimimo.com/v1`) |
            | `MIMO_TTS_MODEL` | No | TTS model (default: `mimo-v2-tts`) |
            | `PLANNER_MODEL` | No | LLM model for planning (default: `mimo-v2-flash`) |
            | `PEXELS_API_KEY` | Recommended | Pexels API key for stock video (free at pexels.com/api) |
            | `PIXABAY_API_KEY` | Recommended | Pixabay API key for more stock video (free at pixabay.com/api/docs) |
            | `YOUTUBE_ENABLED` | No | Set `1` to enable YouTube publishing |
            | `TIKTOK_ENABLED` | No | Set `1` to enable TikTok publishing |
            | `X_ENABLED` | No | Set `1` to enable X/Twitter publishing |

            ### Visual Sources

            | Mode | Description | API Key Required |
            |------|-------------|------------------|
            | **Stock Videos** | Download dynamic stock footage from Pexels/Pixabay | ✅ Pexels and/or Pixabay key |
            | **AI + Stock** | Stock footage with AI image fallback | Pexels/Pixabay recommended |
            | **AI Images** | Generate cinematic images with Pollinations.ai | ❌ Free, no key (static visuals) |

            **Tip**: For dynamic, professional-looking videos, use **Stock Videos** mode. Get free API keys at:
            - [Pexels API](https://pexels.com/api) — Free, instant signup
            - [Pixabay API](https://pixabay.com/api/docs) — Free, instant signup
            """)

            status_btn = gr.Button("🔍 Check Status & API Connection", variant="primary")
            status_output = gr.Markdown()
            status_btn.click(fn=check_status, outputs=[status_output])

app.launch(server_name="0.0.0.0", server_port=7860)
