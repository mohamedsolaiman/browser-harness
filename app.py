#!/usr/bin/env python3
"""Content Automation Studio — Gradio Web UI

Deployed on Hugging Face Spaces. Provides a web interface to:
1. Create content plans (AI scripts your video)
2. Generate TTS voiceover
3. Compose videos with browser screenshots/slides
4. Publish to YouTube, TikTok, X (when enabled)

All API keys stored as HF Space secrets — never exposed.

API: Xiaomi MiMo (OpenAI-compatible)
- Chat: https://api.xiaomimimo.com/v1/chat/completions
- TTS:  https://api.xiaomimimo.com/v1/chat/completions (model: mimo-v2-tts)
"""

import os
import sys
import json
import time
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
    from video.editor import VideoEditor, compose_video, create_slideshow, generate_srt
    _modules_loaded = True


OUTPUT_DIR = Path(os.environ.get("BH_OUTPUT_DIR", "/tmp/content-studio"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLANS_DIR = OUTPUT_DIR / "plans"
PLANS_DIR.mkdir(exist_ok=True)
VIDEOS_DIR = OUTPUT_DIR / "videos"
VIDEOS_DIR.mkdir(exist_ok=True)
AUDIO_DIR = OUTPUT_DIR / "audio"
AUDIO_DIR.mkdir(exist_ok=True)


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
- 🖥️ Visual: {scene.get('visual_type', 'N/A')} — {scene.get('visual_instructions', 'N/A')[:150]}
- 🔗 URL: {scene.get('url', 'N/A')}

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
        return f"❌ Planning failed: {e}\n\n```\n{traceback.format_exc()[-1000:]}\n```", None


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
                          tts_voice, resolution, enable_publish, enable_headless):
    _ensure_modules()
    from planner.planner import ContentPlanner
    from planner.executor import PlanExecutor

    if not topic.strip():
        yield "❌ Please enter a topic.", None, None, None
        return

    platform_list = [p.strip() for p in platforms.split(",") if p.strip()]

    try:
        # Step 1: Plan
        yield "🧠 **Step 1/4: Creating content plan...**\n\nConnecting to MiMo AI to generate script and visual directions...\n\n⏳ This may take 30-60 seconds depending on the topic.", None, None, None

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

        # Step 2: TTS
        yield f"🎙️ **Step 2/4: Generating voiceover...**\n\nPlan: {plan_summary}\n\nGenerating TTS audio for {len(scenes)} scenes using MiMo TTS...\n\n⏳ Each scene takes ~10 seconds.", None, None, None

        executor = PlanExecutor(plan=plan, tts_voice=tts_voice, video_resolution=resolution)

        # Run TTS step
        try:
            executor._step_tts()
        except Exception as e:
            yield f"⚠️ TTS step had issues: {e}\n\nContinuing with visual capture...", None, None, None

        # Step 3: Record/Capture
        yield f"🖥️ **Step 3/4: Creating visual slides...**\n\nPlan: {plan_summary}\n\nGenerating placeholder slides for each scene...", None, None, None

        try:
            executor._step_record(headless=True)
        except Exception as e:
            yield f"⚠️ Visual capture had issues: {e}\n\nContinuing with composition...", None, None, None

        # Step 4: Compose
        yield f"🎬 **Step 4/4: Composing final video...**\n\nPlan: {plan_summary}\n\nAssembling video with audio, titles, and overlays using ffmpeg...", None, None, None

        try:
            executor._step_compose()
        except Exception as e:
            yield f"❌ Video composition failed: {e}\n\n```\n{traceback.format_exc()[-800:]}\n```", None, None, None
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
                            published_info += f"\n⚠️ {platform}: Failed — {p['error']}"
                        else:
                            published_info += f"\n✅ {platform}: Published!"
                else:
                    published_info = "\n⚠️ No platforms were enabled for publishing (requires browser session)."
            except Exception as e:
                published_info = f"\n⚠️ Publishing requires a logged-in browser session: {e}"

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
        # Don't include huge tracebacks that could crash Gradio
        tb = traceback.format_exc()
        if len(tb) > 500:
            tb = tb[-500:]
        yield f"{error_msg}\n\n```\n{tb}\n```", None, None, None


def load_and_show_plan(plan_path):
    if not plan_path:
        return "❌ No plan file specified."
    try:
        with open(plan_path) as f:
            plan = json.load(f)
        return f"```\n{json.dumps(plan, indent=2, ensure_ascii=False)[:3000]}\n```"
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
        "YOUTUBE_ENABLED": bool(os.environ.get("YOUTUBE_ENABLED")),
        "TIKTOK_ENABLED": bool(os.environ.get("TIKTOK_ENABLED")),
        "X_ENABLED": bool(os.environ.get("X_ENABLED")),
    }
    output = "### 🔍 System Status\n\n"
    output += f"- **Mimo API Key**: {'✅ Set' if keys['MIMO_API_KEY'] else '❌ Missing — set MIMO_API_KEY in Space Secrets'}\n"
    output += f"- **Mimo Base URL**: `{keys['MIMO_BASE_URL']}`\n"
    output += f"- **YouTube Publishing**: {'✅ Enabled' if keys['YOUTUBE_ENABLED'] else '⚪ Disabled'}\n"
    output += f"- **TikTok Publishing**: {'✅ Enabled' if keys['TIKTOK_ENABLED'] else '⚪ Disabled'}\n"
    output += f"- **X/Twitter Publishing**: {'✅ Enabled' if keys['X_ENABLED'] else '⚪ Disabled'}\n"
    output += f"\n**Output Directory**: `{OUTPUT_DIR}`"

    # Quick connectivity test
    try:
        import urllib.request
        import urllib.error
        base_url = os.environ.get("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
        api_key = os.environ.get("MIMO_API_KEY", "")
        if api_key:
            test_url = f"{base_url.rstrip('/')}/models"
            req = urllib.request.Request(
                test_url,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                models = [m.get("id", "?") for m in data.get("data", [])[:5]]
                output += f"\n\n**API Connection**: ✅ Working! Available models: {', '.join(models)}"
        else:
            output += "\n\n**API Connection**: ⚠️ No API key set, skipping connectivity test."
    except Exception as e:
        # Try requests library as fallback
        try:
            import requests as req_lib
            resp = req_lib.get(
                f"{base_url.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("id", "?") for m in data.get("data", [])[:5]]
                output += f"\n\n**API Connection**: ✅ Working (via requests)! Available models: {', '.join(models)}"
            else:
                output += f"\n\n**API Connection**: ❌ Failed — HTTP {resp.status_code}"
        except Exception as e2:
            output += f"\n\n**API Connection**: ❌ Failed — {str(e)[:150]}"

    return output


# ──────────────────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────────────────

CSS = """
.gradio-container { max-width: 1100px !important; }
.plan-output { min-height: 300px; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="Content Automation Studio",
    css=CSS,
    theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate"),
) as app:

    gr.Markdown("""
    # 🎬 Content Automation Studio
    **AI-powered video creation and social media publishing.** Plan, narrate, compose, and publish — all from one interface.

    > Powered by **Xiaomi MiMo API** (OpenAI-compatible). API keys stored as Space secrets.
    """)

    with gr.Tabs():
        # ── Tab 1: Full Pipeline ──
        with gr.Tab("🚀 Full Pipeline"):
            gr.Markdown("### End-to-end: Plan → TTS → Compose → Publish")
            with gr.Row():
                with gr.Column(scale=2):
                    pipe_topic = gr.Textbox(
                        label="Video Topic",
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
                            value="tutorial", label="Style"
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
                        pipe_publish = gr.Checkbox(value=False, label="Enable Publishing (requires browser session)")
                        pipe_headless = gr.Checkbox(value=True, label="Slideshow Mode (no browser needed)")

                    pipe_btn = gr.Button("🚀 Run Full Pipeline", variant="primary", size="lg")

                with gr.Column(scale=3):
                    pipe_output = gr.Markdown(label="Progress & Results", elem_classes=["plan-output"])
                    with gr.Row():
                        pipe_video = gr.File(label="📹 Final Video")
                        pipe_audio = gr.File(label="🎙️ Audio")
                        pipe_srt = gr.File(label="📝 Subtitles")

            pipe_btn.click(
                fn=execute_full_pipeline,
                inputs=[
                    pipe_topic, pipe_platforms, pipe_duration, pipe_style,
                    pipe_language, pipe_instructions, pipe_voice, pipe_resolution,
                    pipe_publish, pipe_headless,
                ],
                outputs=[pipe_output, pipe_video, pipe_audio, pipe_srt],
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

        # ── Tab 4: Saved Plans ──
        with gr.Tab("📂 Saved Plans"):
            gr.Markdown("### Browse and manage saved content plans")
            refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
            plans_list = gr.Markdown()
            plan_path_input = gr.Textbox(label="Plan File Path", placeholder="/tmp/content-studio/plans/plan_xxx.json")
            view_btn = gr.Button("👁️ View Plan")
            plan_view = gr.Markdown()

            refresh_btn.click(fn=list_saved_plans, outputs=[plans_list])
            view_btn.click(fn=load_and_show_plan, inputs=[plan_path_input], outputs=[plan_view])

        # ── Tab 5: Settings ──
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
            | `YOUTUBE_ENABLED` | No | Set `1` to enable YouTube publishing |
            | `TIKTOK_ENABLED` | No | Set `1` to enable TikTok publishing |
            | `X_ENABLED` | No | Set `1` to enable X/Twitter publishing |
            """)

            status_btn = gr.Button("🔍 Check Status & API Connection", variant="primary")
            status_output = gr.Markdown()
            status_btn.click(fn=check_status, outputs=[status_output])

app.launch(server_name="0.0.0.0", server_port=7860)
