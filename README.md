---
title: Content Automation Studio
emoji: 🎬
colorFrom: indigo
colorTo: slate
sdk: docker
app_port: 7860
pinned: false
tags:
  - video-generation
  - text-to-speech
  - content-automation
  - youtube
  - social-media
short_description: AI-powered video creation and social media publishing
---

# 🎬 Content Automation Studio

Automated video content creation and social media publishing pipeline:

- **AI Planning** — Generate video scripts with scene-by-scene directions
- **TTS Voiceover** — Convert scripts to natural speech via Mimo API
- **Video Composition** — Assemble videos with titles, overlays, and audio
- **Social Publishing** — Upload to YouTube, TikTok, and X/Twitter

## Setup

Add your secrets in Space Settings → Secrets:
- `MIMO_API_KEY` — Required for TTS and AI planning
- `YOUTUBE_ENABLED` — Set to `1` to enable YouTube publishing
- `TIKTOK_ENABLED` — Set to `1` to enable TikTok publishing
- `X_ENABLED` — Set to `1` to enable X/Twitter publishing
