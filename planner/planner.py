"""AI-driven content planner for automated video creation and publishing.

Uses the Xiaomi MiMo API (OpenAI-compatible) for planning.
Base URL: https://api.mimo-v2.com/v1
Chat models: mimo-v2-pro, mimo-v2-flash

All credentials from environment variables — never hard-coded.
"""

import json
import os
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

# Configuration
MIMO_API_KEY = os.environ.get("MIMO_API_KEY", "")
MIMO_BASE_URL = os.environ.get("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "mimo-v2-flash")

PLAN_OUTPUT_DIR = Path(os.environ.get("BH_PLAN_DIR", Path.home() / "browser-harness-plans"))
PLAN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class ContentPlanner:
    """AI-driven content planner using the MiMo API (OpenAI-compatible).

    Uses chat completions at POST {base_url}/chat/completions
    Model: mimo-v2-flash (fast) or mimo-v2-pro (better reasoning)
    Auth: Authorization: Bearer <key>
    """

    def __init__(self, api_key=None, base_url=None, model=None):
        self.api_key = api_key or MIMO_API_KEY or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or MIMO_BASE_URL).rstrip("/")
        self.model = model or PLANNER_MODEL

        if not self.api_key:
            raise ValueError(
                "No API key set. Set MIMO_API_KEY in .env or environment."
            )

    def plan(self, topic, platforms=None, duration_minutes=3, style="tutorial",
             language="en", extra_instructions=None):
        """Create a content plan for the given topic.

        Args:
            topic: What the video should be about.
            platforms: List of platforms to publish to (youtube, tiktok, x).
            duration_minutes: Target video duration in minutes.
            style: Video style.
            language: Content language code.
            extra_instructions: Additional instructions for the planner.

        Returns:
            Dict with the complete content plan.
        """
        if platforms is None:
            platforms = ["youtube"]

        system_prompt = self._build_system_prompt(style, language)
        user_prompt = self._build_user_prompt(
            topic, platforms, duration_minutes, style, language, extra_instructions
        )

        response = self._chat(system_prompt, user_prompt)
        plan = self._parse_plan(response, topic, platforms)
        return plan

    def _chat(self, system_prompt, user_prompt):
        """Send a chat completion request to the MiMo API."""
        url = f"{self.base_url}/chat/completions"

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
            "max_completion_tokens": 4096,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                data = json.loads(response.read())
                content = data["choices"][0]["message"]["content"]
                return content
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else "unknown"
            raise RuntimeError(
                f"MiMo API error (HTTP {e.code}): {error_body}. "
                f"Check MIMO_API_KEY and MIMO_BASE_URL. "
                f"URL: {url}, Model: {self.model}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot reach MiMo API at {url}: {e.reason}. "
                f"Check MIMO_BASE_URL. Current: {self.base_url}"
            ) from e

    def _build_system_prompt(self, style, language):
        return f"""You are an expert content strategist and video producer. You create detailed, actionable content plans for automated video production.

Your plans must include:
1. A compelling title and description
2. A complete voiceover script with scene-by-scene timing
3. Visual direction for each scene (URLs to navigate, what to record, screenshot instructions)
4. Publishing metadata (tags, categories, hashtags)
5. A publishing schedule for each platform

Output your plan as a JSON object with this exact structure:
{{
  "title": "Video title",
  "description": "Video description for YouTube",
  "style": "{style}",
  "language": "{language}",
  "duration_estimate_seconds": 180,
  "scenes": [
    {{
      "scene_number": 1,
      "duration_seconds": 30,
      "narration": "Voiceover text for this scene",
      "visual_type": "browser_recording|slideshow|title_card|screenshot",
      "visual_instructions": "Navigate to https://example.com and scroll down slowly",
      "url": "https://optional-url-to-navigate.com"
    }}
  ],
  "publishing": {{
    "youtube": {{
      "title": "YouTube title",
      "description": "YouTube description with links",
      "tags": ["tag1", "tag2"],
      "category": "Education",
      "visibility": "public"
    }},
    "tiktok": {{
      "caption": "TikTok caption #hashtag1 #hashtag2",
      "allow_comments": true,
      "allow_duet": false
    }},
    "x": {{
      "tweet": "Tweet text with link #hashtag",
      "thread": ["Optional thread tweet 2"]
    }}
  }}
}}

IMPORTANT: Return ONLY valid JSON. No markdown fences, no commentary."""

    def _build_user_prompt(self, topic, platforms, duration_minutes, style, language, extra_instructions):
        prompt = f"""Create a content plan for a {style} video about: {topic}

Target duration: {duration_minutes} minutes
Target platforms: {', '.join(platforms)}
Language: {language}
"""
        if extra_instructions:
            prompt += f"\nAdditional instructions: {extra_instructions}\n"

        prompt += """
Generate the complete content plan as JSON. Make the narration engaging and informative.
Each scene should be 15-45 seconds long. The visual instructions should be specific enough
for an automated browser to execute (specific URLs, elements to click, scroll directions)."""
        return prompt

    def _parse_plan(self, response, topic, platforms):
        """Parse the LLM response into a structured plan."""
        text = response.strip()

        # Remove markdown code fences if present
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
        except json.JSONDecodeError as e:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    plan = json.loads(text[start:end])
                except json.JSONDecodeError:
                    raise ValueError(f"Could not parse LLM response as JSON: {e}")
            else:
                raise ValueError(f"No JSON found in LLM response: {e}")

        plan["created_at"] = datetime.now().isoformat()
        plan["topic"] = topic
        plan["target_platforms"] = platforms
        plan["status"] = "draft"

        for field in ["title", "scenes"]:
            if field not in plan:
                raise ValueError(f"Plan missing required field: {field}")

        return plan

    def save_plan(self, plan, output_path=None):
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = plan.get("title", "untitled")[:30].replace(" ", "_").replace("/", "_")
            output_path = str(PLAN_OUTPUT_DIR / f"plan_{safe_title}_{timestamp}.json")

        with open(output_path, "w") as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)

        return output_path

    def load_plan(self, path):
        with open(path) as f:
            return json.load(f)


def create_plan(topic, platforms=None, duration_minutes=3, style="tutorial",
                language="en", extra_instructions=None, save=True):
    planner = ContentPlanner()
    plan = planner.plan(topic, platforms, duration_minutes, style, language, extra_instructions)
    if save:
        path = planner.save_plan(plan)
        plan["plan_path"] = path
    return plan
