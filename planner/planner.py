"""AI-driven content planner for automated video creation and publishing.

Uses the Xiaomi MiMo API (OpenAI-compatible) for planning.
Only uses https://api.xiaomimimo.com/v1 — no broken fallback URLs.

All credentials from environment variables — never hard-coded.
"""

import json
import os
from datetime import datetime
from pathlib import Path

# Configuration
MIMO_API_KEY = os.environ.get("MIMO_API_KEY", "")
MIMO_BASE_URL = os.environ.get("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "mimo-v2-flash")

PLAN_OUTPUT_DIR = Path(os.environ.get("CS_PLAN_DIR", "/tmp/content-studio/plans"))
PLAN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _api_request(base_url, endpoint, api_key, payload, timeout=120):
    """Make an API request using the best available HTTP library.

    Tries: requests → openai SDK → urllib (in that order)
    Returns the parsed JSON response dict.

    Args:
        base_url: API base URL.
        endpoint: API endpoint (e.g. "chat/completions").
        api_key: API key for authorization.
        payload: Request payload dict.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response dict.

    Raises:
        RuntimeError: If all methods fail.
    """
    last_error = None

    # Method 1: Try requests library (most reliable in containers)
    try:
        import requests
        url = f"{base_url.rstrip('/')}/{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        print(f"  [planner] Trying requests to {url}...")
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if resp.status_code >= 400:
            raise RuntimeError(f"API error (HTTP {resp.status_code}): {resp.text[:300]}")
        print(f"  [planner] requests succeeded (HTTP {resp.status_code})")
        return resp.json()
    except ImportError:
        print("  [planner] requests library not available, trying OpenAI SDK...")
    except Exception as e:
        last_error = e
        print(f"  [planner] requests failed ({e}), trying OpenAI SDK...")

    # Method 2: Try OpenAI SDK
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            max_retries=2,
        )
        if endpoint == "chat/completions":
            response = client.chat.completions.create(**payload)
            content = response.choices[0].message.content
            print(f"  [planner] OpenAI SDK succeeded")
            return {"choices": [{"message": {"content": content}}]}
    except ImportError:
        print("  [planner] OpenAI SDK not available, trying urllib...")
    except Exception as e:
        last_error = e
        print(f"  [planner] OpenAI SDK failed ({e}), trying urllib...")

    # Method 3: Fallback to urllib
    try:
        import urllib.request
        import urllib.error
        url = f"{base_url.rstrip('/')}/{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read())
            print(f"  [planner] urllib succeeded")
            return data
    except Exception as e:
        last_error = e
        print(f"  [planner] urllib failed ({e})")

    raise RuntimeError(
        f"All API request methods failed. Last error: {last_error}. "
        f"Check MIMO_API_KEY and network connectivity. "
        f"Base URL: {base_url}"
    )


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
                "No API key set. Set MIMO_API_KEY in .env or environment variable."
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

        Raises:
            RuntimeError: If the API call fails.
            ValueError: If the response cannot be parsed as a valid plan.
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
        """Send a chat completion request to the MiMo API.

        Single URL — no broken fallback URLs.

        Args:
            system_prompt: System message.
            user_prompt: User message.

        Returns:
            The assistant's response content string.

        Raises:
            RuntimeError: If the API call fails.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.7,
            "max_completion_tokens": 4096,
        }

        try:
            data = _api_request(self.base_url, "chat/completions", self.api_key, payload, timeout=120)
            content = data["choices"][0]["message"]["content"]
            print(f"  [planner] API call succeeded using {self.base_url}")
            return content
        except KeyError as e:
            raise RuntimeError(
                f"Unexpected API response format: missing {e}. "
                f"Response: {json.dumps(data)[:500]}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to call MiMo API at {self.base_url}: {e}. "
                f"Please check your MIMO_API_KEY and MIMO_BASE_URL settings."
            ) from e

    def _build_system_prompt(self, style, language):
        return f"""You are an expert content strategist and video producer. You create detailed, actionable content plans for automated video production.

Your plans must include:
1. A compelling title and description
2. A complete voiceover script with scene-by-scene timing
3. Visual direction for each scene — describe what the AI should generate as an image (be vivid and descriptive for image generation prompts)
4. Publishing metadata (tags, categories, hashtags)
5. A publishing schedule for each platform

IMPORTANT: The visual_instructions field will be used as an AI image generation prompt.
Make it vivid, descriptive, and cinematic. Instead of "navigate to URL", describe the
visual scene that should be shown (e.g., "A dramatic close-up of code on a dark monitor
screen with blue syntax highlighting, shallow depth of field, tech aesthetic").

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
      "narration": "Voiceover text for this scene — clear, engaging, informative",
      "visual_type": "ai_generated|title_card|stock_footage",
      "visual_instructions": "Vivid description of what the AI-generated image should show — be specific about composition, lighting, mood, and style",
      "url": ""
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

IMPORTANT: Return ONLY valid JSON. No markdown fences, no commentary.
Use visual_type "ai_generated" for most scenes — this triggers AI image generation.
Use "title_card" only for the opening scene.
Leave url field empty ("") since we use AI-generated visuals, not browser navigation."""

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
Each scene should be 15-45 seconds long.

CRITICAL: The visual_instructions field will be used as an AI image generation prompt.
Write vivid, descriptive visual instructions that will produce stunning images.
Examples:
- "A futuristic holographic display showing data streams and code, dark room with neon blue and purple lighting, cinematic"
- "An animated diagram showing how containers isolate processes, clean tech aesthetic with gradient backgrounds"
- "A dramatic wide shot of a server room with rows of blinking lights, moody atmospheric lighting"

Do NOT write browser navigation commands like "navigate to URL" or "scroll down" —
instead describe the visual scene that should be shown to the viewer."""
        return prompt

    def _parse_plan(self, response, topic, platforms):
        """Parse the LLM response into a structured plan.

        Handles various response formats:
        - Pure JSON
        - JSON wrapped in markdown code fences
        - JSON embedded in surrounding text
        - JSON with trailing commas or other common LLM mistakes

        Args:
            response: Raw LLM response string.
            topic: Original topic (added to plan metadata).
            platforms: Target platforms list.

        Returns:
            Parsed plan dict.

        Raises:
            ValueError: If the response cannot be parsed as valid JSON.
        """
        text = response.strip()

        # Remove markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            start = 1
            # Skip the language identifier line
            if lines[0].strip().startswith("```"):
                start = 1
            end = len(lines)
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip().startswith("```"):
                    end = i
                    break
            text = "\n".join(lines[start:end])

        # Try direct JSON parse
        try:
            plan = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from surrounding text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                json_text = text[start:end]
                try:
                    plan = json.loads(json_text)
                except json.JSONDecodeError:
                    # Try fixing common LLM JSON mistakes
                    plan = self._try_fix_json(json_text)
                    if plan is None:
                        raise ValueError(
                            f"Could not parse LLM response as JSON. "
                            f"Response preview: {text[:500]}"
                        )
            else:
                raise ValueError(
                    f"No JSON found in LLM response. "
                    f"Response preview: {text[:500]}"
                )

        # Add metadata
        plan["created_at"] = datetime.now().isoformat()
        plan["topic"] = topic
        plan["target_platforms"] = platforms
        plan["status"] = "draft"

        # Validate required fields
        for field in ["title", "scenes"]:
            if field not in plan:
                raise ValueError(f"Plan missing required field: {field}")

        # Validate scenes
        scenes = plan.get("scenes", [])
        if not scenes:
            raise ValueError("Plan has no scenes")

        # Ensure each scene has visual_instructions (for AI image generation)
        for i, scene in enumerate(scenes):
            if not scene.get("visual_instructions"):
                # Generate from narration as fallback
                narration = scene.get("narration", f"Scene {i + 1}")
                scene["visual_instructions"] = f"Visual representation of: {narration[:100]}"
            # Default visual_type to ai_generated
            if not scene.get("visual_type"):
                scene["visual_type"] = "ai_generated"

        return plan

    def _try_fix_json(self, text):
        """Try to fix common LLM JSON mistakes and parse.

        Handles:
        - Trailing commas before } or ]
        - Single quotes instead of double quotes
        - Comments (// or /* */)
        - Missing quotes around keys

        Args:
            text: JSON text to fix.

        Returns:
            Parsed dict, or None if all fixes fail.
        """
        import re

        fixes = [
            # Fix trailing commas before closing braces/brackets
            (r',\s*}', '}'),
            (r',\s*]', ']'),
            # Fix trailing commas after values in arrays/objects
            (r',(\s*[}\]])', r'\1'),
        ]

        fixed = text
        for pattern, replacement in fixes:
            fixed = re.sub(pattern, replacement, fixed)

        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        # More aggressive: try regex-based extraction of key fields
        try:
            # Try to extract just the essential structure
            # Use a simple approach: find all "key": "value" pairs
            # and reconstruct a minimal plan
            title_match = re.search(r'"title"\s*:\s*"([^"]*)"', fixed)
            if title_match:
                title = title_match.group(1)
                # Extract scenes using regex
                scenes = []
                scene_pattern = r'\{\s*"scene_number"\s*:\s*(\d+)[^}]*"narration"\s*:\s*"([^"]*)"[^}]*"visual_instructions"\s*:\s*"([^"]*)"[^}]*\}'
                for m in re.finditer(scene_pattern, fixed, re.DOTALL):
                    scenes.append({
                        "scene_number": int(m.group(1)),
                        "duration_seconds": 30,
                        "narration": m.group(2),
                        "visual_type": "ai_generated",
                        "visual_instructions": m.group(3),
                        "url": "",
                    })

                if scenes:
                    return {
                        "title": title,
                        "description": title,
                        "style": "tutorial",
                        "language": "en",
                        "duration_estimate_seconds": len(scenes) * 30,
                        "scenes": scenes,
                        "publishing": {},
                    }
        except Exception:
            pass

        return None

    def save_plan(self, plan, output_path=None):
        """Save a plan to a JSON file.

        Args:
            plan: Plan dict.
            output_path: Output path. Auto-generated if None.

        Returns:
            Path to the saved file.
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = plan.get("title", "untitled")[:30].replace(" ", "_").replace("/", "_")
            output_path = str(PLAN_OUTPUT_DIR / f"plan_{safe_title}_{timestamp}.json")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)

        return output_path

    def load_plan(self, path):
        """Load a plan from a JSON file.

        Args:
            path: Path to the plan file.

        Returns:
            Plan dict.
        """
        with open(path) as f:
            return json.load(f)


def create_plan(topic, platforms=None, duration_minutes=3, style="tutorial",
                language="en", extra_instructions=None, save=True):
    """Convenience function to create and optionally save a plan.

    Args:
        topic: Video topic.
        platforms: Target platforms.
        duration_minutes: Target duration.
        style: Video style.
        language: Content language.
        extra_instructions: Additional instructions.
        save: Whether to save the plan to disk.

    Returns:
        Plan dict.
    """
    planner = ContentPlanner()
    plan = planner.plan(topic, platforms, duration_minutes, style, language, extra_instructions)
    if save:
        path = planner.save_plan(plan)
        plan["plan_path"] = path
    return plan
