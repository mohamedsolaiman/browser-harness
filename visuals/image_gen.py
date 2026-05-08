"""AI image generation using Pollinations.ai (free, no API key needed).

Generates cinematic images for video scenes by enhancing visual_instructions
with style keywords and downloading from the Pollinations.ai API.

Usage:
    from visuals.image_gen import generate_image, generate_scene_images, enhance_prompt
    image_path = generate_image("A sunset over mountains", "/tmp/scene1.jpg")
    paths = generate_scene_images(scenes, "/tmp/images/")
"""

import os
import time
import urllib.parse
import hashlib
from pathlib import Path

import requests

# Pollinations.ai base URL (free, no API key)
POLLINATIONS_BASE = "https://image.pollinations.ai/prompt"

# Default image dimensions (16:9 for video)
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720

# Output directory for generated images
IMAGE_OUTPUT_DIR = Path(os.environ.get("CS_IMAGE_DIR", "/tmp/content-studio/images"))
IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Style presets for different visual modes
STYLE_PRESETS = {
    "cinematic": "cinematic, 4k, dramatic lighting, professional, film grain, shallow depth of field, movie scene",
    "corporate": "professional, clean, modern, well-lit, business, high quality, 4k, studio lighting",
    "tech_code": "technology, digital, futuristic, neon lighting, code, cyber, high-tech, 4k, professional",
    "nature": "nature, beautiful, vivid colors, golden hour, landscape photography, 4k, national geographic",
    "abstract": "abstract, artistic, creative, vibrant colors, modern art, high resolution, 4k",
}


def enhance_prompt(visual_instructions: str, style: str = "cinematic") -> str:
    """Enhance a visual instruction prompt with cinematic style keywords.

    Args:
        visual_instructions: The raw visual instructions from the plan.
        style: Visual style preset name (cinematic, corporate, tech_code, nature, abstract).

    Returns:
        Enhanced prompt string with style keywords appended.
    """
    style_suffix = STYLE_PRESETS.get(style, STYLE_PRESETS["cinematic"])

    # Clean up the visual instructions
    prompt = visual_instructions.strip()

    # Handle empty or missing visual instructions
    if not prompt or prompt.lower() in ("n/a", "none"):
        prompt = "educational content overview, informative visual"

    # Remove any URL-like instructions (these are browser navigation commands)
    if prompt.lower().startswith("navigate to"):
        prompt = "educational content overview, informative visual"
    elif "scroll" in prompt.lower() and len(prompt) < 50:
        prompt = "documentary style presentation, informational content"

    # Combine with style keywords
    enhanced = f"{prompt}, {style_suffix}"

    # Truncate if too long (Pollinations has limits)
    if len(enhanced) > 800:
        enhanced = enhanced[:797] + "..."

    return enhanced


def generate_image(
    prompt: str,
    output_path: str = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    seed: int = None,
    style: str = "cinematic",
    retries: int = 3,
    timeout: int = 60,
) -> str:
    """Generate an AI image using Pollinations.ai.

    Args:
        prompt: Image description / visual instructions.
        output_path: Where to save the image. Auto-generated if None.
        width: Image width in pixels.
        height: Image height in pixels.
        seed: Random seed for reproducibility. Auto-generated if None.
        style: Visual style preset to enhance the prompt.
        retries: Number of download attempts.
        timeout: HTTP request timeout in seconds.

    Returns:
        Path to the saved image file.

    Raises:
        RuntimeError: If image generation fails after all retries.
    """
    # Enhance the prompt
    enhanced = enhance_prompt(prompt, style)

    # Generate a seed for reproducibility
    if seed is None:
        seed = int(hashlib.md5(enhanced.encode()).hexdigest()[:8], 16)

    # Build the Pollinations URL
    encoded_prompt = urllib.parse.quote(enhanced, safe="")
    url = f"{POLLINATIONS_BASE}/{encoded_prompt}?width={width}&height={height}&nologo=true&seed={seed}"

    # Set output path
    if output_path is None:
        ts = int(time.time())
        safe_name = hashlib.md5(enhanced.encode()).hexdigest()[:8]
        output_path = str(IMAGE_OUTPUT_DIR / f"scene_{safe_name}_{ts}.jpg")

    # Ensure parent directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Download with retries
    last_error = None
    for attempt in range(retries):
        try:
            print(f"  [image_gen] Generating image (attempt {attempt + 1}/{retries}): {prompt[:60]}...")
            resp = requests.get(url, timeout=timeout, stream=True)
            resp.raise_for_status()

            # Verify we got an image (not an error page)
            content_type = resp.headers.get("content-type", "")
            if "image" not in content_type and len(resp.content) < 1000:
                raise RuntimeError(f"Response doesn't appear to be an image (content-type: {content_type})")

            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Verify file was written
            file_size = os.path.getsize(output_path)
            if file_size < 1000:
                raise RuntimeError(f"Image file too small ({file_size} bytes), likely an error response")

            print(f"  [image_gen] Image saved: {output_path} ({file_size:,} bytes)")
            return output_path

        except requests.exceptions.Timeout as e:
            last_error = e
            print(f"  [image_gen] Timeout on attempt {attempt + 1}, retrying...")
            time.sleep(2)
        except requests.exceptions.RequestException as e:
            last_error = e
            print(f"  [image_gen] Request failed on attempt {attempt + 1}: {e}")
            time.sleep(2)
        except RuntimeError as e:
            last_error = e
            print(f"  [image_gen] Invalid response on attempt {attempt + 1}: {e}")
            time.sleep(1)

    raise RuntimeError(
        f"Image generation failed after {retries} attempts. Last error: {last_error}. "
        f"Prompt: {prompt[:100]}"
    )


def generate_scene_images(
    scenes: list,
    output_dir: str = None,
    style: str = "cinematic",
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> list:
    """Generate one AI image per scene using Pollinations.ai.

    For each scene, uses the `visual_instructions` field as the image prompt.
    Falls back to narration text if visual_instructions is empty.
    If image generation fails for a scene, creates a gradient placeholder.

    Args:
        scenes: List of scene dicts with 'visual_instructions' and 'narration' fields.
        output_dir: Directory to save images. Uses default if None.
        style: Visual style preset.
        width: Image width.
        height: Image height.

    Returns:
        List of file paths to generated images (one per scene).
    """
    if output_dir is None:
        output_dir = str(IMAGE_OUTPUT_DIR)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    image_paths = []

    for i, scene in enumerate(scenes):
        # Get the best prompt from the scene
        prompt = scene.get("visual_instructions", "").strip()
        if not prompt or prompt.lower() in ("n/a", "none", ""):
            prompt = scene.get("narration", f"Scene {i + 1} of educational video")
            # Add visual context to narration-only prompts
            prompt = f"visual representation of: {prompt}"

        output_path = str(Path(output_dir) / f"scene_{i:03d}.jpg")

        try:
            seed = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16) + i
            path = generate_image(
                prompt=prompt,
                output_path=output_path,
                width=width,
                height=height,
                seed=seed,
                style=style,
            )
            image_paths.append(path)
        except Exception as e:
            print(f"  [image_gen] WARNING: Failed to generate image for scene {i + 1}: {e}")
            # Create gradient placeholder
            try:
                placeholder_path = create_gradient_placeholder(
                    text=prompt[:80],
                    output_path=str(Path(output_dir) / f"scene_{i:03d}_placeholder.png"),
                    width=width,
                    height=height,
                    scene_number=i + 1,
                )
                image_paths.append(placeholder_path)
            except Exception as pe:
                print(f"  [image_gen] WARNING: Placeholder also failed: {pe}")
                # Create minimal fallback with ffmpeg
                try:
                    fallback_path = create_ffmpeg_placeholder(
                        text=f"Scene {i + 1}",
                        output_path=str(Path(output_dir) / f"scene_{i:03d}_fallback.png"),
                        width=width,
                        height=height,
                    )
                    image_paths.append(fallback_path)
                except Exception as fe:
                    print(f"  [image_gen] CRITICAL: All image generation failed for scene {i + 1}: {fe}")
                    image_paths.append(None)

    # Filter out None values for reporting
    failed = sum(1 for p in image_paths if p is None)
    success = len(image_paths) - failed
    print(f"  [image_gen] Generated {success}/{len(scenes)} scene images ({failed} placeholders failed)")

    return image_paths


def create_gradient_placeholder(
    text: str,
    output_path: str,
    width: int = 1280,
    height: int = 720,
    scene_number: int = 0,
) -> str:
    """Create a styled gradient placeholder image using Pillow.

    Falls back to ffmpeg if Pillow is not available.

    Args:
        text: Text to display on the placeholder.
        output_path: Where to save the image.
        width: Image width.
        height: Image height.
        scene_number: Scene number for color variation.

    Returns:
        Path to the saved placeholder image.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont

        # Create gradient background
        img = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(img)

        # Color palettes (cycling through different gradient styles)
        palettes = [
            ((20, 30, 48), (36, 59, 85)),    # Dark blue
            ((45, 20, 60), (80, 40, 100)),    # Purple
            ((25, 50, 40), (40, 80, 60)),     # Dark green
            ((60, 20, 20), (90, 35, 35)),     # Dark red
            ((30, 30, 50), (50, 50, 80)),     # Steel blue
        ]
        color_idx = scene_number % len(palettes)
        top_color, bottom_color = palettes[color_idx]

        # Draw gradient
        for y in range(height):
            ratio = y / height
            r = int(top_color[0] + (bottom_color[0] - top_color[0]) * ratio)
            g = int(top_color[1] + (bottom_color[1] - top_color[1]) * ratio)
            b = int(top_color[2] + (bottom_color[2] - top_color[2]) * ratio)
            draw.line([(0, y), (width, y)], fill=(r, g, b))

        # Draw scene number badge
        badge_text = f"Scene {scene_number}" if scene_number > 0 else "Scene"
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
        except (OSError, IOError):
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # Draw text
        # Scene number at top
        bbox = draw.textbbox((0, 0), badge_text, font=font_large)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            ((width - tw) / 2, height * 0.3),
            badge_text,
            fill=(255, 255, 255, 200),
            font=font_large,
        )

        # Description text (wrapped)
        max_chars_per_line = 60
        lines = []
        words = text.split()
        current_line = ""
        for word in words:
            if len(current_line) + len(word) + 1 > max_chars_per_line:
                lines.append(current_line)
                current_line = word
            else:
                current_line = (current_line + " " + word).strip()
        if current_line:
            lines.append(current_line)

        y_offset = height * 0.5
        for line in lines[:3]:  # Max 3 lines
            bbox = draw.textbbox((0, 0), line, font=font_small)
            tw = bbox[2] - bbox[0]
            draw.text(
                ((width - tw) / 2, y_offset),
                line,
                fill=(200, 200, 200, 180),
                font=font_small,
            )
            y_offset += 40

        img.save(output_path, quality=90)
        return output_path

    except ImportError:
        # Pillow not available, use ffmpeg
        return create_ffmpeg_placeholder(text, output_path, width, height)


def create_ffmpeg_placeholder(
    text: str,
    output_path: str,
    width: int = 1280,
    height: int = 720,
) -> str:
    """Create a minimal placeholder image using ffmpeg.

    Args:
        text: Text to display.
        output_path: Output file path.
        width: Image width.
        height: Image height.

    Returns:
        Path to the created placeholder.
    """
    import subprocess

    # Escape text for ffmpeg drawtext
    safe_text = text.replace("'", "").replace(":", "").replace("\\", "")[:60]

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=#1a1a2e:s={width}x{height}:d=1",
        "-frames:v", "1",
        "-vf", (
            f"drawtext=text='{safe_text}':fontsize=36:"
            f"fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2:"
            f"borderw=2:bordercolor=black@0.5"
        ),
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0 and os.path.exists(output_path):
            return output_path
    except Exception:
        pass

    # Absolute last resort: create a tiny black PNG
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:d=1",
        "-frames:v", "1",
        output_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except Exception:
        pass

    return output_path
