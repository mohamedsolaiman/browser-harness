"""Free stock video sourcing from Pexels API.

Downloads stock video clips for video scenes from Pexels.
Requires a PEXELS_API_KEY environment variable — if not set,
all functions return empty results (graceful degradation).

Usage:
    from visuals.stock_video import search_stock_videos, download_stock_video, get_videos_for_topic
    videos = search_stock_videos("ocean waves", api_key)
    path = download_stock_video(videos[0], "/tmp/clip.mp4")
"""

import os
import time
from pathlib import Path

import requests

PEXELS_API_URL = "https://api.pexels.com/videos/search"

# Output directory for downloaded videos
VIDEO_OUTPUT_DIR = Path(os.environ.get("CS_VIDEO_DIR", "/tmp/content-studio/videos"))
VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def search_stock_videos(
    query: str,
    api_key: str = None,
    per_page: int = 5,
    orientation: str = "landscape",
    size: str = "medium",
    timeout: int = 15,
) -> list:
    """Search for stock videos on Pexels.

    Args:
        query: Search query string.
        api_key: Pexels API key. Falls back to PEXELS_API_KEY env var.
        per_page: Number of results to return (max 80).
        orientation: Video orientation ("landscape", "portrait", "square").
        size: Video size preference ("large", "medium", "small").
        timeout: HTTP request timeout in seconds.

    Returns:
        List of video result dicts from Pexels API, or empty list on failure.
    """
    api_key = api_key or os.environ.get("PEXELS_API_KEY", "")
    if not api_key:
        print("  [stock_video] No PEXELS_API_KEY set, skipping stock video search")
        return []

    if not query.strip():
        return []

    try:
        params = {
            "query": query.strip(),
            "per_page": min(per_page, 80),
            "orientation": orientation,
            "size": size,
        }
        headers = {"Authorization": api_key}

        print(f"  [stock_video] Searching Pexels for: {query}")
        resp = requests.get(
            PEXELS_API_URL,
            params=params,
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()

        data = resp.json()
        videos = data.get("videos", [])
        print(f"  [stock_video] Found {len(videos)} videos for '{query}'")
        return videos

    except requests.exceptions.Timeout:
        print(f"  [stock_video] WARNING: Pexels search timed out for '{query}'")
        return []
    except requests.exceptions.RequestException as e:
        print(f"  [stock_video] WARNING: Pexels search failed: {e}")
        return []
    except (ValueError, KeyError) as e:
        print(f"  [stock_video] WARNING: Invalid Pexels response: {e}")
        return []


def download_stock_video(
    video_data: dict,
    output_path: str = None,
    quality: str = "hd",
    timeout: int = 120,
) -> str:
    """Download a stock video from Pexels.

    Picks the best quality video file from the video_files array.

    Args:
        video_data: Video result dict from Pexels API.
        output_path: Where to save the video. Auto-generated if None.
        quality: Preferred quality ("hd", "sd", "largest").
        timeout: Download timeout in seconds.

    Returns:
        Path to the downloaded video file, or None on failure.
    """
    video_files = video_data.get("video_files", [])
    if not video_files:
        print("  [stock_video] WARNING: No video files in result")
        return None

    # Select the best quality video file
    selected = _select_video_file(video_files, quality)
    if not selected:
        print("  [stock_video] WARNING: No suitable video file found")
        return None

    download_url = selected.get("link")
    if not download_url:
        return None

    # Set output path
    if output_path is None:
        video_id = video_data.get("id", "unknown")
        ts = int(time.time())
        output_path = str(VIDEO_OUTPUT_DIR / f"pexels_{video_id}_{ts}.mp4")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"  [stock_video] Downloading video from Pexels ({selected.get('width', '?')}x{selected.get('height', '?')})...")
        resp = requests.get(download_url, timeout=timeout, stream=True)
        resp.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)

        file_size = os.path.getsize(output_path)
        print(f"  [stock_video] Downloaded: {output_path} ({file_size:,} bytes)")
        return output_path

    except requests.exceptions.Timeout:
        print(f"  [stock_video] WARNING: Video download timed out")
        return None
    except requests.exceptions.RequestException as e:
        print(f"  [stock_video] WARNING: Video download failed: {e}")
        return None
    except Exception as e:
        print(f"  [stock_video] WARNING: Unexpected download error: {e}")
        return None


def _select_video_file(video_files: list, quality: str = "hd") -> dict:
    """Select the best video file from Pexels video_files array.

    Prioritizes HD resolution and reasonable file sizes.

    Args:
        video_files: List of video file dicts from Pexels.
        quality: Quality preference.

    Returns:
        Best matching video file dict, or None.
    """
    if not video_files:
        return None

    # Sort by resolution (width * height), descending
    sorted_files = sorted(
        video_files,
        key=lambda f: (f.get("width", 0) or 0) * (f.get("height", 0) or 0),
        reverse=True,
    )

    if quality == "largest":
        return sorted_files[0] if sorted_files else None

    if quality == "hd":
        # Prefer 720p or 1080p
        for vf in sorted_files:
            w = vf.get("width", 0) or 0
            if 1280 <= w <= 1920:
                return vf
        # Fall back to largest available
        return sorted_files[0] if sorted_files else None

    if quality == "sd":
        # Prefer 480p or lower
        for vf in sorted_files:
            w = vf.get("width", 0) or 0
            if w <= 854:
                return vf
        # Fall back to smallest
        return sorted_files[-1] if sorted_files else None

    # Default: return first (largest)
    return sorted_files[0] if sorted_files else None


def get_videos_for_topic(
    topic: str,
    scenes: list,
    api_key: str = None,
    max_videos: int = 5,
) -> list:
    """Download stock videos for each scene based on the topic and visual instructions.

    For each scene, searches Pexels for relevant stock footage.
    If no PEXELS_API_KEY is set, returns empty list (graceful skip).

    Args:
        topic: Main video topic (used for search when scene lacks visual instructions).
        scenes: List of scene dicts.
        api_key: Pexels API key. Falls back to env var.
        max_videos: Maximum number of videos to download.

    Returns:
        List of downloaded video file paths (may be shorter than scenes count).
    """
    api_key = api_key or os.environ.get("PEXELS_API_KEY", "")
    if not api_key:
        print("  [stock_video] No PEXELS_API_KEY set, skipping stock video downloads")
        return []

    video_paths = []

    for i, scene in enumerate(scenes):
        if len(video_paths) >= max_videos:
            break

        # Build search query from scene visual instructions or topic
        visual = scene.get("visual_instructions", "").strip()
        if visual and visual.lower() not in ("n/a", "none", ""):
            # Extract key terms from visual instructions
            # Remove URLs and navigation commands
            query = visual.replace("navigate to", "").replace("scroll", "")
            # Keep it short for search
            query = " ".join(query.split()[:8])
        else:
            query = topic

        # Search for videos
        try:
            results = search_stock_videos(query, api_key, per_page=3)
            if results:
                output_path = str(VIDEO_OUTPUT_DIR / f"stock_scene_{i:03d}.mp4")
                path = download_stock_video(results[0], output_path)
                if path:
                    video_paths.append(path)
                else:
                    video_paths.append(None)
            else:
                video_paths.append(None)
        except Exception as e:
            print(f"  [stock_video] WARNING: Failed for scene {i + 1}: {e}")
            video_paths.append(None)

    success = sum(1 for p in video_paths if p is not None)
    print(f"  [stock_video] Downloaded {success}/{min(len(scenes), max_videos)} stock videos")

    return video_paths
