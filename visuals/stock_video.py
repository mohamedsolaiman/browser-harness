"""Free stock video sourcing from Pexels and Pixabay APIs.

Downloads stock video clips for video scenes from Pexels and/or Pixabay.
- Pexels: Requires PEXELS_API_KEY (free at pexels.com/api)
- Pixabay: Requires PIXABAY_API_KEY (free at pixabay.com/api/docs)

If neither key is set, falls back to AI-generated images gracefully.

Usage:
    from visuals.stock_video import search_stock_videos, download_stock_video, get_videos_for_topic
    videos = search_stock_videos("ocean waves")
    path = download_stock_video(videos[0], "/tmp/clip.mp4")
"""

import os
import time
import tempfile
from pathlib import Path

import requests

PEXELS_API_URL = "https://api.pexels.com/videos/search"
PIXABAY_API_URL = "https://pixabay.com/api/videos/"

# Output directory for downloaded videos
VIDEO_OUTPUT_DIR = Path(os.environ.get("CS_VIDEO_DIR", "/tmp/content-studio/videos"))
VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def search_stock_videos(
    query: str,
    api_key: str = None,
    source: str = "auto",
    per_page: int = 5,
    orientation: str = "landscape",
    size: str = "medium",
    timeout: int = 15,
) -> list:
    """Search for stock videos on Pexels and/or Pixabay.

    Args:
        query: Search query string.
        api_key: API key (tries to detect source automatically).
        source: "pexels", "pixabay", or "auto" (try both).
        per_page: Number of results to return.
        orientation: Video orientation ("landscape", "portrait", "square").
        size: Video size preference.
        timeout: HTTP request timeout in seconds.

    Returns:
        List of normalized video result dicts with 'source', 'download_url',
        'width', 'height', 'duration', 'id' fields, or empty list on failure.
    """
    if not query.strip():
        return []

    results = []
    pexels_key = api_key or os.environ.get("PEXELS_API_KEY", "")
    pixabay_key = os.environ.get("PIXABAY_API_KEY", "")

    # Try Pexels first
    if source in ("auto", "pexels") and pexels_key:
        pexels_results = _search_pexels(query, pexels_key, per_page, orientation, timeout)
        results.extend(pexels_results)

    # Try Pixabay
    if source in ("auto", "pixabay") and pixabay_key:
        pixabay_results = _search_pixabay(query, pixabay_key, per_page, orientation, timeout)
        results.extend(pixabay_results)

    # If no API keys at all, try Pixabay without key (limited)
    if not results and source == "auto" and not pexels_key and not pixabay_key:
        print("  [stock_video] No API keys set, trying Pixabay free access...")
        # Pixabay actually requires a key, so we can't do free access
        print("  [stock_video] Set PEXELS_API_KEY or PIXABAY_API_KEY for stock videos")

    print(f"  [stock_video] Found {len(results)} total videos for '{query}'")
    return results


def _search_pexels(query, api_key, per_page=5, orientation="landscape", timeout=15):
    """Search Pexels for stock videos."""
    try:
        params = {
            "query": query.strip(),
            "per_page": min(per_page, 80),
            "orientation": orientation,
            "size": "medium",
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

        # Normalize results
        normalized = []
        for v in videos:
            video_files = v.get("video_files", [])
            if not video_files:
                continue
            # Pick best quality file
            selected = _select_video_file(video_files, "hd")
            if selected and selected.get("link"):
                normalized.append({
                    "source": "pexels",
                    "id": str(v.get("id", "unknown")),
                    "download_url": selected["link"],
                    "width": selected.get("width", 0) or 0,
                    "height": selected.get("height", 0) or 0,
                    "duration": v.get("duration", 0),
                    "raw_data": v,
                })

        print(f"  [stock_video] Pexels: {len(normalized)} videos for '{query}'")
        return normalized

    except requests.exceptions.Timeout:
        print(f"  [stock_video] WARNING: Pexels search timed out for '{query}'")
        return []
    except requests.exceptions.RequestException as e:
        print(f"  [stock_video] WARNING: Pexels search failed: {e}")
        return []
    except (ValueError, KeyError) as e:
        print(f"  [stock_video] WARNING: Invalid Pexels response: {e}")
        return []


def _search_pixabay(query, api_key, per_page=5, orientation="horizontal", timeout=15):
    """Search Pixabay for stock videos."""
    try:
        params = {
            "key": api_key,
            "q": query.strip(),
            "per_page": min(per_page, 200),
            "video_type": "all",
            "min_width": 1280,
            "orientation": "horizontal" if orientation == "landscape" else orientation,
        }

        print(f"  [stock_video] Searching Pixabay for: {query}")
        resp = requests.get(
            PIXABAY_API_URL,
            params=params,
            timeout=timeout,
        )
        resp.raise_for_status()

        data = resp.json()
        hits = data.get("hits", [])

        # Normalize results
        normalized = []
        for hit in hits:
            videos = hit.get("videos", {})
            # Pick the best quality: large > medium > small > tiny
            for quality in ["large", "medium", "small", "tiny"]:
                vid = videos.get(quality, {})
                if vid and vid.get("url"):
                    normalized.append({
                        "source": "pixabay",
                        "id": str(hit.get("id", "unknown")),
                        "download_url": vid["url"],
                        "width": vid.get("width", 0) or 0,
                        "height": vid.get("height", 0) or 0,
                        "duration": hit.get("duration", 0),
                        "raw_data": hit,
                    })
                    break

        print(f"  [stock_video] Pixabay: {len(normalized)} videos for '{query}'")
        return normalized

    except requests.exceptions.Timeout:
        print(f"  [stock_video] WARNING: Pixabay search timed out for '{query}'")
        return []
    except requests.exceptions.RequestException as e:
        print(f"  [stock_video] WARNING: Pixabay search failed: {e}")
        return []
    except (ValueError, KeyError) as e:
        print(f"  [stock_video] WARNING: Invalid Pixabay response: {e}")
        return []


def download_stock_video(
    video_data: dict,
    output_path: str = None,
    timeout: int = 120,
) -> str:
    """Download a stock video.

    Works with normalized video data from both Pexels and Pixabay.

    Args:
        video_data: Normalized video result dict with 'download_url' field.
        output_path: Where to save the video. Auto-generated if None.
        timeout: Download timeout in seconds.

    Returns:
        Path to the downloaded video file, or None on failure.
    """
    download_url = video_data.get("download_url")
    if not download_url:
        print("  [stock_video] WARNING: No download URL in result")
        return None

    # Set output path
    if output_path is None:
        video_id = video_data.get("id", "unknown")
        source = video_data.get("source", "unknown")
        ts = int(time.time())
        output_path = str(VIDEO_OUTPUT_DIR / f"{source}_{video_id}_{ts}.mp4")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        w = video_data.get("width", "?")
        h = video_data.get("height", "?")
        print(f"  [stock_video] Downloading {video_data.get('source','?')} video ({w}x{h})...")
        resp = requests.get(download_url, timeout=timeout, stream=True)
        resp.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)

        file_size = os.path.getsize(output_path)
        if file_size < 10000:
            print(f"  [stock_video] WARNING: Downloaded file too small ({file_size} bytes), may be corrupt")
            os.unlink(output_path)
            return None

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

    Args:
        video_files: List of video file dicts from Pexels.
        quality: Quality preference ("hd", "sd", "largest").

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
        for vf in sorted_files:
            w = vf.get("width", 0) or 0
            if 1280 <= w <= 1920:
                return vf
        return sorted_files[0] if sorted_files else None

    if quality == "sd":
        for vf in sorted_files:
            w = vf.get("width", 0) or 0
            if w <= 854:
                return vf
        return sorted_files[-1] if sorted_files else None

    return sorted_files[0] if sorted_files else None


def get_videos_for_topic(
    topic: str,
    scenes: list,
    api_key: str = None,
    max_videos: int = 10,
    fallback_to_ai: bool = True,
) -> list:
    """Download stock videos for each scene based on the topic and visual instructions.

    For each scene, searches Pexels/Pixabay for relevant stock footage.
    Downloads the best matching video clip for each scene.

    Args:
        topic: Main video topic.
        scenes: List of scene dicts.
        api_key: API key (optional, will use env vars).
        max_videos: Maximum number of videos to download.
        fallback_to_ai: If True, returns None entries for failed scenes
                        so the caller can fall back to AI images.

    Returns:
        List of downloaded video file paths (one per scene, may contain None).
    """
    pexels_key = api_key or os.environ.get("PEXELS_API_KEY", "")
    pixabay_key = os.environ.get("PIXABAY_API_KEY", "")

    if not pexels_key and not pixabay_key:
        print("  [stock_video] No PEXELS_API_KEY or PIXABAY_API_KEY set, skipping stock video downloads")
        return []

    video_paths = []
    downloaded_urls = set()  # Avoid duplicate downloads

    for i, scene in enumerate(scenes):
        if len([p for p in video_paths if p is not None]) >= max_videos:
            # Fill remaining with None
            video_paths.append(None)
            continue

        # Build search query from scene visual instructions or topic
        visual = scene.get("visual_instructions", "").strip()
        narration = scene.get("narration", "").strip()

        if visual and visual.lower() not in ("n/a", "none", ""):
            # Extract key terms from visual instructions
            query = visual.replace("navigate to", "").replace("scroll", "")
            query = " ".join(query.split()[:6])  # Short query for better results
        elif narration:
            # Extract key nouns from narration
            words = narration.split()
            query = " ".join(words[:5])
        else:
            query = topic

        # Search for videos
        try:
            results = search_stock_videos(query, per_page=5)
            # Filter out already-downloaded videos
            results = [r for r in results if r.get("download_url") not in downloaded_urls]

            if results:
                output_path = str(VIDEO_OUTPUT_DIR / f"stock_scene_{i:03d}.mp4")
                path = download_stock_video(results[0], output_path)
                if path:
                    downloaded_urls.add(results[0].get("download_url"))
                    video_paths.append(path)
                else:
                    video_paths.append(None)
            else:
                # Try with the main topic as fallback query
                if query != topic:
                    results = search_stock_videos(topic, per_page=3)
                    results = [r for r in results if r.get("download_url") not in downloaded_urls]
                    if results:
                        output_path = str(VIDEO_OUTPUT_DIR / f"stock_scene_{i:03d}.mp4")
                        path = download_stock_video(results[0], output_path)
                        if path:
                            downloaded_urls.add(results[0].get("download_url"))
                            video_paths.append(path)
                            continue
                video_paths.append(None)
        except Exception as e:
            print(f"  [stock_video] WARNING: Failed for scene {i + 1}: {e}")
            video_paths.append(None)

    success = sum(1 for p in video_paths if p is not None)
    print(f"  [stock_video] Downloaded {success}/{len(scenes)} stock videos")

    return video_paths
