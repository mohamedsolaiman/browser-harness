"""Visual asset generation for Content Automation Studio.

Provides AI image generation (Pollinations.ai) and stock video sourcing (Pexels API).
All APIs are free or optional — no hard dependencies beyond requests.
"""

from .image_gen import generate_image, generate_scene_images, enhance_prompt
from .stock_video import search_stock_videos, download_stock_video, get_videos_for_topic
