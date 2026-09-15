"""Instagram publishing helpers for F1 analysis posts."""

import logging
import sys
from typing import Sequence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def auto_ig_post(image_paths: Sequence[str], caption: str, region: str = "UK") -> None:
    """Publish one photo or one carousel post to Instagram.

    Uses Playwright browser automation with saved session cookies.
    """
    from playwright_ig_poster import auto_ig_post_playwright

    logger.info("Using browser automation to post to Instagram...")
    auto_ig_post_playwright(image_paths, caption, headless=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python auto_ig_post.py <image_path> <caption> [UK|Taiwan]")
        raise SystemExit(1)
    auto_ig_post([sys.argv[1]], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "UK")
