"""Automatic Instagram posting for F1 analysis plots."""

import time
import os
import logging
from typing import Optional
from playwright.sync_api import sync_playwright
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 5


class InstagramPoster:
    """Handle Instagram posting operations."""

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        headless: bool = False,
    ):
        self.username = username or os.environ.get("INSTAGRAM_USERNAME")
        self.password = password or os.environ.get("INSTAGRAM_PASSWORD")
        self.headless = headless

        if not self.username or not self.password:
            raise ValueError(
                "Instagram credentials not set. "
                "Set INSTAGRAM_USERNAME and INSTAGRAM_PASSWORD environment variables."
            )

        logger.info("InstagramPoster initialized")

    def _dismiss_popup(self, page, name: str, timeout: int = 5000) -> None:
        """Try to dismiss a popup button."""
        try:
            page.get_by_role("button", name=name).click(timeout=timeout)
            time.sleep(2)
        except:
            pass

    def _login(self, page) -> bool:
        """Log in to Instagram."""
        try:
            logger.info("Starting Instagram login...")
            page.goto("https://www.instagram.com/accounts/login/?hl=en")
            time.sleep(5)

            self._dismiss_popup(page, "Allow all cookies")

            page.fill('input[name="email"]', self.username, timeout=10000)
            page.fill('input[name="pass"]', self.password)
            page.get_by_role("button", name="Log in", exact=True).click()
            time.sleep(8)

            self._dismiss_popup(page, "Not now")
            self._dismiss_popup(page, "Not now")

            logger.info("Login successful")
            return True

        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False

    def _upload_post(self, page, image_path: str, caption: str) -> bool:
        """Upload a post to Instagram."""
        try:
            logger.info(f"Starting post upload for {image_path}")

            page.locator("a").filter(has_text="Create").first.click()
            time.sleep(2)

            page.get_by_role("link", name="Post Post").click()
            time.sleep(3)

            page.set_input_files('input[type="file"]', image_path)
            time.sleep(3)

            try:
                page.locator("button").filter(has_text="Select crop").click(
                    timeout=3000
                )
                time.sleep(1)
                page.get_by_role("button", name="Crop portrait icon").click(
                    timeout=3000
                )
                time.sleep(2)
            except:
                pass

            page.get_by_role("button", name="Next").click()
            time.sleep(2)
            page.get_by_role("button", name="Next").click()
            time.sleep(2)

            try:
                page.get_by_label("Write a caption...").fill(caption)
            except:
                page.locator("textarea").first.fill(caption)
            time.sleep(3)

            page.get_by_role("button", name="Share", exact=True).click()
            time.sleep(15)

            logger.info("Post shared successfully")
            return True

        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False

    def post(self, image_path: str, caption: str, retries: int = MAX_RETRIES) -> bool:
        """Post an image to Instagram with retry logic."""
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return False

        for attempt in range(retries):
            try:
                logger.info(f"Posting to Instagram (attempt {attempt + 1}/{retries})")

                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(
                        headless=self.headless, args=["--start-maximized"]
                    )
                    context = browser.new_context(no_viewport=True, locale="en-US")
                    page = context.new_page()

                    try:
                        if not self._login(page):
                            raise Exception("Login failed")

                        if not self._upload_post(page, image_path, caption):
                            raise Exception("Upload failed")

                        logger.info("Post completed successfully")
                        return True

                    finally:
                        context.close()
                        browser.close()

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)

        logger.error(f"Failed to post after {retries} attempts")
        return False


def auto_ig_post(image_path: str, caption: str, headless: bool = False) -> None:
    """Post image to Instagram."""
    poster = InstagramPoster(headless=headless)
    if not poster.post(image_path, caption):
        raise RuntimeError("Failed to post image to Instagram")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python auto_ig_post.py <image_path> <caption>")
        sys.exit(1)

    auto_ig_post(sys.argv[1], sys.argv[2])
