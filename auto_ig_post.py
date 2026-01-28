"""Automatic Instagram posting for F1 analysis plots."""

import time
import os
import logging
from pathlib import Path
from typing import Optional
from playwright.sync_api import sync_playwright
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Logger
logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


class InstagramPoster:
    """Handle Instagram posting operations."""

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        headless: bool = False,
        timeout: int = 30000,
    ):
        """Initialize Instagram poster.

        Args:
            username: Instagram username (reads from env if not provided)
            password: Instagram password (reads from env if not provided)
            headless: Whether to run browser in headless mode
            timeout: Timeout for operations in milliseconds

        Raises:
            ValueError: If credentials are not provided or found in environment
        """
        self.username = username or os.environ.get("INSTAGRAM_USERNAME")
        self.password = password or os.environ.get("INSTAGRAM_PASSWORD")
        self.headless = headless
        self.timeout = timeout

        if not self.username or not self.password:
            raise ValueError(
                "Instagram credentials are not set. "
                "Please provide them as arguments or set "
                "INSTAGRAM_USERNAME and INSTAGRAM_PASSWORD environment variables."
            )

        logger.info("InstagramPoster initialized")

    def _login(self, page) -> bool:
        """Log in to Instagram.

        Args:
            page: Playwright page object

        Returns:
            True if login successful, False otherwise
        """
        try:
            logger.info("Starting Instagram login...")
            page.goto("https://www.instagram.com/?hl=en", timeout=self.timeout)
            time.sleep(2)

            # Accept cookies (best effort)
            try:
                page.get_by_role("button", name="Allow").click(timeout=3000)
            except:
                logger.debug("Cookie acceptance dialog not found")

            time.sleep(1)

            # Fill credentials
            page.fill('input[name="username"]', self.username, timeout=self.timeout)
            page.fill('input[name="password"]', self.password, timeout=self.timeout)
            page.click('button[type="submit"]', timeout=self.timeout)

            # Wait for navigation
            time.sleep(5)

            logger.info("Login successful")
            return True

        except Exception as e:
            logger.error(f"Login failed: {str(e)}")
            return False

    def _upload_post(self, page, image_path: str, caption: str) -> bool:
        """Upload a post to Instagram.

        Args:
            page: Playwright page object
            image_path: Path to the image file
            caption: Post caption text

        Returns:
            True if upload successful, False otherwise
        """
        try:
            logger.info(f"Starting post upload for {image_path}")

            # Dismiss any popups
            time.sleep(3)
            try:
                page.get_by_role("button", name="Not now").click(timeout=3000)
            except:
                pass

            try:
                page.get_by_role("button", name="OK").click(timeout=3000)
            except:
                pass

            time.sleep(2)

            # Click create/new post button
            try:
                page.get_by_role("link", name="Create").click(timeout=5000)
            except:
                try:
                    page.get_by_role("link", name="New post").click(timeout=5000)
                except Exception as e:
                    logger.error(f"Failed to click create button: {str(e)}")
                    return False

            time.sleep(3)

            # Select post type
            try:
                page.get_by_role("link", name="Post").click(timeout=5000)
            except:
                logger.debug("Post option not found, continuing...")

            time.sleep(2)

            # Click to select file
            try:
                page.locator('input[type="file"]').set_input_files(image_path)
            except Exception as e:
                logger.error(f"Failed to select file: {str(e)}")
                return False

            time.sleep(3)

            # Proceed through steps (click next buttons)
            for step in range(3):
                try:
                    page.get_by_role("button", name="Next").click(timeout=5000)
                    time.sleep(2)
                except Exception as e:
                    logger.debug(f"Next button not found at step {step}")

            # Add caption
            try:
                page.locator(
                    'textarea[aria-label*="caption"], '
                    'textarea[aria-label*="description"]'
                ).fill(caption)
            except:
                logger.debug("Caption textarea not found, trying alternative...")
                try:
                    page.locator("textarea").first.fill(caption)
                except Exception as e:
                    logger.error(f"Failed to fill caption: {str(e)}")
                    return False

            time.sleep(2)

            # Share post
            try:
                page.get_by_role("button", name="Share").click(timeout=5000)
                time.sleep(10)  # Wait for upload to complete
                logger.info("Post shared successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to share post: {str(e)}")
                return False

        except Exception as e:
            logger.error(f"Upload failed: {str(e)}")
            return False

    def post(self, image_path: str, caption: str, retries: int = MAX_RETRIES) -> bool:
        """Post an image to Instagram with retry logic.

        Args:
            image_path: Path to the image file
            caption: Post caption text
            retries: Number of retries on failure

        Returns:
            True if post successful, False otherwise
        """
        # Validate image path
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return False

        for attempt in range(retries):
            try:
                logger.info(f"Posting to Instagram (attempt {attempt + 1}/{retries})")

                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(headless=self.headless)
                    context = browser.new_context(
                        viewport={"width": 1920, "height": 1080}
                    )
                    page = context.new_page()
                    page.set_default_timeout(self.timeout)

                    try:
                        # Login
                        if not self._login(page):
                            raise Exception("Login failed")

                        # Upload post
                        if not self._upload_post(page, image_path, caption):
                            raise Exception("Upload failed")

                        logger.info("Post completed successfully")
                        return True

                    finally:
                        context.close()
                        browser.close()

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < retries - 1:
                    logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)

        logger.error(f"Failed to post after {retries} attempts")
        return False


# Backward compatibility function
def auto_ig_post(image_path: str, caption: str, headless: bool = False) -> None:
    """Post image to Instagram (legacy function for backward compatibility).

    Args:
        image_path: Path to image file
        caption: Post caption
        headless: Whether to run browser in headless mode

    Raises:
        RuntimeError: If posting fails
    """
    poster = InstagramPoster(headless=headless)
    success = poster.post(image_path, caption)
    if not success:
        raise RuntimeError("Failed to post image to Instagram")


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 3:
        print("Usage: python auto_ig_post.py <image_path> <caption>")
        sys.exit(1)

    image_path = sys.argv[1]
    caption = sys.argv[2]

    auto_ig_post(image_path, caption)
