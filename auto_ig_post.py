import os
import logging
from instagrapi import Client
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SESSION_FILE = "ig_session.json"


def auto_ig_post(image_path: str, caption: str) -> None:
    username = os.environ.get("INSTAGRAM_USERNAME")
    password = os.environ.get("INSTAGRAM_PASSWORD")

    if not username or not password:
        raise ValueError("Instagram credentials not set.")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        cl = Client()

        if os.path.exists(SESSION_FILE):
            logger.info("Loading saved session...")
            cl.load_settings(SESSION_FILE)
            cl.login(username, password)
        else:
            logger.info("Logging in for the first time...")
            cl.login(username, password)

        cl.dump_settings(SESSION_FILE)
        logger.info("Session saved.")

        logger.info(f"Uploading {image_path}...")
        media = cl.photo_upload(image_path, caption)

        logger.info(f"Post shared successfully! Media ID: {media.pk}")

    except Exception as e:
        logger.error(f"Failed to post: {e}")
        raise


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python auto_ig_post.py <image_path> <caption>")
        sys.exit(1)

    auto_ig_post(sys.argv[1], sys.argv[2])
