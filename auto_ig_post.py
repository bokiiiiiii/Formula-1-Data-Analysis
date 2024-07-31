import re
import time
import os
from playwright.sync_api import Playwright, sync_playwright


def auto_ig_post(image_path: str, caption: str) -> None:
    """Automate Instagram post using Playwright."""

    # Fetching credentials from environment variables
    username = os.environ.get("INSTAGRAM_USERNAME")
    password = os.environ.get("INSTAGRAM_PASSWORD")

    if not username or not password:
        raise ValueError("Instagram credentials are not set in environment variables.")

    def login(page):
        """Log in to Instagram."""
        page.goto("https://www.instagram.com/?hl=zh-tw")
        time.sleep(1)

        page.fill('input[name="username"]', username)
        page.fill('input[name="password"]', password)
        page.click('button[type="submit"]')
        time.sleep(5)

        # Handle pop-ups
        try:
            page.get_by_role("button", name="稍後再說").click()
            time.sleep(3)
            page.get_by_role("button", name="稍後再說").click()
            time.sleep(3)
        except:
            pass

    def upload_post(page):
        """Upload the post to Instagram."""
        page.get_by_role("link", name="新貼文 建立").click()
        page.get_by_role("link", name="貼文 貼文").click()
        time.sleep(3)

        page.locator("div").filter(has_text=re.compile(r"^從電腦選擇$")).nth(1).click()
        time.sleep(3)

        page.set_input_files('input[type="file"]', image_path)
        time.sleep(3)

        page.get_by_role("button", name="下一步").click()
        page.get_by_role("button", name="下一步").click()
        time.sleep(3)

        page.get_by_role("paragraph").click()
        page.get_by_label("撰寫說明文字……").fill(caption)
        time.sleep(10)

        page.get_by_role("button", name="分享").click()
        time.sleep(10)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        try:
            login(page)
            upload_post(page)
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            context.close()
            browser.close()
