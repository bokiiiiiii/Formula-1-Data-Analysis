"""Playwright based Instagram posting helper for carousel and single posts."""

import json
import base64
import hashlib
import hmac
import logging
import os
import re
import time
from pathlib import Path
from typing import Sequence
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

load_dotenv(override=True)
logger = logging.getLogger(__name__)

SESSION_FILE = "ig_session.json"
MAX_CAROUSEL_ITEMS = 20


def generate_totp(secret: str) -> str:
    """Generate the current 6-digit TOTP code from a base32 secret."""
    key = base64.b32decode(secret.replace(" ", "").strip(), casefold=True)
    counter = int(time.time() / 30).to_bytes(8, "big")
    digest = hmac.new(key, counter, hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    value = int.from_bytes(digest[offset : offset + 4], "big") & 0x7FFFFFFF
    return f"{value % 1_000_000:06d}"


def get_playwright_cookies(session_file: str = SESSION_FILE) -> list[dict]:
    """Extract cookies formatted for Playwright context."""
    if not os.path.exists(session_file):
        return []

    with open(session_file, "r", encoding="utf-8") as f:
        session_data = json.load(f)

    cookies_dict = session_data.get("cookies", {})
    return [
        {
            "name": name,
            "value": str(val),
            "domain": ".instagram.com",
            "path": "/",
        }
        for name, val in cookies_dict.items()
    ]


def prepare_carousel_images(image_paths: Sequence[str]) -> list[str]:
    """Validate carousel images."""
    valid_paths = [
        str(Path(path).resolve()) for path in image_paths if Path(path).is_file()
    ]
    if not valid_paths:
        raise FileNotFoundError("No valid images were supplied for the Instagram post.")

    if len(valid_paths) > MAX_CAROUSEL_ITEMS:
        raise ValueError(
            f"Instagram allows at most {MAX_CAROUSEL_ITEMS} carousel slides."
        )
    return valid_paths


def select_portrait_crop(page) -> None:
    """Select Instagram's 4:5 portrait crop used by the original workflow."""
    page.wait_for_timeout(1500)
    crop_control = page.get_by_role(
        "button", name=re.compile(r"Select crop|選取裁切|裁切", re.IGNORECASE)
    ).first
    if crop_control.count() == 0:
        crop_control = (
            page.locator("button")
            .filter(has_text=re.compile(r"Select crop|選取裁切|裁切", re.IGNORECASE))
            .first
        )
    if crop_control.count() > 0 and crop_control.is_visible():
        crop_control.click()
        page.wait_for_timeout(1000)
        portrait_control = page.get_by_role(
            "button",
            name=re.compile(r"Crop portrait icon|Portrait|直式|縱向", re.IGNORECASE),
        ).first
        if portrait_control.count() == 0:
            portrait_control = (
                page.locator("button")
                .filter(
                    has_text=re.compile(
                        r"Crop portrait icon|Portrait|直式|縱向", re.IGNORECASE
                    )
                )
                .first
            )
        if portrait_control.count() > 0 and portrait_control.is_visible():
            portrait_control.click()
            page.wait_for_timeout(1000)
            logger.info("Selected Instagram 4:5 portrait crop.")


def ensure_authenticated_session(page, context) -> None:
    """Ensure page is on feed; perform login / 2FA automatically if prompted."""
    username = os.environ.get("INSTAGRAM_USERNAME")
    password = os.environ.get("INSTAGRAM_PASSWORD")
    seed = os.environ.get("INSTAGRAM_2FA_SEED")

    page.goto("https://www.instagram.com/", timeout=60000)
    page.wait_for_timeout(3000)

    # 1. One-tap continue
    cont_btn = page.locator(
        'div[role="button"]:has-text("Continue"), button:has-text("Continue"), div[role="button"]:has-text("繼續"), button:has-text("繼續")'
    )
    if cont_btn.count() > 0 and cont_btn.first.is_visible():
        logger.info("One-tap profile detected, clicking Continue...")
        cont_btn.first.click()
        page.wait_for_timeout(3000)

    # 2. If password prompt or login page
    pw_input = page.locator('input[type="password"]')
    if pw_input.count() > 0 and pw_input.first.is_visible():
        logger.info("Entering Instagram password...")
        user_input = page.locator('input[name="email"], input[name="username"]')
        if user_input.count() > 0 and user_input.first.is_visible():
            user_input.first.fill(username)
        pw_input.first.fill(password)
        login_btn = page.locator(
            'button[type="submit"], div[role="button"]:has-text("Log in"), button:has-text("Log in"), div[role="button"]:has-text("登入")'
        ).first
        login_btn.click()
        page.wait_for_timeout(6000)

    # 3. If 2FA screen
    if (
        "two_step_verification" in page.url
        or "two_factor" in page.url
        or page.locator('input[name="verificationCode"]').count() > 0
        or page.locator('input[name="approvals_code"]').count() > 0
    ):
        logger.info("2FA required, generating TOTP code...")
        code = generate_totp(seed) if seed else ""
        if code:
            code_input = page.locator(
                'input[name="verificationCode"], input[name="approvals_code"], input[type="tel"], input[type="text"]'
            ).first
            code_input.fill(code)
            code_input.press("Enter")
            page.wait_for_timeout(6000)
    elif (
        page.locator('input[type="text"]').count() > 0 and "accounts/login" in page.url
    ):
        # Check if 2FA code input rendered without two_factor in URL
        code = generate_totp(seed) if seed else ""
        if code:
            page.locator('input[type="text"]').first.fill(code)
            page.locator('input[type="text"]').first.press("Enter")
            page.wait_for_timeout(6000)

    # 4. Dismiss popups
    for _ in range(2):
        for text in [
            "Save info",
            "儲存資訊",
            "Not now",
            "Not Now",
            "稍後再說",
            "Cancel",
            "取消",
        ]:
            try:
                b = page.locator(
                    f'button:has-text("{text}"), div[role="button"]:has-text("{text}")'
                )
                if b.count() > 0 and b.first.is_visible():
                    b.first.click()
                    page.wait_for_timeout(1500)
            except Exception:
                pass

    # Save fresh session cookies
    fresh = context.cookies()
    cookie_dict = {c["name"]: c["value"] for c in fresh}
    session_data = {
        "authorization_data": {
            "sessionid": cookie_dict.get("sessionid", ""),
            "ds_user_id": cookie_dict.get("ds_user_id", ""),
        },
        "cookies": cookie_dict,
    }
    with open(SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2)


def auto_ig_post_playwright(
    image_paths: Sequence[str],
    caption: str,
    headless: bool = True,
) -> None:
    """Upload photos / carousel post to Instagram via Playwright browser automation."""
    valid_images = prepare_carousel_images(image_paths)
    cookies = get_playwright_cookies()

    logger.info(
        "Launching browser automation to publish Instagram post (%d slides)...",
        len(valid_images),
    )
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        )
        if cookies:
            context.add_cookies(cookies)
        page = context.new_page()

        try:
            ensure_authenticated_session(page, context)

            # Click Create / 建立 in sidebar
            create_btn = (
                page.locator(
                    'svg[aria-label="New post"], svg[aria-label="新建貼文"], svg[aria-label="新貼文"], svg[aria-label="建立"], svg[aria-label="Create"]'
                )
                .or_(page.locator('span:text-is("建立"), span:text-is("Create")'))
                .first
            )
            create_btn.click()
            page.wait_for_timeout(2000)

            # Check if there is a "Post" sub-item in menu dropdown
            post_sub = page.locator('span:text-is("Post"), span:text-is("貼文")').first
            if post_sub.count() > 0 and post_sub.is_visible():
                post_sub.click()
                page.wait_for_timeout(2000)

            # Upload initial image or all images
            file_input = page.locator('input[type="file"]').first
            file_input.set_input_files(valid_images)
            page.wait_for_timeout(3000)
            select_portrait_crop(page)

            # Click Next / 下一步 (Crop screen)
            next_btn = page.locator(
                'div[role="button"]:has-text("Next"), button:has-text("Next"), div[role="button"]:has-text("下一步"), button:has-text("下一步")'
            ).first
            if next_btn.is_visible():
                next_btn.click()
                page.wait_for_timeout(2000)

            # Click Next / 下一步 (Filter / edit screen)
            next_btn = page.locator(
                'div[role="button"]:has-text("Next"), button:has-text("Next"), div[role="button"]:has-text("下一步"), button:has-text("下一步")'
            ).first
            if next_btn.is_visible():
                next_btn.click()
                page.wait_for_timeout(2000)

            # Fill caption
            caption_input = page.locator(
                'div[role="textbox"][aria-label*="caption"], div[role="textbox"][aria-label*="說明文字"], div[contenteditable="true"]'
            ).first
            caption_input.click()
            caption_input.fill(caption)
            page.wait_for_timeout(2000)

            # Click the exact Share control inside the visible create-post dialog.
            post_dialog = page.locator(
                'div[role="dialog"][aria-label="Create new post"]'
            ).last
            if post_dialog.count() == 0:
                post_dialog = page.locator('div[role="dialog"]').last
            share_btn = post_dialog.get_by_text(
                re.compile(r"^(Share|分享)$", re.IGNORECASE)
            ).last
            share_btn.click()
            logger.info("Submitting post to Instagram, waiting for confirmation...")

            # Only report success after Instagram renders an explicit confirmation.
            success_text = page.get_by_text(
                re.compile(
                    r"Your post has been shared|你的貼文已發佈|貼文已分享|已分享",
                    re.IGNORECASE,
                )
            )
            try:
                success_text.first.wait_for(state="visible", timeout=90000)
            except Exception as error:
                page.screenshot(path="instagram_post_timeout.png", full_page=True)
                raise TimeoutError(
                    "Instagram did not confirm the post within 90 seconds. "
                    "See instagram_post_timeout.png for the final page state."
                ) from error

            logger.info("Post successfully shared on Instagram!")

        finally:
            context.close()
            browser.close()
