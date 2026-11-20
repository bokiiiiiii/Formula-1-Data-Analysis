"""Exercise authentication against local HTML fixtures; never contact Instagram."""

import json
import os
import unittest
from unittest.mock import mock_open, patch

from playwright.sync_api import sync_playwright

from playwright_ig_poster import ensure_authenticated_session


class AuthenticationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(headless=True)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()

    def setUp(self):
        self.context = self.browser.new_context(service_workers="block")
        self.page = self.context.new_page()
        self.addCleanup(self.context.close)
        self.env = self.enterContext(patch.dict(os.environ, {}, clear=True))
        self.saved_session = self.enterContext(
            patch("playwright_ig_poster.open", mock_open())
        )
        self.enterContext(patch.object(self.page, "wait_for_timeout"))

    def serve(self, html):
        self.page.route(
            "**/*", lambda route: route.fulfill(content_type="text/html", body=html)
        )

    def login_form(self, username_field=True, success=True):
        user = '<input name="username">' if username_field else ""
        script = """
            window.loginValues = Array.from(document.querySelectorAll('input'))
                .map(input => input.value);
        """
        if success:
            script += """
                document.cookie = 'sessionid=test-session; path=/';
                document.body.innerHTML = '<p>Feed</p>';
            """
        self.serve(
            f"<form onsubmit=\"event.preventDefault(); {script}\">"
            f'{user}<input type="password"><button type="submit">Log in</button>'
            "</form>"
        )

    def test_missing_credentials_are_reported_before_fill(self):
        self.login_form()
        with self.assertRaisesRegex(
            ValueError, "INSTAGRAM_USERNAME, INSTAGRAM_PASSWORD"
        ):
            ensure_authenticated_session(self.page, self.context)
        self.assertEqual(self.page.locator('input[type="password"]').input_value(), "")
        self.saved_session.assert_not_called()

    def test_missing_password_is_reported(self):
        self.env["INSTAGRAM_USERNAME"] = "test-user"
        self.login_form()
        with self.assertRaisesRegex(ValueError, "requires INSTAGRAM_PASSWORD"):
            ensure_authenticated_session(self.page, self.context)
        self.saved_session.assert_not_called()

    def test_missing_username_is_reported(self):
        self.env["INSTAGRAM_PASSWORD"] = "test-password"
        self.login_form()
        with self.assertRaisesRegex(ValueError, "requires INSTAGRAM_USERNAME"):
            ensure_authenticated_session(self.page, self.context)
        self.saved_session.assert_not_called()

    def test_valid_credentials_fill_login_and_save_authenticated_cookie(self):
        self.env.update(
            INSTAGRAM_USERNAME="test-user", INSTAGRAM_PASSWORD="test-password"
        )
        self.login_form()
        ensure_authenticated_session(self.page, self.context)
        self.assertEqual(
            self.page.evaluate("window.loginValues"), ["test-user", "test-password"]
        )
        written = "".join(
            call.args[0] for call in self.saved_session().write.call_args_list
        )
        self.assertEqual(json.loads(written)["cookies"]["sessionid"], "test-session")

    def test_password_only_prompt_does_not_require_username(self):
        self.env["INSTAGRAM_PASSWORD"] = "test-password"
        self.login_form(username_field=False)
        ensure_authenticated_session(self.page, self.context)
        self.assertEqual(self.page.evaluate("window.loginValues"), ["test-password"])

    def test_existing_authenticated_session_needs_no_credentials(self):
        self.serve("<p>Feed</p>")
        self.context.add_cookies(
            [
                {
                    "name": "sessionid",
                    "value": "test-session",
                    "url": "https://www.instagram.com/",
                }
            ]
        )
        ensure_authenticated_session(self.page, self.context)
        self.saved_session.assert_called_once()

    def test_failed_login_does_not_overwrite_saved_session(self):
        self.env.update(
            INSTAGRAM_USERNAME="test-user", INSTAGRAM_PASSWORD="test-password"
        )
        self.login_form(success=False)
        with self.assertRaisesRegex(RuntimeError, "login was not completed"):
            ensure_authenticated_session(self.page, self.context)
        self.saved_session.assert_not_called()

    def test_missing_two_factor_seed_is_reported(self):
        self.serve('<input autocomplete="one-time-code">')
        with self.assertRaisesRegex(ValueError, "INSTAGRAM_2FA_SEED"):
            ensure_authenticated_session(self.page, self.context)
        self.saved_session.assert_not_called()

    def test_unauthenticated_cookie_is_not_saved(self):
        self.serve("<p>Unrecognized login page</p>")
        with self.assertRaisesRegex(RuntimeError, "authenticated session cookie"):
            ensure_authenticated_session(self.page, self.context)
        self.saved_session.assert_not_called()


if __name__ == "__main__":
    unittest.main()
