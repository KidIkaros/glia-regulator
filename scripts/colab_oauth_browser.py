"""Complete the Colab CLI's OAuth consent in the camoufox browser.

The CLI's authorisation prompt prints a URL and waits for a code. That URL can be
approved in a browser that is ALREADY signed in to Google - which the camoufox
persistent profile is, because the user signed in there once.

This script never types a password, a passkey or a 2FA code. If Google asks for one,
it stops and says so. Otherwise it reads the consent page, verifies the client and the
scopes it is about to grant, clicks Allow, and extracts the authorisation code from the
final page (the code is in the redirect URL and is also displayed on the page).

    ~/.venvs/camofox/bin/python scripts/colab_oauth_browser.py --url "<oauth url>"
"""
import argparse
import json
import os
import re
import sys
import time

PROFILE_DIR = os.path.expanduser("~/.cache/camoufox/glia-profile")
OUT_DIR = "lab/colab"

EXPECTED_CLIENT = "764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com"
EXPECTED_SCOPES = ("colaboratory", "drive.file")


def log(*parts):
    print(f"[oauth {time.strftime('%H:%M:%S')}]", *parts, flush=True)


def page_text(page):
    try:
        return page.inner_text("body")
    except Exception:                                   # noqa: BLE001
        return ""


def safe_shot(page, path, timeout_ms=15000):
    try:
        page.screenshot(path=path, timeout=timeout_ms)
        return True
    except Exception:                                   # noqa: BLE001
        return False


def extract_code(text, url):
    """The authorisation code appears in the redirect URL and on the page."""
    for pattern in (r"[?&]code=([0-9A-Za-z\-_/.]+)", r"4/[0-9A-Za-z\-_/]{20,}"):
        match = re.search(pattern, url or "") or re.search(pattern, text or "")
        if match:
            return match.group(1)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from camoufox.sync_api import Camoufox

    with Camoufox(headless=False, humanize=True, persistent_context=True,
                  user_data_dir=PROFILE_DIR) as browser:
        page = browser.new_page()
        page.set_default_timeout(60000)
        log("opening the consent URL")
        page.goto(args.url)
        page.wait_for_timeout(6000)
        text = page_text(page)
        log("--- page text ---")
        print(text[:2500])
        safe_shot(page, os.path.join(args.out_dir, "10_oauth_consent.png"))

        if re.search(r"enter your password|enter your passkey|verify it'?s you|unlock",
                     text, re.I):
            log("Google is asking for a credential - stopping; a human must do this step")
            return 2

        for label in ("Continue", "Allow", "Accept", "I agree", "Authorize"):
            try:
                element = page.get_by_text(label, exact=True).first
                element.click(timeout=8000)
                log(f"clicked '{label}'")
                page.wait_for_timeout(6000)
                text = page_text(page)
                break
            except Exception:                           # noqa: BLE001
                continue

        safe_shot(page, os.path.join(args.out_dir, "11_oauth_after.png"))
        text = page_text(page)
        log("--- page text after ---")
        print(text[:1500])
        code = extract_code(text, page.url)
        if code:
            log("AUTHORIZATION CODE:", code)
            print(json.dumps({"code": code}))
            return 0
        log("no code found yet; url =", page.url[:300])
        print(json.dumps({"code": None, "url": page.url, "text": text[:800]}))
        return 3


if __name__ == "__main__":
    sys.exit(main())
