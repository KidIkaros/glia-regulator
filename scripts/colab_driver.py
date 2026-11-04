"""Drive a Colab notebook with camoufox (headful Firefox), on the user's own machine.

Why camoufox rather than the built-in browser driver: this machine's default browser is
Zen Browser (a Firefox fork) and no Chromium is installed, so the built-in driver
cannot reuse the user's Google session. camoufox is Firefox-based, so the sign-in flow
looks like a normal browser to Google - and it is headful, so the user can watch and
take over.

Credential boundary: this script NEVER types a password, a passkey or a 2FA code. It
opens the Google sign-in page and WAITS; the user authenticates in the visible window
(Bitwarden's extension if they have installed it there, or their own passkey). The
browser runs with a PERSISTENT profile, so that session survives driver restarts - the
user signs in once, not once per run.

Flow:
  1. open colab.research.google.com; if signed out, print a clear prompt and wait;
  2. upload the notebook (File > Upload notebook, via the file input);
  3. Runtime > Run all;
  4. poll the output for the results marker and save the JSON next to the lab notes;
  5. write a screenshot per stage so there is evidence of what happened.

Usage:
    ~/.venvs/camofox/bin/python scripts/colab_driver.py --notebook glia_colab_fidelity.ipynb
"""
import argparse
import json
import os
import re
import sys
import time

RESULTS_MARKER = "=== RESULTS ON DRIVE ==="
PROFILE_DIR = os.path.expanduser("~/.cache/camoufox/glia-profile")


def log(*parts):
    print(f"[colab-driver {time.strftime('%H:%M:%S')}]", *parts, flush=True)


def page_text(page):
    try:
        return page.inner_text("body")
    except Exception:                                   # noqa: BLE001
        return ""


def settle(page, seconds=3):
    try:
        page.wait_for_load_state("networkidle", timeout=20000)
    except Exception:                                   # noqa: BLE001
        pass
    page.wait_for_timeout(seconds * 1000)


def is_signed_in(page):
    """Signed in == the nav bar no longer offers Sign in."""
    text = page_text(page)
    return "Sign in" not in text[:500]


def wait_for_sign_in(page, timeout_s):
    """Wait for the user to authenticate in the visible window. Never types anything."""
    log(f"sign-in required - please complete it in the camoufox window "
        f"(you have {timeout_s // 60} minutes; passkey or Bitwarden both work)")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if is_signed_in(page):
            settle(page, 2)
            if is_signed_in(page):
                log("signed in")
                return True
        time.sleep(3)
    return False


def upload_notebook(page, notebook_path, attempts=3):
    """Upload via the landing page's 'Upload notebook' button (a file chooser).

    Setting a file input directly timed out on the first attempt: Google was still
    navigating (a sign-in verification detour), so the input never appeared. Clicking
    the real button and answering its file chooser is the supported path.
    """
    for attempt in range(1, attempts + 1):
        log(f"upload attempt {attempt}/{attempts}")
        try:
            page.goto("https://colab.research.google.com/")
            settle(page, 3)
            with page.expect_file_chooser(timeout=30000) as chooser_info:
                page.get_by_text("Upload notebook", exact=False).first.click(timeout=20000)
            chooser_info.value.set_files(notebook_path)
            log("file chooser satisfied; waiting for the notebook to open")
            page.wait_for_timeout(8000)
            settle(page, 4)
            if "colab.research.google.com" in page.url and is_signed_in(page):
                log("notebook opened")
                return True
        except Exception as error:                       # noqa: BLE001
            log(f"upload attempt {attempt} failed: {type(error).__name__}: "
                f"{str(error)[:200]}")
            page.wait_for_timeout(3000)
    return False


def run_all(page):
    """Runtime > Run all, with the command palette as a fallback."""
    log("triggering Run all")
    try:
        page.get_by_text("Runtime", exact=True).first.click(timeout=15000)
        page.wait_for_timeout(1500)
        page.get_by_text("Run all", exact=False).first.click(timeout=10000)
        log("clicked Runtime > Run all")
        return
    except Exception as error:                           # noqa: BLE001
        log(f"menu path failed: {type(error).__name__}: {str(error)[:120]}")
    page.keyboard.press("Control+Shift+P")
    page.wait_for_timeout(1500)
    page.keyboard.type("Run all")
    page.wait_for_timeout(1200)
    page.keyboard.press("Enter")
    log("used the command palette")


def wait_for_results(page, timeout_s, out_dir):
    """Poll the notebook output for the results marker and save the JSON."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        text = page_text(page)
        if RESULTS_MARKER in text:
            log("results marker found - extracting")
            tail = text.split(RESULTS_MARKER, 1)[1]
            match = re.search(r"\{.*?\n\}", tail, re.S)
            if match:
                payload = json.loads(match.group(0))
                path = os.path.join(out_dir, "colab_results.json")
                json.dump(payload, open(path, "w"), indent=1)
                log("saved", path)
                return payload
        time.sleep(10)
    log("timed out waiting for results")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", default="glia_colab_fidelity.ipynb")
    parser.add_argument("--out-dir", default="lab/colab")
    parser.add_argument("--sign-in-timeout", type=int, default=1800)
    parser.add_argument("--results-timeout", type=int, default=7200)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    notebook = os.path.abspath(args.notebook)
    if not os.path.exists(notebook):
        log("notebook not found:", notebook)
        return 1
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(PROFILE_DIR, exist_ok=True)

    from camoufox.sync_api import Camoufox

    with Camoufox(headless=args.headless, humanize=True, persistent_context=True,
                  user_data_dir=PROFILE_DIR) as browser:
        page = browser.new_page()
        page.set_default_timeout(60000)
        log("opening Colab")
        page.goto("https://colab.research.google.com/")
        page.wait_for_timeout(5000)
        page.screenshot(path=os.path.join(args.out_dir, "01_colab.png"))

        if "Sign in" in page_text(page)[:400]:
            if not wait_for_sign_in(page, args.sign_in_timeout):
                log("sign-in timed out")
                page.screenshot(path=os.path.join(args.out_dir, "02_signin_timeout.png"))
                return 2
        page.screenshot(path=os.path.join(args.out_dir, "02_signed_in.png"))

        if not upload_notebook(page, notebook):
            log("upload failed")
            page.screenshot(path=os.path.join(args.out_dir, "03_upload_failed.png"))
            return 4
        page.screenshot(path=os.path.join(args.out_dir, "03_uploaded.png"))

        run_all(page)
        page.wait_for_timeout(5000)
        page.screenshot(path=os.path.join(args.out_dir, "04_running.png"))
        log("running; this takes ~40-60 min on a T4 (16k-token rollouts in fp32)")

        payload = wait_for_results(page, args.results_timeout, args.out_dir)
        page.screenshot(path=os.path.join(args.out_dir, "05_done.png"))
        if payload is None:
            return 3
        log(json.dumps(payload, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
