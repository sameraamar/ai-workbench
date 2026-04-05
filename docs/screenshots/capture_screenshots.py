"""
Capture fresh UI screenshots for the README.
Run from repo root with the Windows venv activated and both servers running.

  python docs/screenshots/capture_screenshots.py
"""
import asyncio
import sys
import time
from pathlib import Path

OUT = Path(__file__).parent        # docs/screenshots/
IMAGE_PATH = r"C:\Users\saaamar\Downloads\_Image_uu0nwxuu0nwxuu0n.png"
BASE_URL = "http://localhost:8501"

# Viewport that matches a typical 1400px wide browser
VIEWPORT = {"width": 1400, "height": 900}

async def wait_for_streamlit(page, timeout_ms: int = 30_000):
    """Block until Streamlit's running spinner is gone."""
    await page.wait_for_load_state("networkidle", timeout=timeout_ms)
    # Give Streamlit's own re-render cycle a moment to settle
    await asyncio.sleep(3)


async def wait_for_response(page, timeout_ms: int = 180_000):
    """Wait until generation is fully done.

    Two-phase:
    1. Wait for the stChatInput textarea to become DISABLED → generation started.
    2. Wait for it to become ENABLED again → generation done and final rerender complete.
    """
    # Phase 1: confirm submission was received (textarea goes disabled quickly)
    await page.wait_for_function(
        """() => {
            const el = document.querySelector('[data-testid="stChatInput"] textarea');
            return el && el.disabled;
        }""",
        timeout=60_000,
    )
    # Phase 2: wait for generation to finish
    await page.wait_for_function(
        """() => {
            const el = document.querySelector('[data-testid="stChatInput"] textarea');
            return el && !el.disabled;
        }""",
        timeout=timeout_ms,
    )
    # Let Streamlit finish the final DOM paint
    await asyncio.sleep(3)


async def scroll_to_top(page):
    """Scroll the Streamlit app container to the top.

    Streamlit fills the viewport with [data-testid="stAppViewContainer"] which
    has overflow:auto — window.scrollTo doesn't help here.
    """
    await page.evaluate("""
        const containers = [
            document.querySelector('[data-testid="stAppViewContainer"]'),
            document.querySelector('[data-testid="stMain"]'),
            document.documentElement,
            document.body,
        ];
        containers.filter(Boolean).forEach(c => { c.scrollTop = 0; });
        window.scrollTo(0, 0);
    """)
    await asyncio.sleep(0.8)


async def main():
    from playwright.async_api import async_playwright

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport=VIEWPORT)
        page = await ctx.new_page()

        # ── Screenshot 1: startup state ──────────────────────────────
        print("Opening UI…")
        await page.goto(BASE_URL, wait_until="networkidle")
        await wait_for_streamlit(page)
        out1 = OUT / "ui-startup.png"
        await page.screenshot(path=str(out1), full_page=False)
        print(f"  ✓ {out1}")

        # ── Screenshot 2: text prompt + response ─────────────────────
        print("Submitting text prompt…")
        chat_input = page.locator('[data-testid="stChatInput"] textarea')
        await chat_input.wait_for(state="visible", timeout=10_000)
        await chat_input.click()
        await chat_input.type("Write a one-sentence tagline for a local AI sandbox tool.")
        await page.keyboard.press("Enter")
        print("  Waiting for response (may take a while)…")
        await wait_for_response(page, timeout_ms=180_000)
        await scroll_to_top(page)
        out2 = OUT / "ui-text-response.png"
        await page.screenshot(path=str(out2), full_page=False)
        print(f"  ✓ {out2}")

        # ── Screenshot 3: image upload + describe ────────────────────
        print("Uploading image…")
        # Click the Upload tab to make sure the file uploader is visible
        upload_tab = page.get_by_text("📁 Upload", exact=True)
        await upload_tab.click()
        await asyncio.sleep(0.5)

        file_input = page.locator('[data-testid="stFileUploaderDropzone"] input[type="file"]')
        await file_input.set_input_files(IMAGE_PATH)
        await asyncio.sleep(1)   # let Streamlit process the upload

        chat_input2 = page.locator('[data-testid="stChatInput"] textarea')
        await chat_input2.click()
        await chat_input2.type("Describe what you see in this image in a few sentences.")
        await page.keyboard.press("Enter")
        print("  Waiting for image-description response…")
        await wait_for_response(page, timeout_ms=180_000)
        await scroll_to_top(page)
        out3 = OUT / "ui-image-description.png"
        await page.screenshot(path=str(out3), full_page=False)
        print(f"  ✓ {out3}")

        await browser.close()

    print("\nAll screenshots saved to docs/screenshots/")


if __name__ == "__main__":
    asyncio.run(main())
