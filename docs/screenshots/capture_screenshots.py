"""
Capture a UI demo and produce an animated GIF for the README.

Scenes:
  1. Startup — model connected, sidebar visible, conversation empty
  2. Text prompt submitted — streaming in progress (mid-generation)
  3. Text response complete — full response visible with run metrics
  4. Image uploaded — file attached, visible in the upload area
  5. Image prompt streaming — model responding mid-generation
  6. Image description complete — final response with attachment label

Requirements:
  pip install playwright Pillow
  playwright install chromium

Run from the repo root with both servers (model-serving + UI) running:
  python docs/screenshots/capture_screenshots.py
"""
import asyncio
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image

OUT = Path(__file__).parent        # docs/screenshots/
TEST_IMAGE = OUT / "test-image.png"
BASE_URL = "http://localhost:8501"
GIF_PATH = OUT / "demo.gif"

VIEWPORT = {"width": 1400, "height": 900}

# GIF timing (milliseconds)
HOLD_MS = 3500       # how long each key scene is shown
STREAMING_MS = 1200  # how long each streaming snapshot is shown


async def wait_for_streamlit(page, timeout_ms: int = 60_000):
    """Block until Streamlit finishes its initial render cycle."""
    await page.wait_for_load_state("networkidle", timeout=timeout_ms)
    # Wait for the sidebar to be fully rendered (model selector visible)
    try:
        await page.locator('[data-testid="stSidebar"]').wait_for(
            state="visible", timeout=15_000,
        )
    except Exception:
        pass
    # Extra settle time for Streamlit's JS hydration
    await asyncio.sleep(3)


async def wait_for_generation_start(page, timeout_ms: int = 60_000):
    """Wait until the chat input becomes disabled (generation started)."""
    await page.wait_for_function(
        """() => {
            const el = document.querySelector('[data-testid="stChatInput"] textarea');
            return el && el.disabled;
        }""",
        timeout=timeout_ms,
    )


async def wait_for_generation_done(page, timeout_ms: int = 180_000):
    """Wait until the chat input re-enables (generation complete)."""
    await page.wait_for_function(
        """() => {
            const el = document.querySelector('[data-testid="stChatInput"] textarea');
            return el && !el.disabled;
        }""",
        timeout=timeout_ms,
    )
    await asyncio.sleep(3)


async def scroll_to_top(page):
    """Scroll Streamlit's app container to the top."""
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


async def capture(page, label: str) -> Image.Image:
    """Take a screenshot and return it as a PIL Image."""
    raw = await page.screenshot(full_page=False)
    img = Image.open(BytesIO(raw)).convert("RGBA")
    png_path = OUT / f"{label}.png"
    img.save(png_path)
    print(f"  ✓ {png_path}")
    return img


def build_gif(frames: list[tuple[Image.Image, int]]) -> None:
    """Stitch frames into an animated GIF."""
    if not frames:
        print("No frames captured — skipping GIF.")
        return

    rgb_images = [img.convert("RGB") for img, _ in frames]
    durations = [d for _, d in frames]

    rgb_images[0].save(
        GIF_PATH,
        save_all=True,
        append_images=rgb_images[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    size_kb = GIF_PATH.stat().st_size / 1024
    print(f"\n✓ GIF saved: {GIF_PATH} ({size_kb:.0f} KB, {len(frames)} frames)")


async def main():
    from playwright.async_api import async_playwright

    if not TEST_IMAGE.exists():
        print(f"ERROR: Test image not found at {TEST_IMAGE}")
        sys.exit(1)

    frames: list[tuple[Image.Image, int]] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport=VIEWPORT)
        page = await ctx.new_page()

        # ── Scene 1: Startup ────────────────────────────────────
        print("Scene 1: Startup (waiting for full render)…")
        await page.goto(BASE_URL, wait_until="networkidle")
        await wait_for_streamlit(page)
        img = await capture(page, "ui-startup")
        frames.append((img, HOLD_MS))

        # ── Scene 1b: Show model dropdown open ─────────────────
        print("  Opening model dropdown…")
        model_select = page.locator('[data-testid="stSidebar"] [data-testid="stSelectbox"]').first
        await model_select.click()
        await asyncio.sleep(1)
        img = await capture(page, "ui-model-dropdown")
        frames.append((img, HOLD_MS))
        # Close dropdown by pressing Escape (keeps current selection)
        await page.keyboard.press("Escape")
        await asyncio.sleep(0.5)

        # ── Scene 2: Text prompt — streaming ────────────────────
        print("Scene 2: Text prompt…")
        chat_input = page.locator('[data-testid="stChatInput"] textarea')
        await chat_input.wait_for(state="visible", timeout=10_000)
        await chat_input.click()
        await chat_input.type("Write a one-sentence tagline for a local AI sandbox tool.")
        await page.keyboard.press("Enter")

        # Wait for generation to start, then capture 2 streaming snapshots
        print("  Capturing streaming…")
        await wait_for_generation_start(page)
        await asyncio.sleep(2)
        img = await capture(page, "ui-text-streaming-1")
        frames.append((img, STREAMING_MS))
        await asyncio.sleep(3)
        img = await capture(page, "ui-text-streaming-2")
        frames.append((img, STREAMING_MS))

        # ── Scene 3: Text response complete ─────────────────────
        print("  Waiting for completion…")
        await wait_for_generation_done(page)
        await scroll_to_top(page)
        img = await capture(page, "ui-text-response")
        frames.append((img, HOLD_MS))

        # ── Scene 4: Image uploaded (before sending) ────────────
        print("Scene 4: Image upload…")
        upload_tab = page.get_by_text("📁 Upload", exact=True)
        await upload_tab.click()
        await asyncio.sleep(0.5)
        file_input = page.locator(
            '[data-testid="stFileUploaderDropzone"] input[type="file"]'
        )
        await file_input.set_input_files(str(TEST_IMAGE))
        await asyncio.sleep(2)  # let Streamlit render the upload preview
        img = await capture(page, "ui-image-uploaded")
        frames.append((img, HOLD_MS))

        # ── Scene 5: Image prompt — streaming ───────────────────
        print("Scene 5: Image description…")
        chat_input2 = page.locator('[data-testid="stChatInput"] textarea')
        await chat_input2.click()
        await chat_input2.type("Describe what you see in this image in detail.")
        await page.keyboard.press("Enter")

        print("  Capturing streaming…")
        await wait_for_generation_start(page)
        await asyncio.sleep(3)
        img = await capture(page, "ui-image-streaming")
        frames.append((img, STREAMING_MS))

        # ── Scene 6: Image description complete ─────────────────
        print("  Waiting for completion…")
        await wait_for_generation_done(page)
        await scroll_to_top(page)
        img = await capture(page, "ui-image-description")
        frames.append((img, HOLD_MS))

        await browser.close()

    # ── Build the GIF ────────────────────────────────────────────
    print("\nBuilding GIF…")
    build_gif(frames)
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
