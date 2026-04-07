"""
Capture a UI demo and produce an animated GIF for the README.

Scenes:
  1. Startup — full UI loaded
  2. Model dropdown open
  3. Text streaming
  4. Text response complete
  5. Video preview — grid of sampled frames
  6. Video uploaded in UI
  7-9. Video description streaming (3 snapshots)
  10. Video description complete (top)
  11. Scrolled to show more

Requirements:
  pip install playwright Pillow opencv-python
  playwright install chromium

Run from the repo root with both servers running:
  python docs/screenshots/capture_screenshots.py
"""
import asyncio
import sys
from io import BytesIO
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent
TEST_IMAGE = OUT / "test-image.png"
TEST_VIDEO = OUT / "test-video.mp4"
BASE_URL = "http://localhost:8501"
GIF_PATH = OUT / "demo.gif"

VIEWPORT = {"width": 1400, "height": 1200}
HOLD_MS = 1200
STREAMING_MS = 600

CHAT_SELECTOR = '[data-testid*="Chat"] textarea'


def build_video_preview(video_path: Path, cols: int = 4, thumb_w: int = 320) -> Image.Image:
    """Create a grid of sampled frames from the video as a single PIL image."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = cols * 2  # 2 rows
    indexes = [int(total * i / n_frames) for i in range(n_frames)]

    thumbs = []
    for idx in indexes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        ratio = thumb_w / img.width
        img = img.resize((thumb_w, int(img.height * ratio)))
        # Add timestamp label
        ts = idx / fps if fps > 0 else 0
        draw = ImageDraw.Draw(img)
        label = f"{ts:.1f}s"
        draw.rectangle([0, 0, 60, 20], fill=(0, 0, 0, 180))
        draw.text((4, 2), label, fill="white")
        thumbs.append(img)
    cap.release()

    if not thumbs:
        return Image.new("RGB", (800, 400), (30, 30, 40))

    th = thumbs[0].height
    padding = 4
    grid_w = cols * thumb_w + (cols + 1) * padding
    rows = (len(thumbs) + cols - 1) // cols
    grid_h = rows * th + (rows + 1) * padding + 40  # +40 for title

    canvas = Image.new("RGB", (grid_w, grid_h), (30, 30, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text((padding, 8), f"Video: {video_path.name}  ({total} frames, {total/fps:.1f}s)", fill="white")

    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, cols)
        x = padding + c * (thumb_w + padding)
        y = 40 + padding + r * (th + padding)
        canvas.paste(thumb, (x, y))

    return canvas


async def wait_for_streamlit(page):
    await page.wait_for_load_state("networkidle", timeout=60_000)
    try:
        await page.locator('[data-testid="stSidebar"]').wait_for(state="visible", timeout=15_000)
    except Exception:
        pass
    await asyncio.sleep(4)


async def wait_gen_start(page):
    await page.wait_for_function(
        """() => { const el = document.querySelector('[data-testid*="Chat"] textarea'); return el && el.disabled; }""",
        timeout=60_000,
    )


async def wait_gen_done(page):
    await page.wait_for_function(
        """() => { const el = document.querySelector('[data-testid*="Chat"] textarea'); return el && !el.disabled; }""",
        timeout=600_000,
    )
    await asyncio.sleep(3)


async def scroll_top(page):
    await page.evaluate("""
        [document.querySelector('[data-testid="stAppViewContainer"]'),
         document.querySelector('[data-testid="stMain"]'),
         document.documentElement, document.body]
        .filter(Boolean).forEach(c => c.scrollTop = 0);
        window.scrollTo(0, 0);
    """)
    await asyncio.sleep(0.8)


async def scroll_chat(page, px=500):
    await page.evaluate(f"""
        const el = document.querySelectorAll('[data-testid="stVerticalBlockBorderWrapper"]');
        if (el.length) el[0].scrollTop += {px};
    """)
    await asyncio.sleep(0.8)


async def snap(page, label):
    raw = await page.screenshot(full_page=False)
    img = Image.open(BytesIO(raw)).convert("RGBA")
    path = OUT / f"{label}.png"
    img.save(path)
    print(f"  ✓ {path.name}")
    return img


def build_gif(frames):
    if not frames:
        return
    rgb = [img.convert("RGB") for img, _ in frames]
    dur = [d for _, d in frames]
    rgb[0].save(GIF_PATH, save_all=True, append_images=rgb[1:], duration=dur, loop=0, optimize=True)
    print(f"\n✓ GIF: {GIF_PATH} ({GIF_PATH.stat().st_size/1024:.0f} KB, {len(frames)} frames)")


async def main():
    from playwright.async_api import async_playwright

    for f, n in [(TEST_IMAGE, "test image"), (TEST_VIDEO, "test video")]:
        if not f.exists():
            print(f"ERROR: {n} not found at {f}")
            sys.exit(1)

    frames = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport=VIEWPORT)
        page = await ctx.new_page()

        # ── 1: Startup ──────────────────────────────────────────
        print("1. Startup…")
        await page.goto(BASE_URL, wait_until="networkidle")
        await wait_for_streamlit(page)
        frames.append((await snap(page, "ui-startup"), HOLD_MS))

        # ── 2: Model dropdown ───────────────────────────────────
        print("2. Model dropdown…")
        try:
            for sel in ['[data-testid="stSidebar"] [data-baseweb="select"]',
                        '[data-testid="stSidebar"] .stSelectbox']:
                loc = page.locator(sel).first
                if await loc.count() > 0:
                    await loc.click(timeout=5_000)
                    await asyncio.sleep(1)
                    frames.append((await snap(page, "ui-model-dropdown"), HOLD_MS))
                    await page.keyboard.press("Escape")
                    await asyncio.sleep(0.5)
                    break
        except Exception:
            print("  ⚠ Skipped dropdown")

        # Reload for clean state
        await page.goto(BASE_URL, wait_until="networkidle")
        await wait_for_streamlit(page)
        await asyncio.sleep(3)

        # ── 3: Text prompt ───────────────────────────────────────
        print("3. Text prompt…")
        ci = page.locator(CHAT_SELECTOR)
        await ci.wait_for(state="visible", timeout=60_000)
        await ci.click()
        await ci.type("Write a one-sentence tagline for a local AI sandbox tool.")
        await page.keyboard.press("Enter")
        await wait_gen_start(page)
        await asyncio.sleep(3)
        frames.append((await snap(page, "ui-text-streaming"), STREAMING_MS))

        # ── 4: Text complete ─────────────────────────────────────
        print("4. Text complete…")
        await wait_gen_done(page)
        await scroll_top(page)
        frames.append((await snap(page, "ui-text-response"), HOLD_MS))

        # ── 5: Video preview (generated offline) ─────────────────
        print("5. Video preview…")
        preview = build_video_preview(TEST_VIDEO)
        # Pad to viewport width for consistent GIF frame size
        padded = Image.new("RGB", (VIEWPORT["width"], VIEWPORT["height"]), (30, 30, 40))
        x_off = (VIEWPORT["width"] - preview.width) // 2
        y_off = (VIEWPORT["height"] - preview.height) // 2
        padded.paste(preview, (x_off, max(y_off, 20)))
        (OUT / "ui-video-preview.png").unlink(missing_ok=True)
        padded.save(OUT / "ui-video-preview.png")
        print(f"  ✓ ui-video-preview.png")
        frames.append((padded.convert("RGBA"), HOLD_MS))

        # ── 6: Video uploaded ─────────────────────────────────────
        print("6. Video upload…")
        tab = page.get_by_text("📁 Upload", exact=True)
        await tab.click()
        await asyncio.sleep(0.5)
        fi = page.locator('[data-testid="stFileUploaderDropzone"] input[type="file"]')
        await fi.set_input_files(str(TEST_VIDEO))
        await asyncio.sleep(2)
        frames.append((await snap(page, "ui-video-uploaded"), HOLD_MS))

        # ── 7-9: Video description streaming ──────────────────────
        print("7. Video description streaming…")
        ci2 = page.locator(CHAT_SELECTOR)
        await ci2.click()
        await ci2.type("Describe the content of this video in detail. Describe each scene, every element, colors, objects, and mood. Be thorough and descriptive.")
        await page.keyboard.press("Enter")
        await wait_gen_start(page)
        for i in range(3):
            await asyncio.sleep(6)
            frames.append((await snap(page, f"ui-video-streaming-{i+1}"), STREAMING_MS))
            print(f"  streaming frame {i+1}/3")

        # ── 10: Video description complete ────────────────────────
        print("10. Video description complete…")
        await wait_gen_done(page)
        await scroll_top(page)
        frames.append((await snap(page, "ui-video-description-top"), HOLD_MS))

        # ── 11: Scrolled ─────────────────────────────────────────
        print("11. Scroll…")
        await scroll_chat(page, 600)
        frames.append((await snap(page, "ui-video-description-scroll"), HOLD_MS))

        await browser.close()

    print("\nBuilding GIF…")
    build_gif(frames)
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
