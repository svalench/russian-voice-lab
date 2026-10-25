"""Run against tests/ui_server.py; real browser/VAD with explicit fake model backend."""
from pathlib import Path
import json
import time

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test-results"
OUT.mkdir(exist_ok=True)

with sync_playwright() as p:
    chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    browser = p.chromium.launch(
        executable_path=str(chrome) if chrome.exists() else None,
        headless=True, args=["--use-fake-device-for-media-stream", "--autoplay-policy=no-user-gesture-required"],
    )
    context = browser.new_context(viewport={"width": 1440, "height": 1000}, permissions=["microphone"])
    page = context.new_page()
    errors, remote = [], []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("request", lambda request: remote.append(request.url) if not request.url.startswith(("http://127.0.0.1:8766", "ws://127.0.0.1:8766", "blob:", "data:")) else None)
    page.goto("http://127.0.0.1:8766")
    page.locator("#start").wait_for(state="visible")
    page.wait_for_function("!document.querySelector('#start').disabled")
    page.screenshot(path=str(OUT / "desktop.png"), full_page=True)
    page.locator("#preview").click()
    page.wait_for_function("document.querySelector('#state').textContent === 'Говорю'")
    at = time.perf_counter()
    page.locator("#stop").click()
    page.wait_for_function("document.querySelector('#state').textContent === 'Готов к разговору'")
    stop_ms = round((time.perf_counter() - at) * 1000)
    page.locator("#start").click()
    page.wait_for_function("document.querySelector('#state').textContent === 'Слушаю вас'", timeout=30000)
    assert page.locator("#error").is_hidden(), page.locator("#error").inner_text()
    page.locator("#pause").fill("1000")
    assert page.locator("#pauseValue").inner_text() == "1000 мс"
    page.locator("#reset").click()
    page.locator("#stop").click()
    with page.expect_download() as download:
        page.locator("#export").click()
    exported = json.loads(Path(download.value.path()).read_text())
    assert "turns" in exported
    page.set_viewport_size({"width": 390, "height": 844})
    page.screenshot(path=str(OUT / "mobile.png"), full_page=True)
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    # The actual bundled ONNX model and AudioWorklet must have initialized above.
    assert not errors, errors
    assert not remote, remote
    context.close()
    denied = browser.new_context(permissions=[])
    page = denied.new_page()
    cdp = denied.new_cdp_session(page)
    target = cdp.send("Target.getTargetInfo")["targetInfo"]
    cdp.send("Browser.setPermission", {"permission": {"name": "microphone"}, "setting": "denied", "origin": "http://127.0.0.1:8766", "browserContextId": target["browserContextId"]})
    page.goto("http://127.0.0.1:8766")
    page.wait_for_function("!document.querySelector('#start').disabled")
    page.locator("#start").click()
    page.locator("#error").wait_for(state="visible", timeout=15000)
    assert "микрофон" in page.locator("#error").inner_text().lower()
    browser.close()
    report = {"test_backend": "FakeEngine", "real_browser_vad": True, "browser_errors": errors,
              "remote_requests": remote, "stop_click_ms": stop_ms, "permission_denial": "passed"}
    (OUT / "browser.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))
