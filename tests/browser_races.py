"""Delayed microphone permission + connection loss regression checks (fake backend)."""
from pathlib import Path
import json
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
with sync_playwright() as p:
    chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    browser = p.chromium.launch(executable_path=str(chrome) if chrome.exists() else None, headless=True,
                                args=["--autoplay-policy=no-user-gesture-required"])
    context = browser.new_context()
    context.add_init_script("""(() => {
      const Original = WebSocket; window.__sockets = []; window.__media = [];
      window.WebSocket = class extends Original { constructor(...args) { super(...args); window.__sockets.push(this); } };
      navigator.mediaDevices.getUserMedia = () => new Promise(resolve => {
        const ctx = new AudioContext(), destination = ctx.createMediaStreamDestination();
        const zero = ctx.createConstantSource(), gain = ctx.createGain(); gain.gain.value=0;
        zero.connect(gain).connect(destination); zero.start();
        window.__media.push({stream: destination.stream, grant: () => resolve(destination.stream), ctx});
      });
    })();""")
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto("http://127.0.0.1:8766")
    page.wait_for_function("!document.querySelector('#start').disabled")
    page.locator("#start").click()
    page.wait_for_function("window.__media.length === 1")
    page.locator("#stop").click()
    page.locator("#start").click()
    page.wait_for_function("window.__media.length === 2")
    page.evaluate("window.__media[0].grant()")
    page.wait_for_function("window.__media[0].stream.getTracks()[0].readyState === 'ended'")
    assert page.locator("#start").is_disabled()
    page.evaluate("window.__media[1].grant()")
    page.wait_for_function("document.querySelector('#state').textContent === 'Слушаю вас'", timeout=30000)
    assert page.evaluate("window.__media[1].stream.getTracks()[0].readyState") == "live"
    page.locator("#stop").click()
    page.locator("#start").click()
    page.wait_for_function("window.__media.length === 3")
    page.evaluate("window.__sockets.at(-1).close()")
    page.wait_for_function("!document.querySelector('#start').disabled")
    page.locator("#start").click()
    page.wait_for_function("window.__media.length === 4")
    page.evaluate("window.__media[2].grant()")
    page.wait_for_function("window.__media[2].stream.getTracks()[0].readyState === 'ended'")
    page.evaluate("window.__media[3].grant()")
    page.wait_for_function("document.querySelector('#state').textContent === 'Слушаю вас'", timeout=30000)
    assert page.evaluate("window.__media[3].stream.getTracks()[0].readyState") == "live"
    page.locator("#reset").click()
    page.wait_for_function("window.__media.length === 5")
    page.locator("#stop").click()
    page.locator("#start").click()
    page.wait_for_function("window.__media.length === 6")
    page.evaluate("window.__media[4].grant()")
    page.wait_for_function("window.__media[4].stream.getTracks()[0].readyState === 'ended'")
    page.evaluate("window.__media[5].grant()")
    page.wait_for_function("document.querySelector('#state').textContent === 'Слушаю вас'", timeout=30000)
    assert page.evaluate("window.__media[5].stream.getTracks()[0].readyState") == "live"
    page.locator("#stop").click()
    page.reload()
    page.wait_for_function("!document.querySelector('#preview').disabled")
    page.locator("#preview").click()
    page.wait_for_function("document.querySelector('#state').textContent === 'Говорю'")
    page.wait_for_function("document.querySelector('#state').textContent === 'Готов к разговору'")
    page.evaluate("window.__sockets.at(-1).close()")
    page.wait_for_function("document.querySelector('#connection').textContent === 'Отключено'")
    page.locator("#preview").click()
    page.wait_for_function("document.querySelector('#state').textContent === 'Говорю'")
    page.wait_for_function("document.querySelector('#state').textContent === 'Готов к разговору'")
    assert page.locator('.message.assistant').count() == 2
    assert page.locator('.message.assistant .body').nth(0).inner_text() == page.locator('.message.assistant .body').nth(1).inner_text()
    with page.expect_download() as download:
        page.locator('#export').click()
    assert len(json.loads(Path(download.value.path()).read_text())['turns']) == 2
    assert not errors, errors
    result = {"delayed_permission_stop_start": "passed", "disconnect_during_permission": "passed", "stop_during_reset": "passed", "reconnect_transcript_and_export": "passed", "browser_errors": errors}
    (ROOT / "test-results/browser-races.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    browser.close()
