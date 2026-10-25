"""Real-model browser test with synthetic neural speech, never the user's mic."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test-results"
OUT.mkdir(exist_ok=True)
fixture = (ROOT / ".cache/synthetic-input.wav").read_bytes()

with sync_playwright() as p:
    chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    browser = p.chromium.launch(executable_path=str(chrome) if chrome.exists() else None, headless=True,
                                args=["--autoplay-policy=no-user-gesture-required"])
    context = browser.new_context(viewport={"width": 1440, "height": 1000})
    context.add_init_script("""(() => {
      let ctx, destination, source;
      navigator.mediaDevices.getUserMedia = async () => {
        ctx = new AudioContext(); await ctx.resume();
        destination = ctx.createMediaStreamDestination();
        const zero = ctx.createConstantSource(), gain = ctx.createGain(); gain.gain.value = 0;
        zero.connect(gain).connect(destination); zero.start();
        window.__feedSpeech = async () => {
          const bytes = await (await fetch('/fixture.wav')).arrayBuffer();
          const buffer = await ctx.decodeAudioData(bytes);
          source = ctx.createBufferSource(); source.buffer = buffer; source.connect(destination);
          source.start(); return performance.now();
        };
        return destination.stream;
      };
    })();""")
    page = context.new_page()
    errors, external = [], []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("request", lambda r: external.append(r.url) if not r.url.startswith(("http://127.0.0.1:8765", "ws://127.0.0.1:8765", "data:", "blob:")) else None)
    page.route("**/fixture.wav", lambda route: route.fulfill(body=fixture, content_type="audio/wav"))
    page.goto("http://127.0.0.1:8765")
    page.wait_for_function("!document.querySelector('#start').disabled", timeout=120000)
    status = page.request.get('http://127.0.0.1:8765/api/status').json()
    assert page.locator("#voice").input_value() == "Ирина"
    page.locator("#start").click()
    page.wait_for_function("document.querySelector('#state').textContent === 'Слушаю вас'", timeout=30000)
    page.evaluate("window.__feedSpeech()")
    page.wait_for_function("document.querySelector('#state').textContent === 'Говорю'", timeout=60000)
    assert "Валентин" in page.locator(".message.user").first.inner_text()
    first = page.locator("#metricFirst").inner_text()
    page.wait_for_timeout(350)
    began = page.evaluate("window.__feedSpeech()")
    page.locator(".message.assistant.interrupted").first.wait_for(timeout=15000)
    interrupt_ms = round(page.evaluate("performance.now()") - began)
    page.wait_for_function("document.querySelectorAll('.message.user').length >= 2", timeout=60000)
    page.wait_for_function("document.querySelector('#state').textContent === 'Слушаю вас'", timeout=60000)
    assert page.locator("#error").is_hidden(), page.locator("#error").inner_text()
    page.screenshot(path=str(OUT / "real-models.png"), full_page=True)
    page.locator("#stop").click()
    assert not errors, errors
    assert not external, external
    report = {"backend": "real local models", "model_ids": status['model_ids'], "llm_profile": status.get('llm_profile', '4b'), "input": "synthetic neural speech", "first_audio_display": first,
              "speech_start_to_interrupted_ms": interrupt_ms, "user_turns": page.locator('.message.user').count(),
              "external_requests": external, "browser_errors": errors}
    (OUT / "real-models.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))
    browser.close()
