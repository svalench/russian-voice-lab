"""Real-model WebSocket soak. Playback is acknowledged synthetically (no mic)."""
import argparse
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import onnxruntime
onnxruntime.disable_telemetry_events()
from piper import PiperVoice
from websockets.sync.client import connect
from scripts.benchmark import QUESTIONS
from voice_lab.config import PIPER_FILE, PIPER_PREFIX, model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=10)
    parser.add_argument("--turns", type=int, default=10)
    args = parser.parse_args()
    until = time.monotonic() + 120
    while True:
        with urllib.request.urlopen("http://127.0.0.1:8765/api/status", timeout=10) as response:
            status = json.load(response)
        if status["state"] == "ready":
            assert status["tts_backend"] == "piper"
            break
        if time.monotonic() > until or status["state"] == "error":
            raise RuntimeError(status.get("error") or "Models not ready")
        time.sleep(0.25)
    voice = PiperVoice.load(model_path("piper") / PIPER_PREFIX / PIPER_FILE)
    inputs = []
    for text in QUESTIONS:
        parts = list(voice.synthesize(text))
        samples = np.concatenate([part.audio_float_array for part in parts])
        rate = parts[0].sample_rate
        audio = np.interp(np.arange(0, len(samples), rate / 16000), np.arange(len(samples)), samples).astype("<f4")
        inputs.append(audio)
    report = {"backend": "real models on localhost:8765", "input": "Piper neural synthesis", "playback": "synthetic receipts", "turns": []}
    began = time.monotonic()
    with connect("ws://127.0.0.1:8765/api/session", max_size=2**20, origin="http://127.0.0.1:8765") as ws:
        assert json.loads(ws.recv())["type"] == "connected"
        def send(data):
            ws.send(json.dumps(data))
        send({"type": "configure", "voice": "Ирина", "instruction": "Отвечай кратко, одним предложением."})
        assert json.loads(ws.recv())["type"] == "configured"
        index = 0
        while index < args.turns or time.monotonic() - began < args.minutes * 60:
            send({"type": "speech_start"})
            event = json.loads(ws.recv())
            assert event["type"] == "cancelled"
            turn_id = event["turn_id"]
            assert json.loads(ws.recv())["state"] == "listening"
            audio = inputs[index % len(inputs)]
            for offset in range(0, len(audio), 512):
                ws.send(audio[offset:offset + 512].tobytes())
            start = time.monotonic()
            send({"type": "speech_end"})
            receipts = []
            record = {"input": QUESTIONS[index % len(QUESTIONS)]}
            answer = ""
            while True:
                data = ws.recv(timeout=90)
                event = json.loads(data)
                if event["type"] == "error":
                    raise RuntimeError(event["message"])
                assert event.get("turn_id", turn_id) == turn_id
                if event["type"] == "audio":
                    raw = ws.recv(timeout=90)
                    assert isinstance(raw, bytes)
                    receipts.append([event["chunk_id"], len(raw) // 4])
                    send({"type": "playback", "turn_id": turn_id, "chunks": receipts})
                elif event["type"] == "transcript":
                    record["recognized"] = event["text"]
                elif event["type"] == "text_delta":
                    answer += event["text"]
                elif event["type"] == "metrics":
                    record["metrics"] = event
                elif event["type"] == "done":
                    break
                elif event["type"] == "empty":
                    raise RuntimeError("Empty transcription")
            assert receipts and answer and record.get("recognized")
            send({"type": "playback", "turn_id": turn_id, "chunks": receipts})
            with urllib.request.urlopen("http://127.0.0.1:8765/api/status", timeout=10) as response:
                status = json.load(response)
            record.update(answer=answer, wall_ms=round((time.monotonic() - start) * 1000),
                          mlx_active_mb=status.get("mlx_active_mb"), mlx_peak_mb=status.get("mlx_peak_mb"),
                          rss_mb=status["memory_mb"], available_mb=status["available_memory_mb"])
            report["turns"].append(record)
            report["seconds"] = round(time.monotonic() - began, 2)
            (ROOT / "test-results/session-soak.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            index += 1
            print(f"Turn {index}: first audio {record['metrics']['first_audio_ms']} ms, TTS RTF {record['metrics']['tts_rtf']}, elapsed {report['seconds']} s", flush=True)
    print("Soak complete", flush=True)


if __name__ == "__main__":
    main()
