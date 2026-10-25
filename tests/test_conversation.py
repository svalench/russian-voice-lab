import time
from threading import Event

import numpy as np
import pytest
from fastapi.testclient import TestClient

from voice_lab.engine import check_cancel, run_turn
from voice_lab.server import create_app
from voice_lab.session import Round


class FakeEngine:
    def __init__(self):
        self.status = {"state": "ready", "models": {k: "ready" for k in ("asr", "llm", "tts")}, "error": None}
        self.entered = Event()
        self.gate = Event()
        self.gate.set()
        self.history_seen = None

    def load(self):
        pass

    def transcribe(self, audio, cancel):
        self.entered.set()
        while not self.gate.wait(0.01):
            check_cancel(cancel)
        check_cancel(cancel)
        return "Привет"

    def respond(self, history, instruction, user, cancel):
        self.history_seen = history
        yield "Здравствуйте. "
        yield "Как ваши дела?"

    def speak(self, text, voice, cancel):
        for _ in range(2):
            check_cancel(cancel)
            yield np.zeros(3200, np.float32), 16000

    def clear_cache(self):
        pass


def start_turn(ws):
    ws.send_json({"type": "speech_start"})
    cancelled = ws.receive_json()
    assert cancelled["type"] == "cancelled"
    assert ws.receive_json()["state"] == "listening"
    for _ in range(10):
        ws.send_bytes(np.zeros(512, dtype="<f4").tobytes())
    ws.send_json({"type": "speech_end"})
    return cancelled["turn_id"]


def collect(ws):
    events = []
    receipts = []
    while True:
        event = ws.receive_json()
        events.append(event)
        if event["type"] == "audio":
            audio = ws.receive_bytes()
            assert len(audio) == 3200 * 4
            receipts.append([event["chunk_id"], 3200])
        if event["type"] in ("done", "error"):
            return events, receipts


def test_full_conversation_and_reset():
    engine = FakeEngine()
    with TestClient(create_app(engine)) as client:
        assert client.get("/api/status").json()["state"] == "ready"
        assert client.get("/").status_code == 200
        with client.websocket_connect("/api/session") as ws:
            assert ws.receive_json()["type"] == "connected"
            ws.send_json({"type": "configure", "voice": "Ryan", "instruction": "Кратко"})
            assert ws.receive_json()["type"] == "configured"
            turn = start_turn(ws)
            events, receipts = collect(ws)
            assert all(e["turn_id"] == turn for e in events)
            assert any(e["type"] == "transcript" and e["text"] == "Привет" for e in events)
            assert len(receipts) == 4
            start = {"type": "speech_start", "playback": {"turn_id": turn, "chunks": receipts}}
            ws.send_json(start)
            ws.receive_json(); ws.receive_json()
            for _ in range(10):
                ws.send_bytes(np.zeros(512, dtype="<f4").tobytes())
            ws.send_json({"type": "speech_end"})
            collect(ws)
            assert engine.history_seen == [
                {"role": "user", "content": "Привет"},
                {"role": "assistant", "content": "Здравствуйте. Как ваши дела?"},
            ]
            ws.send_json({"type": "reset"})
            assert ws.receive_json()["type"] == "cancelled"
            assert ws.receive_json()["type"] == "reset"


def test_interrupt_during_inference_keeps_server_responsive_and_drops_stale_output():
    engine = FakeEngine()
    engine.gate.clear()
    with TestClient(create_app(engine)) as client:
        with client.websocket_connect("/api/session") as ws:
            ws.receive_json()
            old = start_turn(ws)
            assert engine.entered.wait(2)
            at = time.monotonic()
            assert client.get("/api/status").status_code == 200
            assert time.monotonic() - at < 1
            ws.send_json({"type": "stop"})
            # A state event may precede the cancellation receipt.
            event = ws.receive_json()
            while event["type"] != "cancelled":
                event = ws.receive_json()
            assert event["turn_id"] > old
            engine.gate.set()
            fresh = start_turn(ws)
            events, _ = collect(ws)
            assert all(e["turn_id"] == fresh for e in events)


def test_second_tab_and_remote_origin_rejected():
    with TestClient(create_app(FakeEngine())) as client:
        with client.websocket_connect("/api/session") as first:
            first.receive_json()
            with client.websocket_connect("/api/session") as second:
                assert "другой вкладке" in second.receive_json()["message"]
        from starlette.websockets import WebSocketDisconnect
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/api/session", headers={"origin": "https://other.example"}):
                pass


def test_invalid_audio_is_reported_without_crashing():
    with TestClient(create_app(FakeEngine())) as client:
        with client.websocket_connect("/api/session") as ws:
            ws.receive_json()
            ws.send_bytes(b"bad")
            assert ws.receive_json()["type"] == "error"
            ws.send_bytes(np.array([float("nan")], dtype="<f4").tobytes())
            assert ws.receive_json()["type"] == "error"
            ws.send_json({"type": "configure", "voice": "unknown"})
            assert ws.receive_json()["type"] == "error"


def test_history_only_contains_heard_audio_and_completed_phrase():
    round_ = Round(1, user="Вопрос")
    round_.phrases = {1: {"text": "Первая фраза.", "ended": True}, 2: {"text": "Вторая не услышана целиком.", "ended": False}}
    round_.chunks = {
        1: {"phrase_id": 1, "audio": np.ones(2400, np.float32), "played": 2400, "sample_rate": 24000},
        2: {"phrase_id": 2, "audio": np.ones(4800, np.float32), "played": 2400, "sample_rate": 24000},
        3: {"phrase_id": 2, "audio": np.ones(4800, np.float32), "played": 0, "sample_rate": 24000},
    }
    heard = round_.heard()
    assert heard["content"] == "Первая фраза."
    assert len(heard["_heard_audio"]) == 1600
    engine = FakeEngine()
    events = []
    run_turn(engine, np.zeros(3200), [{"role": "user", "content": "Вопрос"}, heard], "", "Ryan", Event(), events.append)
    assert engine.history_seen[-1]["content"] == "Первая фраза. Привет"
    assert events[0]["type"] == "history_update"


def test_cancel_stops_audio_and_final_response():
    cancel = Event()
    events = []
    def emit(event):
        events.append(event)
        if event["type"] == "audio":
            cancel.set()
    run_turn(FakeEngine(), np.zeros(3200), [], "", "Ryan", cancel, emit)
    assert sum(e["type"] == "audio" for e in events) == 1
    assert not any(e["type"] == "done" for e in events)
