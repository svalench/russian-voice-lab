"""Session protocol, audio buffering, playback receipts and bounded inference."""
import asyncio
import concurrent.futures
import contextlib
import json
from collections import deque
from dataclasses import dataclass, field
from threading import Event

import numpy as np
from fastapi import WebSocket

from .config import VOICES
from .engine import Cancelled, run_turn

MAX_SAMPLES = 16000 * 60


@dataclass
class Round:
    id: int
    user: str = ""
    phrases: dict = field(default_factory=dict)
    chunks: dict = field(default_factory=dict)
    preview: bool = False
    heard_override: str | None = None
    done: bool = False

    def heard(self):
        """Use complete phrase text; re-recognize only the audible partial phrase."""
        if self.heard_override is not None:
            return {"role": "assistant", "content": self.heard_override}
        completed = []
        partial = []
        rate = 24000
        for phrase_id, phrase in self.phrases.items():
            chunks = [x for x in self.chunks.values() if x["phrase_id"] == phrase_id]
            if phrase["ended"] and chunks and all(x["played"] == len(x["audio"]) for x in chunks):
                completed.append(phrase["text"])
            else:
                for chunk in chunks:
                    if chunk["played"]:
                        partial.append(chunk["audio"][:chunk["played"]])
                        rate = chunk["sample_rate"]
                break
        result = {"role": "assistant", "content": " ".join(completed)}
        if partial:
            audio = np.concatenate(partial)
            # Resample the actually played fragment for the same local 16 kHz ASR.
            if rate != 16000:
                audio = np.interp(np.arange(0, len(audio), rate / 16000), np.arange(len(audio)), audio).astype(np.float32)
            result.update(_heard_audio=audio, _round_id=self.id)
        return result


class Session:
    def __init__(self, ws, engine, executor):
        self.ws = ws
        self.engine = engine
        self.executor = executor
        self.outbox = asyncio.Queue(maxsize=24)
        self.cancel = Event()
        self.future = None
        self.rounds = []
        self.current = None
        self.turn_id = 0
        self.chunk_id = 0
        self.prebuffer = deque(maxlen=10)  # 320 ms at Silero v5's frame size.
        self.audio = []
        self.sample_count = 0
        self.recording = False
        self.instruction = ""
        self.voice = getattr(engine, "voices", VOICES)[0]
        self.closed = False

    def receipt(self, data):
        round_ = next((r for r in self.rounds if r.id == data.get("turn_id")), None)
        if round_:
            for chunk_id, count in data.get("chunks", [])[:1500]:
                chunk = round_.chunks.get(chunk_id)
                if chunk:
                    chunk["played"] = max(chunk["played"], min(max(int(count), 0), len(chunk["audio"])))
            self.compact_round(round_)

    def compact_round(self, round_):
        if round_.done and round_.chunks and all(c["played"] == len(c["audio"]) for c in round_.chunks.values()):
            heard = round_.heard()
            round_.heard_override = heard["content"]
            round_.chunks.clear()

    def history(self):
        messages = []
        for round_ in self.rounds:
            if round_.preview or not round_.user:
                continue
            messages.append({"role": "user", "content": round_.user})
            assistant = round_.heard()
            if assistant["content"] or "_heard_audio" in assistant:
                messages.append(assistant)
            else:
                # Merge consecutive user turns when the assistant was interrupted before speaking.
                if len(messages) >= 2 and messages[-2]["role"] == "user":
                    messages[-2]["content"] += "\n" + messages.pop()["content"]
        return messages

    async def send(self, event):
        await self.ws.send_json(event)

    async def interrupt(self, receipt=None):
        if receipt:
            self.receipt(receipt)
        self.cancel.set()
        if self.future and not self.future.running():
            self.future.cancel()
        if self.current:
            heard = self.current.heard()
            if "_heard_audio" not in heard:
                self.current.heard_override = heard["content"]
                self.current.chunks.clear()
        self.turn_id += 1
        self.recording = False
        self.audio = []
        self.sample_count = 0
        await self.send({"type": "cancelled", "turn_id": self.turn_id})

    async def launch(self, preview=False):
        if self.engine.status["state"] != "ready":
            await self.send({"type": "error", "message": "Модели ещё не готовы. Дождитесь загрузки."})
            return
        # At most one executing + one pending turn. A replaced pending turn is cancelled.
        if self.future and not self.future.done() and not self.future.running():
            self.future.cancel()
        token = self.cancel = Event()
        turn_id = self.turn_id
        history = self.history()
        round_ = Round(turn_id, preview=preview)
        self.rounds.append(round_)
        self.rounds = self.rounds[-256:]
        self.current = round_
        audio = np.concatenate(self.audio) if self.audio else np.zeros(0, np.float32)
        self.audio = []
        loop = asyncio.get_running_loop()

        def emit(event):
            if token.is_set() or self.closed:
                raise Cancelled()
            event["turn_id"] = turn_id
            future = asyncio.run_coroutine_threadsafe(self.outbox.put(event), loop)
            while True:
                try:
                    future.result(timeout=0.1)
                    break
                except concurrent.futures.TimeoutError:
                    if token.is_set() or self.closed:
                        future.cancel()
                        raise Cancelled()

        self.future = self.executor.submit(
            run_turn, self.engine, audio, history, self.instruction, self.voice, token, emit, preview,
        )

    async def sender(self):
        while True:
            event = await self.outbox.get()
            if event["turn_id"] != self.turn_id or self.cancel.is_set():
                continue
            round_ = self.current
            kind = event["type"]
            if kind == "history_update":
                old = next((r for r in self.rounds if r.id == event["round_id"]), None)
                if old:
                    old.heard_override = event["text"]
                    old.chunks.clear()
                await self.send(event)
            elif kind == "transcript":
                round_.user = event["text"]
                await self.send(event)
            elif kind == "phrase":
                round_.phrases[event["phrase_id"]] = {"text": event["text"], "ended": False}
                await self.send(event)
            elif kind == "phrase_end":
                round_.phrases[event["phrase_id"]]["ended"] = True
                await self.send(event)
            elif kind == "audio":
                self.chunk_id += 1
                raw = event.pop("samples")
                event["chunk_id"] = self.chunk_id
                round_.chunks[self.chunk_id] = {
                    "audio": np.frombuffer(raw, dtype="<f4"), "played": 0,
                    "phrase_id": event["phrase_id"], "sample_rate": event["sample_rate"],
                }
                # Metadata and binary frame are sent by this sole sender, without interleaving.
                await self.send(event)
                await self.ws.send_bytes(raw)
            elif kind == "done":
                round_.done = True
                self.compact_round(round_)
                await self.send(event)
            else:
                await self.send(event)

    async def handle(self, data):
        if not isinstance(data, dict):
            raise ValueError("Ожидается JSON-объект")
        kind = data.get("type")
        if kind == "configure":
            if self.recording or (self.future and not self.future.done() and not self.cancel.is_set()):
                await self.send({"type": "error", "message": "Настройки меняются между репликами."})
                return
            voice = data.get("voice", "Ryan")
            if voice not in getattr(self.engine, "voices", VOICES):
                raise ValueError("Неизвестный голос")
            self.voice = voice
            self.instruction = str(data.get("instruction", ""))[:2000]
            await self.send({"type": "configured", "voice": voice})
        elif kind == "speech_start":
            await self.interrupt(data.get("playback"))
            self.recording = True
            self.audio = list(self.prebuffer)
            self.sample_count = sum(len(x) for x in self.audio)
            await self.send({"type": "state", "state": "listening", "turn_id": self.turn_id})
        elif kind == "speech_end":
            if self.recording:
                self.recording = False
                if self.sample_count >= 1600:
                    await self.launch()
        elif kind in ("speech_misfire", "stop"):
            await self.interrupt(data.get("playback"))
            self.prebuffer.clear()
        elif kind == "reset":
            await self.interrupt(data.get("playback"))
            self.rounds.clear()
            self.prebuffer.clear()
            await self.send({"type": "reset"})
        elif kind == "preview":
            await self.interrupt(data.get("playback"))
            await self.launch(preview=True)
        elif kind == "playback":
            self.receipt(data)
        else:
            raise ValueError("Неизвестное событие")

    async def receive_audio(self, raw):
        if len(raw) % 4 or len(raw) > 8192:
            raise ValueError("Некорректный PCM-фрагмент")
        samples = np.frombuffer(raw, dtype="<f4").copy()
        if not np.all(np.isfinite(samples)):
            raise ValueError("Некорректные значения PCM")
        self.prebuffer.append(samples)
        if self.recording:
            self.audio.append(samples)
            self.sample_count += len(samples)
            if self.sample_count > MAX_SAMPLES:
                await self.interrupt()
                await self.send({"type": "error", "message": "Реплика длиннее минуты. Сделайте паузу и повторите короче."})

    async def run(self):
        sender = asyncio.create_task(self.sender())
        try:
            await self.send({"type": "connected", "turn_id": self.turn_id})
            while True:
                message = await self.ws.receive()
                if message["type"] == "websocket.disconnect":
                    return
                try:
                    if message.get("bytes") is not None:
                        await self.receive_audio(message["bytes"])
                    elif len(message.get("text", "")) <= 100_000:
                        await self.handle(json.loads(message["text"]))
                    else:
                        raise ValueError("Слишком большое сообщение")
                except (ValueError, TypeError, KeyError) as exc:
                    await self.send({"type": "error", "message": f"Ошибка протокола: {exc}"})
        finally:
            self.closed = True
            self.cancel.set()
            if self.future:
                self.future.cancel()
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sender
