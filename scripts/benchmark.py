"""Real-model offline smoke/conversation/soak check. Does not record a microphone."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import socket
import sys
import time
from threading import Event
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import psutil
from voice_lab.config import LLM_MODELS, MODELS
from voice_lab.engine import MLXEngine, run_turn

QUESTIONS = [
    "Привет! Давай познакомимся. Меня зовут Валентин.",
    "Как меня зовут? Ответь коротко.",
    "Сколько будет двадцать пять плюс семнадцать?",
    "Расскажи одним предложением о Минске.",
    "А где находится этот город?",
    "Посоветуй занятие на дождливый вечер дома.",
    "А если я хочу провести вечер с друзьями?",
    "Повтори дату: двадцать пятое октября две тысячи двадцать шестого года.",
    "Что лучше взять на прогулку, если обещают дождь?",
    "Спасибо за разговор. Пожелай мне хорошего дня.",
]


def block_network(*args, **kwargs):
    raise RuntimeError("Сетевое соединение заблокировано для offline-проверки")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=1)
    parser.add_argument("--minutes", type=float, default=0)
    parser.add_argument("--output", default=".cache/benchmark.json")
    parser.add_argument("--tts", choices=["qwen", "piper"], default="piper")
    parser.add_argument("--llm", choices=list(LLM_MODELS))
    parser.add_argument("--input-cache", help="Reuse identical synthetic WAV inputs across model comparisons")
    args = parser.parse_args()
    report = {"models": MODELS, "platform": platform.platform(), "offline": True, "input": "synthetic TTS loopback, not human microphone", "turns": []}
    started = time.perf_counter()
    engine = MLXEngine(tts_backend=args.tts, llm_profile=args.llm)
    report["models"] = engine.model_ids
    report["tts_backend"] = args.tts
    report["llm_profile"] = engine.llm_profile
    with patch.object(socket.socket, "connect", block_network), patch.object(socket.socket, "connect_ex", block_network):
        engine.load()
        report["load_seconds"] = round(time.perf_counter() - started, 2)
        if engine.status["state"] != "ready":
            raise RuntimeError(engine.status["error"])
        print(f"Models loaded in {report['load_seconds']} s", flush=True)
        history = []
        began = time.perf_counter()
        index = 0
        while index < args.turns or time.perf_counter() - began < args.minutes * 60:
            text = QUESTIONS[index % len(QUESTIONS)]
            cancel = Event()
            # Build input with the same neural voice. This verifies plumbing,
            # not independent ASR accuracy or human-perceived pronunciation.
            import soundfile as sf
            fixture = None
            if args.input_cache:
                key = hashlib.sha256((engine.model_ids['tts'] + engine.voices[0] + text).encode()).hexdigest()[:16]
                directory = ROOT / args.input_cache
                directory.mkdir(parents=True, exist_ok=True)
                fixture = directory / f"{key}.wav"
            reused = fixture is not None and fixture.exists()
            if reused:
                audio, rate = sf.read(fixture, dtype="float32")
            else:
                input_parts = []
                for audio, rate in engine.speak(text, engine.voices[0], cancel):
                    input_parts.append(audio)
                    if len(input_parts) == 1:
                        print(f"Turn {index + 1}: input TTS first chunk", flush=True)
                if not input_parts:
                    raise RuntimeError("TTS не вернул входное аудио")
                audio = np.concatenate(input_parts)
                if fixture is not None:
                    sf.write(fixture, audio, rate, subtype="FLOAT")
            print(f"Turn {index + 1}: input audio ready ({len(audio) / rate:.2f} s)", flush=True)
            if index == 0:
                sf.write(ROOT / ".cache/synthetic-input.wav", audio, rate)
            if rate != 16000:
                audio = np.interp(np.arange(0, len(audio), rate / 16000), np.arange(len(audio)), audio).astype(np.float32)
            events = []
            def emit(event):
                if event["type"] != "audio":
                    events.append(event)
            run_turn(engine, audio, history, "", engine.voices[0], cancel, emit)
            failure = next((x for x in events if x["type"] == "error"), None)
            if failure:
                raise RuntimeError(failure["message"])
            user = next((x["text"] for x in events if x["type"] == "transcript"), "")
            answer = "".join(x["text"] for x in events if x["type"] == "text_delta")
            if not user or not answer or not any(x["type"] == "done" for x in events):
                raise RuntimeError("Пустой результат или незавершённый ответ")
            history.extend([{"role": "user", "content": user}, {"role": "assistant", "content": answer}])
            metrics = next(x for x in events if x["type"] == "metrics")
            record = {"input": text, "recognized": user, "answer": answer, "metrics": metrics,
                      "input_fixture": str(fixture.relative_to(ROOT)) if fixture is not None else None,
                      "input_reused": reused,
                      "mlx_active_mb": engine.status.get("mlx_active_mb"),
                      "mlx_peak_mb": engine.status.get("mlx_peak_mb"),
                      "rss_mb": round(psutil.Process().memory_info().rss / 1024**2),
                      "available_mb": round(psutil.virtual_memory().available / 1024**2)}
            report["turns"].append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            index += 1
            report["conversation_seconds"] = round(time.perf_counter() - began, 2)
            path = ROOT / args.output
            path.parent.mkdir(exist_ok=True, parents=True)
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Отчёт: {ROOT / args.output}")


if __name__ == "__main__":
    main()
