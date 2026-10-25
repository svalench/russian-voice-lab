"""All MLX operations live on the server's single inference thread."""
import re
import time
import os
from threading import Event

from .config import LLM_MODELS, MODELS, PIPER_FILE, PIPER_PREFIX, PIPER_REPO, SYSTEM_PROMPT, VOICES, local_environment, model_path, select_llm


class Cancelled(Exception):
    pass


def check_cancel(cancel):
    if cancel.is_set():
        raise Cancelled()


def phrase_boundary(text, final=False, first=False):
    """Keep short complete clauses; cap a long unpunctuated phrase at a word."""
    match = re.search(r"[.!?;:\n]\s", text)
    if match:
        return match.end()
    if first and len(text) >= 30 and (match := re.search(r",\s", text)):
        return match.end()
    if len(text) >= (60 if first else 100):
        cut = text.rfind(" ", 25, 55 if first else 90)
        if cut > 0:
            return cut + 1
    return len(text) if final else 0


class MLXEngine:
    def __init__(self, tts_backend=None, llm_profile=None):
        self.llm_profile = select_llm(llm_profile)
        self.tts_backend = tts_backend or os.environ.get("VOICE_LAB_TTS", "piper")
        if self.tts_backend not in ("qwen", "piper"):
            raise ValueError("VOICE_LAB_TTS: ожидается qwen или piper")
        self.model_ids = dict(MODELS)
        self.model_ids["llm"] = LLM_MODELS[self.llm_profile]
        if self.tts_backend == "piper":
            self.model_ids["tts"] = f"{PIPER_REPO}/{PIPER_PREFIX}/{PIPER_FILE}"
        self.cached_tokens = []
        self.prompt_cache = None
        self.history_drop = 0
        self.history_anchor = None
        self.status = {"state": "loading", "models": {k: "waiting" for k in MODELS}, "error": None}

    def load(self):
        local_environment()
        try:
            import mlx.core as mx
            from mlx_audio.stt.utils import load_model as load_asr
            from mlx_audio.tts.utils import load_model as load_tts
            from mlx_lm import load

            self.mx = mx
            for key, loader in (("asr", load_asr), ("llm", load), ("tts", load_tts)):
                self.status["models"][key] = "loading"
                if key == "tts" and self.tts_backend == "piper":
                    import onnxruntime
                    onnxruntime.disable_telemetry_events()
                    from piper import PiperVoice
                    path = model_path("piper") / PIPER_PREFIX / PIPER_FILE
                    if not path.exists():
                        raise RuntimeError("Нет голоса Ирина. Выполните ./scripts/setup.sh --fast-voice")
                    self.tts = PiperVoice.load(path)
                    self.status["models"][key] = "ready"
                    continue
                path = model_path(key, self.llm_profile)
                if not (path / "config.json").exists() or not (path / ".ready").exists():
                    raise RuntimeError(f"Модель {key} не найдена. Выполните ./scripts/setup.sh")
                value = loader(str(path))
                if key == "llm":
                    self.llm, self.tokenizer = value
                else:
                    setattr(self, key, value)
                self.status["models"][key] = "ready"
            if self.asr._tokenizer is None or self.asr._feature_extractor is None:
                raise RuntimeError("Не загружен токенизатор распознавания")
            if self.tts_backend == "qwen":
                if self.tts.tokenizer is None or self.tts.speech_tokenizer is None:
                    raise RuntimeError("Не загружен речевой токенизатор TTS")
                available = {s.lower() for s in self.tts.supported_speakers}
                self.voices = [v for v in VOICES if v.lower() in available]
            else:
                self.voices = ["Ирина"]
            self.memory_stats()
            self.status["state"] = "ready"
        except Exception as exc:
            self.status.update(state="error", error=f"Не удалось загрузить модели: {exc}")

    def transcribe(self, audio, cancel):
        check_cancel(cancel)
        # Qwen3-ASR consumes mono audio at 16 kHz; no temporary recording on disk.
        # Decode the whole token sequence to preserve Cyrillic UTF-8 fragments,
        # and check cancellation between tokens instead of blocking on a turn.
        from mlx_lm.sample_utils import make_logits_processors
        tokens = []
        stream = self.asr.stream_generate(
            audio, language="Russian", max_tokens=min(768, max(64, int(len(audio) / 16000 * 14) + 32)),
            logits_processors=make_logits_processors(repetition_penalty=1.1, repetition_context_size=64),
        )
        try:
            for token, _ in stream:
                check_cancel(cancel)
                tokens.append(int(token))
                if len(tokens) >= 20 and len(set(tokens[-20:])) <= 3:
                    raise RuntimeError("Распознавание зациклилось. Повторите реплику чётче; проверьте микрофон и наушники.")
        finally:
            stream.close()
        check_cancel(cancel)
        return self.asr._tokenizer.decode(tokens, skip_special_tokens=True).strip()

    def messages(self, history, instruction, user):
        anchor = (history[0]["content"] if history else None, instruction)
        if anchor != self.history_anchor or self.history_drop > len(history):
            self.history_drop = 0
            self.history_anchor = anchor
        messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n" + instruction[:2000]}]
        messages.extend(history[self.history_drop:])
        messages.append({"role": "user", "content": user})
        # Keep the prefix between turns. On overflow reclaim half the context,
        # rather than dropping an old turn on EVERY subsequent user message.
        compact = False
        while True:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            size = len(self.tokenizer.encode(prompt))
            if size <= (2048 if compact else 3840) or (compact and len(messages) <= 2 and size <= 3840):
                return prompt
            compact = True
            if len(messages) <= 2:
                raise RuntimeError("Слишком длинная реплика. Скажите короче.")
            if len(messages) > 3:
                del messages[1:3]
                self.history_drop += 2
            else:
                del messages[1:2]
                self.history_drop += 1

    def respond(self, history, instruction, user, cancel):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

        prompt = self.messages(history, instruction, user)
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        common = 0
        for previous, current in zip(self.cached_tokens, tokens[:-1]):
            if previous != current:
                break
            common += 1
        if self.prompt_cache is None:
            self.prompt_cache = make_prompt_cache(self.llm)
        else:
            trim_prompt_cache(self.prompt_cache, self.prompt_cache[0].offset - common)
        self.cached_tokens = self.cached_tokens[:common]
        known = tokens.copy()
        check_cancel(cancel)
        stream = stream_generate(
            self.llm, self.tokenizer, prompt=tokens[common:], prompt_cache=self.prompt_cache, max_tokens=256,
            sampler=make_sampler(temp=0.65, top_p=0.9),
            prefill_step_size=128, prompt_progress_callback=lambda *_: check_cancel(cancel),
        )
        try:
            for response in stream:
                check_cancel(cancel)
                known.append(response.token)
                if response.text:
                    # Some models emit emoji despite the voice prompt; do not
                    # send them to a phonemizer or the neural speech model.
                    yield re.sub(r"[\U0001F000-\U0001FAFF\uFE0F\u200D]", "", response.text)
        finally:
            stream.close()
            # Streaming may precompute one token beyond the last delivered one.
            offset = self.prompt_cache[0].offset
            if offset > len(known):
                trim_prompt_cache(self.prompt_cache, offset - len(known))
            self.cached_tokens = known[:self.prompt_cache[0].offset]

    def speak(self, text, voice, cancel):
        check_cancel(cancel)
        if self.tts_backend == "piper":
            for chunk in self.tts.synthesize(text):
                samples = chunk.audio_float_array
                size = int(chunk.sample_rate * 0.32)
                for offset in range(0, len(samples), size):
                    check_cancel(cancel)
                    yield samples[offset:offset + size], chunk.sample_rate
            return
        stream = self.tts.generate_custom_voice(
            text=text, speaker=voice, language="Russian", stream=True,
            streaming_interval=0.32, max_tokens=min(256, max(64, len(text) * 2)),
        )
        try:
            for result in stream:
                check_cancel(cancel)
                audio = result.audio
                self.mx.eval(audio)
                import numpy as np
                yield np.array(audio, dtype=np.float32).reshape(-1), result.sample_rate
        finally:
            stream.close()
            self.tts.speech_tokenizer.decoder.reset_streaming_state()

    def clear_cache(self):
        self.mx.clear_cache()
        self.memory_stats()

    def memory_stats(self):
        self.status["mlx_active_mb"] = round(self.mx.get_active_memory() / 1024**2)
        self.status["mlx_peak_mb"] = round(self.mx.get_peak_memory() / 1024**2)


def run_turn(engine, audio, history, instruction, voice, cancel, emit, preview=False):
    """Synchronous, cancellable producer. emit applies bounded backpressure."""
    import numpy as np

    metrics = {}
    started = time.perf_counter()
    try:
        # Completed phrases have exact text. For an interrupted phrase recognize
        # only the audio confirmed played by the browser, never the unheard tail.
        history = [dict(message) for message in history]
        for message in history:
            if "_heard_audio" in message:
                heard_audio = message.pop("_heard_audio")
                round_id = message.pop("_round_id")
                tail = engine.transcribe(heard_audio, cancel) if len(heard_audio) >= 1600 else ""
                message["content"] = (message["content"] + " " + tail).strip()
                emit({"type": "history_update", "round_id": round_id, "text": message["content"]})
        history = [m for m in history if m["content"]]
        if preview:
            user = "Привет! Я ваш собеседник. Давайте поговорим по-русски."
            phrases = iter([user])
        else:
            emit({"type": "state", "state": "transcribing"})
            at = time.perf_counter()
            user = engine.transcribe(audio, cancel)
            metrics["asr_ms"] = round((time.perf_counter() - at) * 1000)
            if not user:
                emit({"type": "empty"})
                return
            emit({"type": "transcript", "text": user})
            emit({"type": "state", "state": "thinking"})
            phrases = engine.respond(history, instruction, user, cancel)

        pending = ""
        phrase_id = 0
        first_token = True
        tts_time = 0.0
        audio_time = 0.0
        def synthesize(text):
            nonlocal phrase_id, tts_time, audio_time
            check_cancel(cancel)
            phrase_id += 1
            emit({"type": "phrase", "phrase_id": phrase_id, "text": text})
            at = time.perf_counter()
            for samples, sample_rate in engine.speak(text, voice, cancel):
                check_cancel(cancel)
                if not len(samples):
                    continue
                if "first_audio_ms" not in metrics:
                    metrics["first_audio_ms"] = round((time.perf_counter() - started) * 1000)
                audio_time += len(samples) / sample_rate
                emit({"type": "audio", "phrase_id": phrase_id, "sample_rate": sample_rate,
                      "samples": np.asarray(samples, dtype="<f4").tobytes()})
            tts_time += time.perf_counter() - at
            emit({"type": "phrase_end", "phrase_id": phrase_id})

        try:
            for delta in phrases:
                check_cancel(cancel)
                if first_token:
                    metrics["llm_first_token_ms"] = round((time.perf_counter() - started) * 1000) - metrics.get("asr_ms", 0)
                    first_token = False
                pending += delta
                emit({"type": "text_delta", "text": delta})
                while (end := phrase_boundary(pending, first=phrase_id == 0)):
                    text, pending = pending[:end].strip(), pending[end:]
                    if text:
                        synthesize(text)
            if pending.strip():
                synthesize(pending.strip())
        finally:
            close = getattr(phrases, "close", None)
            if close:
                close()
        metrics.update(tts_ms=round(tts_time * 1000), audio_seconds=round(audio_time, 2),
                       tts_rtf=round(tts_time / audio_time, 3) if audio_time else None,
                       total_ms=round((time.perf_counter() - started) * 1000))
        emit({"type": "metrics", **metrics})
        emit({"type": "done"})
    except Cancelled:
        pass
    except Exception as exc:
        if not cancel.is_set():
            emit({"type": "error", "message": f"Ошибка обработки: {exc}"})
    finally:
        engine.clear_cache()
