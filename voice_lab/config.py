from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
LLM_MODELS = {
    "small": "mlx-community/Qwen3-1.7B-4bit",
    "4b": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
}
MODELS = {
    "asr": "mlx-community/Qwen3-ASR-0.6B-4bit",
    "llm": LLM_MODELS["small"],
    "tts": "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit",
}
VOICES = ["Ryan", "Aiden", "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ono_Anna", "Sohee"]
PIPER_REPO = "rhasspy/piper-voices"
PIPER_PREFIX = "ru/ru_RU/irina/medium"
PIPER_FILE = "ru_RU-irina-medium.onnx"
SYSTEM_PROMPT = (
    "Ты дружелюбный собеседник. Всегда отвечай на русском языке, естественно и кратко, "
    "обычно одним-тремя предложениями. Не используй Markdown, списки и эмодзи: "
    "твой ответ будет произнесён вслух. Не выдумывай факты."
)


def local_environment():
    os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def select_llm(profile=None):
    profile = profile or os.environ.get("VOICE_LAB_LLM", "small")
    if profile not in LLM_MODELS:
        raise ValueError("VOICE_LAB_LLM: ожидается small или 4b")
    return profile


def model_key(name, llm_profile=None):
    if name == "llm" and select_llm(llm_profile) == "small":
        return "llm-small"
    return name


def model_path(name, llm_profile=None):
    return ROOT / ".models" / model_key(name, llm_profile)
