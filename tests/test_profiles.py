import pytest

from voice_lab.config import LLM_MODELS, model_path, select_llm
from voice_lab.engine import MLXEngine


def test_default_small_model_and_local_neural_voice(monkeypatch):
    monkeypatch.delenv("VOICE_LAB_LLM", raising=False)
    monkeypatch.delenv("VOICE_LAB_TTS", raising=False)
    engine = MLXEngine()
    assert engine.llm_profile == "small"
    assert engine.model_ids["llm"] == LLM_MODELS["small"]
    assert engine.tts_backend == "piper"


def test_profiles_use_separate_weights_and_allow_rollback(monkeypatch):
    monkeypatch.setenv("VOICE_LAB_LLM", "small")
    assert model_path("llm").name == "llm-small"
    engine = MLXEngine(llm_profile="4b")
    assert engine.model_ids["llm"] == LLM_MODELS["4b"]
    assert model_path("llm", "4b").name == "llm"
    assert model_path("asr", "small") == model_path("asr", "4b")
    assert model_path("piper", "small") == model_path("piper", "4b")


def test_invalid_profile_fails_before_loading_models(monkeypatch):
    monkeypatch.setenv("VOICE_LAB_LLM", "unknown")
    with pytest.raises(ValueError, match="VOICE_LAB_LLM"):
        select_llm()
