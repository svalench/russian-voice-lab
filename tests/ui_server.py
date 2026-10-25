"""Explicit test-only server; the production app never substitutes fake models."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.test_conversation import FakeEngine
from voice_lab.server import create_app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(create_app(FakeEngine()), host="127.0.0.1", port=8766, log_level="warning")
