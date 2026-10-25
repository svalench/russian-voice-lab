#!/bin/zsh
set -euo pipefail
cd "${0:A:h:h}"
if [[ ! -x .venv/bin/python ]]; then
  print -u2 "Сначала выполните ./scripts/setup.sh"
  exit 1
fi
export HF_HOME="$PWD/.cache/huggingface"
export VOICE_LAB_LLM="${VOICE_LAB_LLM:-small}" VOICE_LAB_TTS="${VOICE_LAB_TTS:-piper}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
exec .venv/bin/python -m uvicorn voice_lab.server:app --host 127.0.0.1 --port 8765 --ws websockets --ws-max-size 131072 --no-access-log
