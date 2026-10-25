#!/bin/zsh
set -euo pipefail
cd "${0:A:h:h}"
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"
export HF_HOME="$PWD/.cache/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
fast_voice=1
cleanup=()
for arg in "$@"; do
  case "$arg" in
    --fast-voice) fast_voice=1 ;;
    --qwen-voice) fast_voice=0 ;;
    --clean-build-cache) cleanup=(--clean-build-cache) ;;
    *) print -u2 "Неизвестный параметр: $arg"; exit 1 ;;
  esac
done
if [[ "$(uname -m)" != "arm64" ]]; then
  print -u2 "Требуется Mac с Apple Silicon; запустите терминал без Rosetta."
  exit 1
fi
command -v uv >/dev/null || { print -u2 "Установите uv: brew install uv"; exit 1; }
command -v npm >/dev/null || { print -u2 "Установите Node.js: brew install node"; exit 1; }
python3 scripts/prepare.py preflight "${cleanup[@]}"
if (( fast_voice )); then
  uv sync --frozen --extra fast-voice
else
  uv sync --frozen
fi
npm ci --cache .cache/npm --ignore-scripts --no-audit --no-fund
.venv/bin/python scripts/prepare.py assets
if (( fast_voice )); then
  .venv/bin/python scripts/prepare.py models --skip-qwen-tts "${cleanup[@]}"
  .venv/bin/python scripts/download_piper.py
else
  .venv/bin/python scripts/prepare.py models "${cleanup[@]}"
fi
print "Готово. Запустите ./scripts/run.sh и откройте http://127.0.0.1:8765"
