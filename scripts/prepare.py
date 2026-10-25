#!/usr/bin/env python3
"""Preparation: measured disk budget, narrowly scoped cache cleanup, local assets."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voice_lab.config import LLM_MODELS, MODELS, model_key, model_path, select_llm

GIB = 1024 ** 3
RESERVE = 5 * GIB


def ensure_space(required, allow_cleanup=False):
    free = shutil.disk_usage(ROOT).free
    print(f"Свободно {free / GIB:.1f} ГБ; требуется {required / GIB:.1f} ГБ + запас 5 ГБ.", flush=True)
    if free >= required + RESERVE:
        return
    if allow_cleanup:
        processes = subprocess.run(["ps", "-axo", "comm="], capture_output=True, text=True, check=True).stdout
        active = {"xcodebuild", "swift-frontend", "swiftc", "clang", "clang++", "XCBBuildService"}
        if any(Path(line.strip()).name in active for line in processes.splitlines()):
            raise RuntimeError("Идёт сборка Xcode/Swift. Очистка DerivedData пропущена; завершите сборку и повторите.")
        cache = Path.home() / "Library/Developer/Xcode/DerivedData"
        # Only per-project rebuildable directories. Keep module caches and links.
        candidates = sorted((p for p in cache.iterdir() if p.is_dir() and not p.is_symlink() and "-" in p.name), key=lambda p: p.stat().st_mtime) if cache.exists() else []
        for path in candidates:
            if shutil.disk_usage(ROOT).free >= required + RESERVE:
                break
            print(f"Удаляется пересоздаваемый кэш сборки: {path}", flush=True)
            shutil.rmtree(path)
    free = shutil.disk_usage(ROOT).free
    if free < required + RESERVE:
        raise RuntimeError(f"Не хватает {(required + RESERVE - free) / GIB:.1f} ГБ. Освободите место; личные файлы не удалялись.")


def assets():
    destination = ROOT / "web/assets"
    vad = ROOT / "node_modules/@ricky0123/vad-web/dist"
    ort = ROOT / "node_modules/onnxruntime-web/dist"
    for target in ("vad", "ort"):
        (destination / target).mkdir(parents=True, exist_ok=True)
    for name in ("bundle.min.js", "bundle.min.js.LICENSE.txt", "vad.worklet.bundle.min.js", "silero_vad_v5.onnx"):
        shutil.copy2(vad / name, destination / "vad" / name)
    for name in ("ort.wasm.min.js", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"):
        shutil.copy2(ort / name, destination / "ort" / name)
    # Keep upstream licenses alongside the vendored local files.
    for package, target in ((vad.parent, "vad"), (ort.parent, "ort")):
        for license in package.glob("LICENSE*"):
            if license.is_file():
                shutil.copy2(license, destination / target / license.name)
    print("Silero VAD и ONNX Runtime скопированы в web/assets (без CDN).", flush=True)


def models(allow_cleanup=False, llm_profile=None, only_llm=False, skip_qwen_tts=False):
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    cache = ROOT / ".models"
    cache.mkdir(exist_ok=True)
    manifest_file = cache / "manifest.json"
    manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}
    profile = select_llm(llm_profile)
    repos = dict(MODELS, llm=LLM_MODELS[profile])
    if only_llm:
        repos = {"llm": repos["llm"]}
    elif skip_qwen_tts:
        repos.pop("tts")
    infos = {}
    remaining = 0
    for key, repo in repos.items():
        record = manifest.get(model_key(key, profile), {})
        revision = record.get("revision") if record.get("repo") == repo else None
        info = api.model_info(repo, revision=revision, files_metadata=True)
        infos[key] = info
        for sibling in info.siblings:
            local = model_path(key, profile) / sibling.rfilename
            size = sibling.size or 0
            if not local.exists() or local.stat().st_size != size:
                remaining += size
    ensure_space(remaining + 512 * 1024**2, allow_cleanup)
    for key, repo in repos.items():
        info = infos[key]
        print(f"Загрузка {repo} @ {info.sha}", flush=True)
        path = model_path(key, profile)
        snapshot_download(repo, revision=info.sha, local_dir=path, max_workers=3)
        manifest[model_key(key, profile)] = {"repo": repo, "revision": info.sha, "bytes": sum(s.size or 0 for s in info.siblings)}
        (path / ".ready").write_text(info.sha + "\n")
        manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Все модели готовы к запуску без интернета.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "assets", "models"])
    parser.add_argument("--clean-build-cache", action="store_true")
    parser.add_argument("--llm", choices=list(LLM_MODELS))
    parser.add_argument("--only-llm", action="store_true")
    parser.add_argument("--skip-qwen-tts", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "preflight":
            # An existing environment/weights consume their budget already.
            used = sum(p.stat().st_size for folder in (".models", ".venv", ".cache", "node_modules")
                       for p in (ROOT / folder).rglob("*") if p.is_file())
            ensure_space(max(0, 9 * GIB - used), args.clean_build_cache)
        elif args.command == "assets":
            assets()
        else:
            models(args.clean_build_cache, args.llm, args.only_llm, args.skip_qwen_tts)
    except Exception as exc:
        print(f"Подготовка остановлена: {exc}", file=sys.stderr)
        sys.exit(1)
