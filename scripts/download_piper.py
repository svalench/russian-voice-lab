"""Optional small, Russian neural voice; never used as a silent fallback."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
from voice_lab.config import PIPER_FILE, PIPER_PREFIX, PIPER_REPO
from scripts.prepare import ensure_space

manifest_path = ROOT / ".models/manifest.json"
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
revision = manifest.get("piper", {}).get("revision", "main")
metadata = get_hf_file_metadata(hf_hub_url(PIPER_REPO, f"{PIPER_PREFIX}/{PIPER_FILE}", revision=revision))
revision = metadata.commit_hash
path = ROOT / ".models/piper"
ensure_space((metadata.size or 70_000_000) if not (path / PIPER_PREFIX / PIPER_FILE).exists() else 0)
for name in (PIPER_FILE, PIPER_FILE + ".json", "MODEL_CARD"):
    hf_hub_download(PIPER_REPO, f"{PIPER_PREFIX}/{name}", revision=revision, local_dir=path)
manifest["piper"] = {"repo": PIPER_REPO, "revision": revision, "voice": PIPER_PREFIX, "bytes": metadata.size}
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
(path / ".ready").write_text(revision + "\n")
print("Русский нейроголос Ирина готов. Запуск: VOICE_LAB_TTS=piper ./scripts/run.sh")
