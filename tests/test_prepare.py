from types import SimpleNamespace

import pytest

from scripts import prepare


def test_disk_budget_keeps_five_gb_and_does_not_delete_by_default(monkeypatch):
    monkeypatch.setattr(prepare.shutil, "disk_usage", lambda _: SimpleNamespace(free=6 * prepare.GIB))
    prepare.ensure_space(prepare.GIB)
    with pytest.raises(RuntimeError, match="Не хватает"):
        prepare.ensure_space(2 * prepare.GIB)


def test_cleanup_is_blocked_by_active_build(monkeypatch):
    monkeypatch.setattr(prepare.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    monkeypatch.setattr(prepare.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="/usr/bin/xcodebuild\n"))
    with pytest.raises(RuntimeError, match="Идёт сборка"):
        prepare.ensure_space(prepare.GIB, allow_cleanup=True)


def test_cleanup_only_removes_required_old_build_and_preserves_other_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare.Path, "home", lambda: tmp_path)
    cache = tmp_path / "Library/Developer/Xcode/DerivedData"
    old, newer, shared = cache / "Old-build", cache / "New-build", cache / "ModuleCache.noindex"
    for p in (old, newer, shared):
        p.mkdir(parents=True)
        (p / "data").write_text("cache")
    import os
    os.utime(old, (1, 1))
    (cache / "Linked-build").symlink_to(newer, target_is_directory=True)
    monkeypatch.setattr(prepare.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="/Applications/Xcode.app/Xcode\n"))
    monkeypatch.setattr(prepare.shutil, "disk_usage", lambda _: SimpleNamespace(free=0 if old.exists() else 8 * prepare.GIB))
    prepare.ensure_space(prepare.GIB, allow_cleanup=True)
    assert not old.exists()
    assert newer.exists() and shared.exists() and (cache / "Linked-build").is_symlink()
