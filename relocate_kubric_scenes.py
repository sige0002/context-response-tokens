"""Relocate immutable, completed Kubric scenes to SSD1 without changing paths.

Only metadata-committed scenes are eligible. Verify recorded hashes before and
after copying, then atomically exchange the source directory with a symlink.
The running preparer only writes new .partial scenes, which are excluded.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
from datetime import datetime, timezone


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate(directory, metadata, metadata_hash):
    assert set(p.name for p in directory.iterdir()) == {"metadata.json", "images.zip", "points.npz"}
    assert digest(directory / "metadata.json") == metadata_hash
    for name, record in metadata["files"].items():
        path = directory / name
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == record["bytes"]
        assert digest(path) == record["sha256"], str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    assert args.limit > 0
    root = Path(__file__).resolve().parent / "data/cotracker3_kubric/scenes"
    mount = Path("/media/sadasue/SSD1")
    assert mount.is_mount()
    destination = mount / "mywam/data/cotracker3_kubric/scenes"
    destination.mkdir(parents=True, exist_ok=True)
    report = Path(args.report)
    assert not report.exists()
    candidates = [p for p in sorted(root.iterdir()) if p.name.isdecimal()
                  and p.is_dir() and not p.is_symlink() and (p / "metadata.json").is_file()][:args.limit]
    assert candidates
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    journal = report.with_suffix(".jsonl")
    completed, copied_bytes = [], 0
    before = shutil.disk_usage(root).free
    for source in candidates:
        metadata = json.loads((source / "metadata.json").read_text())
        assert metadata["id"] == source.name
        assert set(metadata["files"]) == {"images.zip", "points.npz"}
        metadata_hash = digest(source / "metadata.json")
        validate(source, metadata, metadata_hash)
        stamps = {p.name:(p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) for p in source.iterdir()}
        target = destination / source.name
        temporary = destination / (source.name + ".copying")
        if not target.exists():
            assert not temporary.exists(), f"Unfinished copy requires inspection: {temporary}"
            shutil.copytree(source, temporary, copy_function=shutil.copy2)
            validate(temporary, metadata, metadata_hash)
            temporary.rename(target)
        validate(target, metadata, metadata_hash)
        assert stamps == {p.name:(p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) for p in source.iterdir()}
        link = root / ("." + source.name + ".ssd1_exchange")
        assert not link.exists() and not link.is_symlink()
        link.symlink_to(target, target_is_directory=True)
        row = dict(scene=source.name, target=str(target), metadata_sha256=metadata_hash,
                   files=metadata["files"], status="verified_before_exchange")
        with journal.open("a") as log:
            log.write(json.dumps(row)+"\n")
            log.flush()
            os.fsync(log.fileno())
        # RENAME_EXCHANGE preserves a valid source path throughout the switch.
        if rename(-100, os.fsencode(link), -100, os.fsencode(source), 2):
            raise OSError(ctypes.get_errno(), "Atomic directory/link exchange failed")
        assert source.is_symlink() and source.resolve() == target
        assert link.is_dir() and not link.is_symlink()
        assert digest(source / "metadata.json") == metadata_hash
        shutil.rmtree(link)
        copied_bytes += sum(f["bytes"] for f in metadata["files"].values())
        completed.append(source.name)
        with journal.open("a") as log:
            log.write(json.dumps(dict(scene=source.name, status="relocated"))+"\n")
        state = dict(observed_at=datetime.now(timezone.utc).isoformat(), status="running",
                     requested=len(candidates), completed=len(completed), content_bytes=copied_bytes,
                     destination=str(destination), original_paths_preserved=True,
                     root_free_bytes=shutil.disk_usage(root).free,
                     ssd1_free_bytes=shutil.disk_usage(mount).free)
        if len(completed)%25 == 0:
            report.write_text(json.dumps(state, indent=2)+"\n")
            print(json.dumps(state), flush=True)
    state.update(status="complete", scenes=completed, root_free_bytes_before=before,
                 source_sha256=digest(__file__))
    report.write_text(json.dumps(state, indent=2)+"\n")
    print(json.dumps({k:v for k,v in state.items() if k!='scenes'}), flush=True)


if __name__ == "__main__":
    main()
