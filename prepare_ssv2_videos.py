"""Extract official train/validation videos for full-data temporal training.

Keeps compressed original videos, allowing temporal sampling during training.
Data stays under data/ and must not be redistributed. No test labels are used.
"""
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time

from extract_ssv2 import JoinedParts
from prepare_ssv2 import PARTS


def main():
    root = Path("data/ssv2")
    output = root / "videos_trainval"
    directory = output / "videos"
    directory.mkdir(parents=True, exist_ok=True)
    labels = json.loads((root / "labels/labels.json").read_text())
    splits = {}
    for split, name in (("train", "train"), ("val", "validation")):
        rows = json.loads((root / f"labels/{name}.json").read_text())
        splits[split] = [dict(id=row["id"], group=row["id"],
                             label=int(labels[row["template"].replace("[", "").replace("]", "")]),
                             video=f"videos/{row['id']}.webm") for row in rows]
    ids = [{r["id"] for r in rows} for rows in splits.values()]
    assert not ids[0] & ids[1]
    wanted = ids[0] | ids[1]
    assert all(i.isdecimal() for i in wanted)
    plan = dict(scope="full official training and validation; no held-out test score",
                counts={s: len(rows) for s, rows in splits.items()},
                annotation_sha256={name: hashlib.sha256((root / f"labels/{name}.json").read_bytes()).hexdigest()
                                   for name in ("labels", "train", "validation")})
    plan_path = output / "selection.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Existing output has different annotations")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    parts = [root / name for name in PARTS]
    for path in parts:
        if path.stat().st_size != json.loads(path.with_name(path.name + ".json").read_text())["bytes"]:
            raise ValueError(f"Incomplete source archive: {path}")
    found, written, reused = {}, 0, 0
    start = time.monotonic()
    joined = JoinedParts(parts)
    try:
        with tarfile.open(fileobj=joined, mode="r|gz") as archive:
            for member in archive:
                name = Path(member.name)
                if not member.isfile() or name.suffix != ".webm" or name.stem not in wanted:
                    continue
                clip_id = name.stem
                if clip_id in found:
                    raise ValueError(f"Repeated archive video: {clip_id}")
                target = directory / f"{clip_id}.webm"
                if target.exists() and target.stat().st_size == member.size:
                    contents = target.read_bytes()
                    reused += 1
                else:
                    with archive.extractfile(member) as source:
                        contents = source.read()
                    if len(contents) != member.size:
                        raise ValueError(f"Truncated archive entry: {clip_id}")
                    temporary = target.with_suffix(".webm.partial")
                    temporary.write_bytes(contents)
                    temporary.replace(target)
                    written += 1
                found[clip_id] = dict(bytes=member.size, sha256=hashlib.sha256(contents).hexdigest())
                if len(found) % 5000 == 0:
                    progress = dict(extracted=len(found), total=len(wanted), written=written, reused=reused,
                                    seconds=time.monotonic() - start)
                    (output / "progress.json").write_text(json.dumps(progress) + "\n")
                    print(json.dumps(progress), flush=True)
    finally:
        joined.close()
        # Drop only this task's already-consumed archive cache. This avoids
        # starving fresh CUDA contexts of free RAM on the unified-memory GB10.
        for path in parts:
            with path.open("rb") as source:
                os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f"Missing {len(missing)} official videos: {sorted(missing)[:10]}")
    for split, rows in splits.items():
        temporary = output / f"{split}.jsonl.partial"
        with temporary.open("w") as stream:
            for row in rows:
                stream.write(json.dumps(dict(**row, **found[row["id"]])) + "\n")
        temporary.replace(output / f"{split}.jsonl")
    final = dict(status="complete", **plan, bytes=sum(r["bytes"] for r in found.values()),
                 seconds=time.monotonic() - start)
    (output / "complete.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
