"""Prepare every CoTracker3 Kubric scene for 224-input dense-encoder experiments.

Downloads one source archive per worker, verifies its upstream LFS SHA256,
retains all frames/tracks, and removes that temporary archive after verification.
The official source is pinned. A fixed 300-scene validation split is held out
from the 5869 released archives before training; RoboTAP is never used here.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
import tarfile
import time
import zipfile

import numpy as np
from PIL import Image, __version__ as PIL_VERSION
import requests

from kubric_data import load_points, save_points

REPO = "facebook/CoTracker3_Kubric"
REVISION = "a23b6e6e332595c9d92944680a10415e93bfd1e1"
SCENES = 5869  # Enumerated all pages of this pinned release; not a rounded paper count.


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def source_manifest(root):
    path = root / "sources.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["repo"] != REPO or saved["revision"] != REVISION:
            raise ValueError("Prepared source revision differs")
        return saved
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}?limit=1000"
    rows = []
    while url:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        rows.extend(r for r in response.json() if r["path"].endswith(".tar.gz"))
        url = response.links.get("next", {}).get("url")
    rows.sort(key=lambda r: r["path"])
    if [r["path"] for r in rows] != [f"{i:04d}.tar.gz" for i in range(SCENES)]:
        raise ValueError(f"Expected all {SCENES} archives in the pinned release")
    saved = dict(repo=REPO, revision=REVISION, license="Apache-2.0", rows=rows)
    write_json(path, saved)
    return saved


def download(root, row):
    target = root / row["path"]
    expected = row["lfs"]["oid"]
    if target.exists():
        if target.stat().st_size != row["size"] or digest(target) != expected:
            raise ValueError("Existing source archive differs from upstream hash")
        return target
    partial = target.with_suffix(target.suffix + ".partial")
    for attempt in range(4):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset < row["size"]:
                url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{row['path']}"
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                with requests.get(url, headers=headers, stream=True, timeout=(30, 60)) as response:
                    response.raise_for_status()
                    if offset and response.status_code == 206:
                        if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                            raise ValueError("Invalid resumed HTTP range")
                        mode = "ab"
                    else:
                        mode = "wb"
                    with partial.open(mode) as destination:
                        for block in response.iter_content(2**20):
                            destination.write(block)
            if partial.stat().st_size != row["size"] or digest(partial) != expected:
                raise ValueError("Downloaded archive size or SHA256 mismatch")
            partial.replace(target)
            return target
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def validate_prepared(directory, row):
    metadata = json.loads((directory / "metadata.json").read_text())
    if (metadata["source_sha256"] != row["lfs"]["oid"] or metadata["source_revision"] != REVISION
            or metadata["image_size"] != 224 or metadata["format_version"] != 1):
        raise ValueError("Scene preparation settings differ")
    for name, info in metadata["files"].items():
        path = directory / name
        if path.stat().st_size != info["bytes"] or digest(path) != info["sha256"]:
            raise ValueError(f"Prepared data changed: {path}")
    return metadata


def prepare_scene(root, row, reserve_bytes):
    scene = row["path"].removesuffix(".tar.gz")
    directory = root / "scenes" / scene
    if (directory / "metadata.json").exists():
        return validate_prepared(directory, row)
    if shutil.disk_usage(root).free < reserve_bytes + 2 * 2**30:
        raise OSError("Preparation stopped before consuming the reserved free space")
    archive_path = download(root, row)
    temporary = directory.with_name(scene + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    points = occluded = native_size = None
    frames = []
    with tarfile.open(archive_path, "r|gz") as archive, zipfile.ZipFile(temporary / "images.zip", "w", compression=zipfile.ZIP_STORED) as images:
        for member in archive:
            if member.name == f"{scene}/{scene}_trajs_2d.npy":
                points = np.load(io.BytesIO(archive.extractfile(member).read()), allow_pickle=False)
            elif member.name == f"{scene}/{scene}_visibility.npy":
                occluded = np.load(io.BytesIO(archive.extractfile(member).read()), allow_pickle=False)
            elif member.name.startswith(f"{scene}/frames/") and member.name.endswith(".png"):
                name = Path(member.name).name
                if name != f"{len(frames):03d}.png":
                    raise ValueError("Unexpected or unordered source frame")
                image = Image.open(io.BytesIO(archive.extractfile(member).read())).convert("RGB")
                if native_size is not None and native_size != image.size:
                    raise ValueError("Image size changes within scene")
                native_size = image.size
                resized = image.resize((224, 224), Image.Resampling.BILINEAR)
                contents = io.BytesIO()
                resized.save(contents, format="PNG", compress_level=3)
                images.writestr(name, contents.getvalue())
                frames.append(name)
    if points is None or occluded is None or len(frames) != 120 or points.shape != (32768, 120, 2):
        raise ValueError("Incomplete scene or unexpected annotation dimensions")
    try:
        save_points(temporary / "points.npz", points, occluded)
    except ValueError as exc:
        raise ValueError(f"Scene {scene} annotation validation failed: {exc}") from exc
    restored, restored_occ = load_points(temporary / "points.npz")
    if not np.array_equal(restored.view(np.uint8), points.view(np.uint8)) or not np.array_equal(restored_occ, occluded):
        raise ValueError("Annotation storage failed exact round trip")
    with zipfile.ZipFile(temporary / "images.zip") as archive:
        if archive.namelist() != frames or archive.testzip() is not None:
            raise ValueError("PNG archive failed CRC validation")
    metadata = dict(format_version=1, id=scene, source_revision=REVISION, source_sha256=row["lfs"]["oid"],
        frames=len(frames), tracks=len(points), image_size=224, native_width=native_size[0], native_height=native_size[1],
        rgb_transform="PIL RGB, bilinear resize to 224, lossless PNG", pillow=PIL_VERSION,
        point_storage="float32 little-endian byte planes; exact byte round trip checked",
        nonfinite_occluded_track_frames=int((~np.isfinite(points).all(axis=-1)).sum()),
        occlusion="Unchanged upstream visibility array; True means occluded per official loader",
        files={name:dict(bytes=(temporary / name).stat().st_size, sha256=digest(temporary / name)) for name in ("images.zip", "points.npz")})
    write_json(temporary / "metadata.json", metadata)
    temporary.replace(directory)
    archive_path.unlink()  # This task's verified, consumed download cache only.
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/cotracker3_kubric")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--reserve-gib", type=float, default=40)
    parser.add_argument("--limit", type=int, default=0, help="Development preparation only; never writes full split manifests")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or args.reserve_gib < 20 or not 0 <= args.limit <= SCENES:
        parser.error("Invalid resource configuration")
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / "preparation_source"
    snapshot.mkdir(exist_ok=True)
    for name in ("prepare_kubric_points.py", "kubric_data.py", "uv.lock"):
        original, saved = Path(__file__).parent / name, snapshot / name
        if saved.exists() and digest(saved) != digest(original):
            raise ValueError(f"Preparation source changed; review before resuming: {name}")
        if not saved.exists():
            shutil.copyfile(original, saved)
    write_json(root / "execution.json", dict(workers=args.workers, reserve_gib=args.reserve_gib, limit=args.limit,
               source_sha256={name:digest(snapshot / name) for name in ("prepare_kubric_points.py", "kubric_data.py", "uv.lock")}))
    source = source_manifest(root)
    plan = dict(repo=REPO, revision=REVISION, scenes=SCENES, validation_scenes=300, split_seed=42,
                split="Seed-42 permutation of sorted scene IDs; first 300 validation, remaining 5569 training",
                image_size=224, all_120_frames=True, all_32768_tracks=True, label_quantization=False,
                use="Synthetic point-readout training and validation; RoboTAP remains external evaluation only")
    plan_path = root / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Existing data plan differs")
    write_json(plan_path, plan)
    rows = source["rows"][:args.limit] if args.limit else source["rows"]
    completed, started = [], time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(prepare_scene, root, row, args.reserve_gib * 2**30) for row in rows]
        try:
            for future in as_completed(pending):
                completed.append(future.result())
                if len(completed) % 10 == 0 or len(completed) == len(rows):
                    progress = dict(completed=len(completed), requested=len(rows), full_scenes=SCENES,
                                    prepared_bytes=sum(sum(f["bytes"] for f in r["files"].values()) for r in completed),
                                    free_bytes=shutil.disk_usage(root).free, seconds=time.monotonic()-started)
                    write_json(root / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    if args.limit:
        print("Development subset prepared; full training/validation manifests not created", flush=True)
        return
    shuffled = sorted(r["id"] for r in completed)
    random.Random(42).shuffle(shuffled)
    validation = set(shuffled[:300])
    for split in ("train", "validation"):
        selected = [r for r in sorted(completed, key=lambda r:r["id"]) if (r["id"] in validation) == (split == "validation")]
        path = root / f"{split}.jsonl"
        temporary = path.with_suffix(".jsonl.partial")
        temporary.write_text("".join(json.dumps(dict(**r, directory=f"scenes/{r['id']}")) + "\n" for r in selected))
        temporary.replace(path)
    write_json(root / "complete.json", dict(**plan, train_sha256=digest(root / "train.jsonl"),
                                            validation_sha256=digest(root / "validation.jsonl")))


if __name__ == "__main__":
    main()
