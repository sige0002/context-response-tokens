"""Fetch official SSv2 archives and prepare reproducible frame manifests.

Research-use data stays under data/. No data is published by this script.
Download checks length and records a local SHA256; no upstream checksum exists
in this workflow. Full archive extraction is intentionally a separate step.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import zipfile

import requests

BASE = "https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Something-Something-V2/"
LABELS = "20bn-something-something-download-package-labels.zip"
PARTS = ["20bn-something-something-v2-00", "20bn-something-something-v2-01"]


def download_once(name, root):
    target = root / name
    record = root / (name + ".json")
    if target.exists() and record.exists():
        metadata = json.loads(record.read_text())
        if target.stat().st_size != metadata["bytes"]:
            raise ValueError(f"Completed file size changed: {target}")
        print(f"already downloaded: {name}", flush=True)
        return
    temporary = target.with_name(name + ".partial")
    offset = temporary.stat().st_size if temporary.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(BASE + name, headers=headers, stream=True, timeout=(30, 120)) as r:
        r.raise_for_status()
        if offset and r.status_code == 206:
            if not r.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise ValueError("Server returned the wrong range")
        elif r.status_code == 200:
            offset = 0  # server does not support resume; replace partial safely
        else:
            raise ValueError(f"Unexpected download status: {r.status_code}")
        expected = offset + int(r.headers["Content-Length"])
        size, last_print = offset, time.monotonic()
        with temporary.open("ab" if offset else "wb") as f:
            for block in r.iter_content(4 * 1024 * 1024):
                f.write(block)
                size += len(block)
                if time.monotonic() - last_print > 20:
                    print(f"{name}: {size}/{expected} bytes", flush=True)
                    last_print = time.monotonic()
        if size != expected:
            raise ValueError(f"Incomplete download: {size}/{expected}")
    with temporary.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    temporary.replace(target)
    record.write_text(json.dumps(dict(url=BASE + name, bytes=size, sha256=digest,
                                      upstream_checksum_verified=False), indent=2) + "\n")
    print(f"complete: {name} ({size} bytes)", flush=True)


def download(name, root):
    for attempt in range(8):
        try:
            return download_once(name, root)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError) as error:
            print(f"{name}: interrupted ({type(error).__name__}); resume attempt {attempt + 1}/8", flush=True)
            if attempt == 7:
                raise
            time.sleep(2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data/ssv2")
    p.add_argument("--labels-only", action="store_true")
    args = p.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    download(LABELS, root)
    labels = root / "labels"
    labels.mkdir(exist_ok=True)
    with zipfile.ZipFile(root / LABELS) as z:
        for entry in z.infolist():
            if entry.filename.endswith(".json"):
                # Flatten trusted-format annotation files, never extract paths.
                (labels / Path(entry.filename).name).write_bytes(z.read(entry))
    print("annotations:", sorted(p.name for p in labels.iterdir()), flush=True)
    if not args.labels_only:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda name: download(name, root), PARTS))


if __name__ == "__main__":
    main()
