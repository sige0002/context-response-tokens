"""Download and extract the official evaluation-only RoboTAP release.

Source: google-deepmind/tapnet, commit ceedd60bb9f6c9f7c74796e4445aff6c6ddf7c9b.
Data/annotations: CC BY 4.0, credited to the RoboTAP authors (ICRA 2024).
No model, training split, or tracking performance is created by this script.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile

import requests

URL = "https://storage.googleapis.com/dm-tapnet/robotap/robotap.zip"
SOURCE = "https://github.com/google-deepmind/tapnet/blob/ceedd60bb9f6c9f7c74796e4445aff6c6ddf7c9b/tapnet/tapvid/README.md"


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(root):
    target = root / "robotap.zip"
    record = root / "download.json"
    if target.exists() and record.exists():
        saved = json.loads(record.read_text())
        if target.stat().st_size != saved["bytes"] or file_hash(target) != saved["sha256"]:
            raise ValueError("Previously downloaded archive changed")
        return target, saved
    head = requests.head(URL, timeout=30)
    head.raise_for_status()
    identity = dict(url=URL, bytes=int(head.headers["Content-Length"]), etag=head.headers["ETag"])
    temporary, metadata = root / "robotap.zip.partial", root / "partial.json"
    if temporary.exists():
        if not metadata.exists() or json.loads(metadata.read_text()) != identity:
            raise ValueError("Partial download belongs to another release")
    else:
        metadata.write_text(json.dumps(identity, indent=2) + "\n")
    for attempt in range(6):
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset == identity["bytes"]:
            break
        if offset > identity["bytes"]:
            raise ValueError("Partial download exceeds archive length")
        try:
            headers = {"If-Match": identity["etag"], "Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with requests.get(URL, headers=headers, stream=True, timeout=(30, 60)) as response:
                response.raise_for_status()
                if offset and (response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                    raise ValueError("Download server did not honor the requested resume offset")
                if not offset and response.status_code != 200:
                    raise ValueError("Unexpected download status")
                if int(response.headers["Content-Length"]) + offset != identity["bytes"]:
                    raise ValueError("Download length changed")
                written, last = offset, time.monotonic()
                with temporary.open("ab" if offset else "wb") as stream:
                    for block in response.iter_content(8 * 1024 * 1024):
                        stream.write(block)
                        written += len(block)
                        if time.monotonic() - last > 20:
                            print(json.dumps(dict(event="download", bytes=written, total=identity["bytes"])), flush=True)
                            last = time.monotonic()
                if written != identity["bytes"]:
                    raise ValueError("Incomplete archive")
            break
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as error:
            print(json.dumps(dict(event="retry", attempt=attempt+1, error=type(error).__name__)), flush=True)
            if attempt == 5:
                raise
            time.sleep(2)
    if temporary.stat().st_size != identity["bytes"]:
        raise ValueError("Download budget exhausted before completion")
    saved = dict(identity, sha256=file_hash(temporary), source=SOURCE,
                 license="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
                 upstream_checksum_verified=False)
    temporary.replace(target)
    record.write_text(json.dumps(saved, indent=2) + "\n")
    return target, saved


def extract(archive, root):
    manifest = root / "extraction.json"
    previous = json.loads(manifest.read_text()) if manifest.exists() else {}
    entries = []
    with zipfile.ZipFile(archive) as zipped:
        files = [entry for entry in zipped.infolist() if not entry.is_dir()]
        names = [Path(entry.filename).name for entry in files]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate archive basenames")
        raw = root / "raw"
        raw.mkdir(exist_ok=True)
        required = sum(entry.file_size for entry in files if not (raw / Path(entry.filename).name).exists())
        if shutil.disk_usage(root).free < required + 1024**3:
            raise OSError("Insufficient disk space for extracted release")
        for entry in files:
            name = Path(entry.filename).name
            if name != "README.md" and not (name.startswith("robotap_split") and name.endswith(".pkl")):
                raise ValueError(f"Unexpected release file: {entry.filename}")
            target = raw / name
            prior = previous.get(name)
            if target.exists() and prior:
                if target.stat().st_size != entry.file_size or file_hash(target) != prior["sha256"]:
                    raise ValueError(f"Extracted data changed: {name}")
                entries.append(prior)
                continue
            temporary = target.with_name(target.name + ".partial")
            digest = hashlib.sha256()
            with zipped.open(entry) as source, temporary.open("wb") as destination:
                # Reading each member to EOF also verifies the archive's CRC32.
                while block := source.read(8 * 1024 * 1024):
                    destination.write(block)
                    digest.update(block)
            if temporary.stat().st_size != entry.file_size:
                raise ValueError(f"Incomplete extraction: {name}")
            temporary.replace(target)
            row = dict(name=name, bytes=entry.file_size, sha256=digest.hexdigest(), archive_crc32=entry.CRC)
            entries.append(row)
            previous[name] = row
            manifest.write_text(json.dumps(previous, indent=2) + "\n")
            print(json.dumps(dict(event="extracted", **row)), flush=True)
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/robotap")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    archive, metadata = download(root)
    files = [] if args.download_only else extract(archive, root)
    print(json.dumps(dict(event="complete", archive=metadata, extracted_files=len(files))), flush=True)


if __name__ == "__main__":
    main()
