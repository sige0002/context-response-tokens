"""Install a verified local pixi executable for this Linux aarch64 paper build."""
import hashlib
import io
import json
from pathlib import Path
import platform
import tarfile
import urllib.request


def main():
    root = Path(__file__).resolve().parent
    info = json.loads((root/'build_tools.json').read_text())
    if platform.system() != 'Linux' or platform.machine() not in ('aarch64', 'arm64'):
        raise RuntimeError('Use an officially installed pixi on other platforms; this bootstrap is linux-aarch64 only')
    destination = root/'.tools/pixi'
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() != info['pixi_binary_sha256']:
            raise ValueError('Existing pixi differs; refusing to overwrite it')
        print(destination)
        return
    url = f"https://github.com/prefix-dev/pixi/releases/download/v{info['pixi_version']}/pixi-aarch64-unknown-linux-musl.tar.gz"
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != info['pixi_archive_sha256']:
        raise ValueError('Downloaded archive differs from release digest')
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        matches = [m for m in archive.getmembers() if Path(m.name).name == 'pixi' and m.isfile()]
        if len(matches) != 1:
            raise ValueError('Unexpected release archive')
        executable = archive.extractfile(matches[0]).read()
    if hashlib.sha256(executable).hexdigest() != info['pixi_binary_sha256']:
        raise ValueError('Executable differs from release digest')
    destination.parent.mkdir(exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    temporary.write_bytes(executable)
    temporary.chmod(0o755)
    temporary.replace(destination)
    print(destination)


if __name__ == '__main__':
    main()
