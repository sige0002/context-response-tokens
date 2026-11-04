"""Prepare an isolated uv project for the published GatedMambaMix baseline.

The active Mamba-3 project remains unchanged. Copy its locked dependency set
and pinned Mamba source, then limit the optional Mamba-1 CUDA build to GB10's
SM121 target and use C++20 as required by the installed PyTorch headers.
Only compiler settings change; model and kernel math remain unchanged.
"""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / "baselines/gmmix"
MAMBA_COMMIT = "e9594ce1c732d97440f0332fdc43170a2294dbfa"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source = ROOT / "vendor/mamba"
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != MAMBA_COMMIT:
        raise ValueError("Mamba source is not the pinned reference version")
    DESTINATION.mkdir(parents=True, exist_ok=False)
    target = DESTINATION / "vendor/mamba"
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(
        ".git", "__pycache__", ".pytest_cache", "build", "dist", "*.egg-info", "*.so"))
    original = (target / "setup.py").read_text()
    marker = "    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI"
    if original.count(marker) != 1:
        raise ValueError("Unexpected upstream build script; review compiler-target patch")
    replacement = ("    # Local build target only: GB10 (SM121), same selective-scan sources.\n"
                   "    if not HIP_BUILD:\n"
                   "        cc_flag = [\"-gencode\", \"arch=compute_121,code=sm_121\"]\n\n" + marker)
    patched = original.replace(marker, replacement).replace("-std=c++17", "-std=c++20")
    (target / "setup.py").write_text(patched)
    (DESTINATION / "sm121_build.patch").write_text("".join(difflib.unified_diff(
        original.splitlines(True), patched.splitlines(True), fromfile="upstream/setup.py", tofile="local/setup.py")))
    for name in ("pyproject.toml", "uv.lock"):
        text = (ROOT / name).read_text()
        if text.count('name = "mamba3-sandbox"') != 1:
            raise ValueError("Unexpected source project identity")
        (DESTINATION / name).write_text(text.replace('name = "mamba3-sandbox"', 'name = "gmmix-reference"'))
    manifest = {}
    for path in sorted(target.rglob("*")):
        if path.is_file():
            name = str(path.relative_to(target))
            expected, actual = sha(source / name), sha(path)
            if name != "setup.py" and actual != expected:
                raise ValueError(f"Unexpected model/kernel source edit: {name}")
            manifest[name] = dict(upstream_sha256=expected, local_sha256=actual)
    info = dict(scope="Isolated baseline build preparation, not a trained baseline or performance result",
        upstream_commit=commit, source_files=manifest,
        inherited_environment={name:sha(ROOT / name) for name in ("pyproject.toml", "uv.lock")},
        local_environment={name:sha(DESTINATION / name) for name in ("pyproject.toml", "uv.lock")},
        build_target="CUDA 13.0 / SM121 only / C++20", math_changes=False,
        required_environment=dict(CUDA_HOME="/usr/local/cuda", MAMBA_KEEP_CUDA_BUILD="TRUE",
                                  MAMBA_FORCE_BUILD="TRUE", MAX_JOBS="2"),
        optional_causal_conv1d="Not installed by this preparation; Mamba-1 can use its PyTorch causal convolution fallback")
    (DESTINATION / "source_manifest.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(dict(project=str(DESTINATION), copied_files=len(manifest), modified_source_files=["setup.py"])))


if __name__ == "__main__":
    main()
