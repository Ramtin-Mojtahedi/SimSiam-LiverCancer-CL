#!/usr/bin/env python3
"""Verify package integrity before use."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

EXPECTED_MODELS = {
    "densenet121-kather100k.pth",
    "densenet161-kather100k.pth",
    "densenet169-kather100k.pth",
    "densenet201-kather100k.pth",
    "resnet18-kather100k.pth",
    "resnet34-kather100k.pth",
    "resnet50-kather100k.pth",
    "resnet101-kather100k.pth",
    "resnext50_32x4d-kather100k.pth",
    "resnext101_32x8d-kather100k.pth",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    package_root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    checksum_path = package_root / "CHECKSUMS_SHA256.txt"
    if not checksum_path.exists():
        raise FileNotFoundError(checksum_path)

    errors: list[str] = []
    checked = 0
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected_hash, relative_path = line.split("  ", 1)
        path = package_root / relative_path
        if not path.exists():
            errors.append(f"Missing: {relative_path}")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            errors.append(f"SHA-256 mismatch: {relative_path}: {actual_hash} != {expected_hash}")
        checked += 1

    found_models = {path.name for path in (package_root / "models").glob("*.pth")}
    if found_models != EXPECTED_MODELS:
        errors.append(
            f"Checkpoint inventory mismatch. Found={sorted(found_models)}, expected={sorted(EXPECTED_MODELS)}"
        )
    for model_path in (package_root / "models").glob("*.pth"):
        if model_path.stat().st_size < 1_000_000:
            errors.append(f"Checkpoint unexpectedly small: {model_path}")

    if errors:
        raise RuntimeError("\n".join(errors))
    print(f"PASS: verified {checked} files and all {len(EXPECTED_MODELS)} checkpoints in {package_root}")


if __name__ == "__main__":
    main()
