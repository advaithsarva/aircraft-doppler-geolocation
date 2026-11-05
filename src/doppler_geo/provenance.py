"""Stable content fingerprints and numerical environment metadata."""

import hashlib
import importlib.metadata
import platform
from pathlib import Path


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment():
    packages = {}
    for name in ["numpy", "scipy", "pyproj", "soundfile", "polars", "pyarrow", "duckdb"]:
        packages[name] = importlib.metadata.version(name)
    return {"python": platform.python_version(), "packages": packages}
