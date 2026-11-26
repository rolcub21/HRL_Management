"""PointMaze adaptation contract and artifact helpers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
import numpy as np

from projects.tfsn_consequence_value_iteration.shared.settings import (
    MANIFEST_PATH,
    METHOD_MANIFEST,
)


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REPOSITORY = PROJECT.parents[1]
PROTOCOL_PATH = HERE / "protocol.json"


def load_json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_repository(path):
    value = Path(path)
    return value.resolve() if value.is_absolute() else (REPOSITORY / value).resolve()


def protocol():
    value = load_json(PROTOCOL_PATH)
    reference = resolve_repository(value["method_reference"])
    if reference != HERE / "METHOD_REFERENCE.md":
        raise ValueError("PointMaze method reference changed")
    manifest = resolve_repository(value["shared_method_manifest"])
    if manifest != MANIFEST_PATH.resolve():
        raise ValueError("PointMaze does not use the shared method manifest")
    if value["method_revision"] != METHOD_MANIFEST["method_revision"]:
        raise ValueError("PointMaze and the shared method revision differ")
    return value


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)

