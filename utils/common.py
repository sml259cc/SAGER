#!/usr/bin/env python3
"""Shared hashing and main-experiment configuration helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def canonical_value(value: Any) -> Any:
    """Return the JSON-normalized form used for identity comparisons."""
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_config_path(path: str | Path, repo: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = repo / candidate
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"config must be a direct file: {candidate}")
    return candidate.resolve()


def load_experiment_config(path: str | Path, repo: Path) -> dict[str, Any]:
    """Load a complete main-experiment configuration."""
    source = resolve_config_path(path, repo)
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    validate_main_config(raw)
    return raw


def validate_main_config(raw: Mapping[str, Any]) -> None:
    """Validate the main-experiment configuration and optimizer settings."""
    if raw.get("schema_version") != "sager_paper_config_v2":
        raise ValueError("Use a sager_paper_config_v2 configuration")
    expected = {"schema_version", "model", "objective", "training", "datasets"}
    if set(raw) != expected:
        raise ValueError(f"Config sections must be {sorted(expected)}")
    training = raw["training"]
    if set(training) != {"batch_dialogues", "joint_epochs", "learning_rate", "weight_decay", "gradient_clip_norm"}:
        raise ValueError("Unexpected or missing training setting")
    if training["learning_rate"] != 8e-5 or training["weight_decay"] != 1e-5:
        raise ValueError("The paper requires AdamW lr=8e-5 and weight_decay=1e-5")


def effective_config_sha256(config: Mapping[str, Any]) -> str:
    """Hash the complete experiment configuration."""
    return canonical_sha256(config)


def source_tree_sha256(repo: Path, files: Sequence[str]) -> str:
    return canonical_sha256({name: file_sha256(repo / name) for name in files})
