#!/usr/bin/env python3
"""Evaluate a SAGER checkpoint on a text+audio split."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from model.sager import SAGERConfig, SAGERModel
from utils import dataloader as D
from utils.common import (
    canonical_sha256,
    canonical_value,
    effective_config_sha256,
    file_sha256,
    load_experiment_config,
    resolve_config_path,
    source_tree_sha256,
)

SOURCE_FILES = (
    "model/__init__.py",
    "utils/__init__.py",
    "model/module_strength.py",
    "model/components.py",
    "model/backbone.py",
    "model/sager.py",
    "utils/dataloader.py",
    "train.py",
    "evaluate.py",
    "utils/common.py",
    "main.py",
)


def _source_sha256() -> str:
    return source_tree_sha256(REPO, SOURCE_FILES)


def _config_instance(values: Mapping[str, Any]) -> SAGERConfig:
    prepared = dict(values)
    for key in ("temporal_kernels", "path_local_scales"):
        if key in prepared:
            prepared[key] = tuple(prepared[key])
    return SAGERConfig(**prepared)


def _normalized_model_config(values: Mapping[str, Any]) -> dict[str, Any]:
    normalized = canonical_value(dataclasses.asdict(_config_instance(values)))
    return normalized


def _effective_model_values(raw: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    valid = {item.name for item in dataclasses.fields(SAGERConfig)}
    unknown = sorted(set(raw["model"]) - valid)
    if unknown:
        raise ValueError(f"unknown model config keys: {unknown}")
    values = {key: value for key, value in raw["model"].items() if key in valid}
    values["num_classes"] = len(spec["class_names"])
    return values


def _assert_binding(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} binding differs")


def _checkpoint_seed(payload: Mapping[str, Any], requested: int | None) -> int:
    value = payload.get("seed")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("checkpoint seed is invalid")
    if requested is not None:
        _assert_binding(value, requested, "checkpoint seed")
    return value


def _validate_checkpoint_binding(
    payload: Mapping[str, Any],
    *,
    dataset: str,
    requested_seed: int | None,
    effective_sha256: str,
    expected_model_config: Mapping[str, Any],
    source_sha256: str,
    modalities: list[str],
) -> tuple[dict[str, Any], int]:
    _assert_binding(payload.get("dataset_id"), dataset, "checkpoint dataset")
    seed = _checkpoint_seed(payload, requested_seed)
    if not isinstance(payload.get("state_dict"), Mapping):
        raise ValueError("checkpoint state_dict is missing")
    try:
        observed_model_config = _normalized_model_config(payload["model_config"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint model config is invalid") from exc
    expected_model_sha256 = canonical_sha256(expected_model_config)
    _assert_binding(
        canonical_sha256(observed_model_config),
        expected_model_sha256,
        "checkpoint normalized model config",
    )

    _assert_binding(payload.get("schema_version"), "sager_checkpoint_v3", "checkpoint schema")
    _assert_binding(payload.get("model_id"), "SAGER", "checkpoint model")
    _assert_binding(payload.get("effective_config_sha256"), effective_sha256, "checkpoint effective config")
    _assert_binding(payload.get("model_config_sha256"), expected_model_sha256, "checkpoint model config hash")
    _assert_binding(payload.get("source_sha256"), source_sha256, "checkpoint source")
    _assert_binding(payload.get("modalities"), modalities, "checkpoint modalities")
    return observed_model_config, seed


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_under_repo(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    candidate = (REPO / path).resolve()
    return candidate if candidate.exists() or not path.exists() else path.resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("iemocap_legacy7433", "meld_official"))
    parser.add_argument("--split", default="test", choices=("train", "dev", "test"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=None, help="Optional nonnegative seed binding")
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.config is None:
        retained = resolve_under_repo(args.checkpoint).parent / "effective_config.yaml"
        args.config = retained if retained.is_file() else REPO / "configs" / (args.dataset + ".yaml")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.seed is not None and args.seed < 0:
        raise ValueError("seed must be nonnegative")
    args.config = resolve_config_path(args.config, REPO)
    raw = load_experiment_config(args.config, REPO)
    spec = raw["datasets"][args.dataset]
    effective_sha256 = effective_config_sha256(raw)

    expected_model_config = _normalized_model_config(_effective_model_values(raw, spec))

    model_config_sha256 = canonical_sha256(expected_model_config)
    modalities = ["text", "audio"]
    source_sha256 = _source_sha256()
    bundle = resolve_under_repo(args.bundle or Path("features") / args.dataset)
    checkpoint_path = resolve_under_repo(args.checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    if tuple(payload.get("class_names", ())) != tuple(spec["class_names"]):
        raise ValueError("checkpoint class order differs")
    observed_model_config, checkpoint_seed = _validate_checkpoint_binding(
        payload,
        dataset=args.dataset,
        requested_seed=args.seed,
        effective_sha256=effective_sha256,
        expected_model_config=expected_model_config,
        source_sha256=source_sha256,
        modalities=modalities,
    )

    receipt_path = pred_path = None
    if args.output is not None:
        output = resolve_under_repo(args.output)
        receipt_path = output if output.suffix == ".json" else output / "eval_receipt.json"
        pred_path = output.with_name(output.stem + "_predictions.npz") if output.suffix == ".json" else output / "predictions.npz"
        if (receipt_path.exists() or pred_path.exists()) and not args.overwrite:
            raise FileExistsError(f"evaluation output exists: {receipt_path.parent}")
    artifact = D.load_artifact(
        bundle / f"{args.split}.rows.npz",
        split=args.split,
        rows=int(spec[f"{args.split}_rows"]),
        class_names=spec["class_names"],
    )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device(args.device)
    config = _config_instance(observed_model_config)
    model = SAGERModel(config)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    D.set_seed(checkpoint_seed)
    metrics, logits, predictions = D.evaluate(model, artifact, int(raw["training"]["batch_dialogues"]), device)
    record = {
        "schema_version": "sager_eval_receipt_v3",
        "status": "PASS",
        "model_id": "SAGER",
        "dataset_id": args.dataset,
        "seed": checkpoint_seed,
        "split": args.split,
        "modalities": modalities,
        "effective_config_sha256": effective_sha256,
        "model_config_sha256": model_config_sha256,
        "source_sha256": source_sha256,
        "config": {"path": str(args.config), "source_sha256": file_sha256(args.config)},
        "checkpoint": {"path": str(checkpoint_path), "sha256": file_sha256(checkpoint_path)},
        "artifact_sha256": artifact.sha256,
        "rows": artifact.rows,
        "metrics": metrics,
    }
    print(json.dumps({"status": "PASS", "metrics": metrics}, sort_keys=True), flush=True)
    if receipt_path is not None and pred_path is not None:
        _atomic_npz(pred_path, utterance_id=artifact.utterance_id, labels=artifact.labels, predictions=predictions, logits=logits)
        record["predictions"] = {"path": str(pred_path), "sha256": file_sha256(pred_path), "rows": artifact.rows}
        record["payload_sha256"] = canonical_sha256(record)
        _atomic_json(receipt_path, record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
