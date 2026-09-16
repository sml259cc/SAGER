#!/usr/bin/env python3
"""Train SAGER on a text+audio feature bundle and select the best dev checkpoint."""
from __future__ import annotations

import argparse
import dataclasses
import fcntl
import json
import os
import random
import socket
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from model.sager import SAGERConfig, SAGERModel, compute_sager_objective
from model.module_strength import project
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


def _rel(path: Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(REPO))
    except ValueError:
        return str(resolved)


def _source_sha256() -> str:
    return source_tree_sha256(REPO, SOURCE_FILES)


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _atomic_history(path: Path, history: Sequence[Mapping[str, Any]]) -> None:
    text = "".join(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in history)
    _atomic_text(path, text)


def _acquire_run_lock(destination: Path, *, dataset: str, seed: int) -> Any:
    path = destination / "run.lock"
    stream = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        stream.close()
        raise RuntimeError(f"another process already owns this run: {destination}") from exc
    stream.seek(0)
    stream.truncate()
    stream.write(json.dumps({
        "dataset": dataset,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "seed": seed,
        "started_at_unix": time.time(),
    }, sort_keys=True) + "\n")
    stream.flush()
    os.fsync(stream.fileno())
    return stream


def _atomic_npz(path: Path, **arrays: Any) -> None:
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


def _torch_load(path: Path) -> Mapping[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _restore_rng(snapshot: Mapping[str, Any], device: torch.device) -> None:
    random.setstate(snapshot["python_rng_state"])
    np.random.set_state(snapshot["numpy_rng_state"])
    torch.set_rng_state(snapshot["torch_rng_state"])
    cuda_state = snapshot.get("cuda_rng_state_all")
    if device.type == "cuda" and cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)


def _safe_to_replace(destination: Path) -> bool:
    protected = {Path("/"), Path.home().resolve(), REPO.resolve(), REPO.parent.resolve()}
    return destination.resolve() not in protected and len(destination.resolve().parts) >= 4


def _normalized_model_config(values: Mapping[str, Any]) -> dict[str, Any]:
    prepared = dict(values)
    for key in ("temporal_kernels", "path_local_scales"):
        if key in prepared:
            prepared[key] = tuple(prepared[key])
    normalized = canonical_value(dataclasses.asdict(SAGERConfig(**prepared)))
    return normalized


def _assert_binding(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} binding differs")


def _validate_checkpoint_payload(payload: Mapping[str, Any], binding: Mapping[str, Any]) -> None:
    _assert_binding(payload.get("schema_version"), "sager_checkpoint_v3", "checkpoint schema")
    _assert_binding(payload.get("model_id"), "SAGER", "checkpoint model")
    _assert_binding(payload.get("dataset_id"), binding["dataset_id"], "checkpoint dataset")
    _assert_binding(payload.get("seed"), binding["seed"], "checkpoint seed")

    _assert_binding(
        payload.get("effective_config_sha256"),
        binding["effective_config_sha256"],
        "checkpoint effective config",
    )
    _assert_binding(payload.get("source_sha256"), binding["source_sha256"], "checkpoint source")
    try:
        observed_model_config = _normalized_model_config(payload["model_config"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint model config is invalid") from exc
    _assert_binding(
        canonical_sha256(observed_model_config),
        binding["model_config_sha256"],
        "checkpoint model config",
    )
    _assert_binding(payload.get("model_config_sha256"), binding["model_config_sha256"], "checkpoint model config hash")
    _assert_binding(payload.get("modalities"), binding["modalities"], "checkpoint modalities")
    _assert_binding(payload.get("class_names"), binding["class_names"], "checkpoint class order")
    if not isinstance(payload.get("state_dict"), Mapping):
        raise ValueError("checkpoint state_dict is missing")


def _validate_completed_run(receipt_path: Path, checkpoint_path: Path, binding: Mapping[str, Any]) -> None:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    recorded_payload_sha256 = receipt.get("payload_sha256")
    unsigned_receipt = {key: value for key, value in receipt.items() if key != "payload_sha256"}
    _assert_binding(recorded_payload_sha256, canonical_sha256(unsigned_receipt), "training receipt payload")
    _assert_binding(receipt.get("schema_version"), "sager_training_receipt_v3", "training receipt schema")
    _assert_binding(receipt.get("status"), "PASS", "training status")
    _assert_binding(receipt.get("dataset_id"), binding["dataset_id"], "training dataset")
    _assert_binding(receipt.get("seed"), binding["seed"], "training seed")

    _assert_binding(receipt.get("modalities"), binding["modalities"], "training modalities")
    config_record = receipt.get("config") or {}
    _assert_binding(
        config_record.get("effective_sha256"),
        binding["effective_config_sha256"],
        "training effective config",
    )
    model_record = receipt.get("model_config") or {}
    _assert_binding(model_record.get("sha256"), binding["model_config_sha256"], "training model config")
    _assert_binding(receipt.get("source_sha256"), binding["source_sha256"], "training source")
    if not checkpoint_path.is_file():
        raise RuntimeError("training receipt exists but checkpoint is missing")
    checkpoint_sha256 = file_sha256(checkpoint_path)
    _assert_binding(
        (receipt.get("checkpoint") or {}).get("sha256"),
        checkpoint_sha256,
        "training checkpoint hash",
    )
    _validate_checkpoint_payload(_torch_load(checkpoint_path), binding)


def _run_auto_test(args: argparse.Namespace, destination: Path, binding: Mapping[str, Any]) -> int:
    checkpoint = destination / "checkpoint.pt"
    output = destination / "test"
    receipt_path = output / "eval_receipt.json"
    if receipt_path.is_file():
        existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        recorded_payload_sha256 = existing.get("payload_sha256")
        unsigned_receipt = {key: value for key, value in existing.items() if key != "payload_sha256"}
        _assert_binding(recorded_payload_sha256, canonical_sha256(unsigned_receipt), "test receipt payload")
        expected = {
            "status": "PASS",
            "dataset_id": binding["dataset_id"],
            "seed": binding["seed"],
            "effective_config_sha256": binding["effective_config_sha256"],
            "model_config_sha256": binding["model_config_sha256"],
            "source_sha256": binding["source_sha256"],
            "modalities": binding["modalities"],
        }
        for key, value in expected.items():
            _assert_binding(existing.get(key), value, f"test {key}")
        _assert_binding(
            (existing.get("checkpoint") or {}).get("sha256"),
            file_sha256(checkpoint),
            "test checkpoint hash",
        )
        print(json.dumps({"status": "SKIP", "reason": "test already complete", "receipt": _rel(receipt_path)}, sort_keys=True), flush=True)
        return 0
    from evaluate import main as evaluate_main

    evaluate_args = [
        "--dataset", args.dataset,
        "--split", "test",
        "--checkpoint", str(checkpoint),
        "--seed", str(args.seed),
        "--config", str(args.config),
        "--output", str(output),
        "--device", args.device,
    ]
    if args.bundle is not None:
        evaluate_args.extend(("--bundle", str(args.bundle)))
    return evaluate_main(evaluate_args)


def model_values(raw: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    valid = {item.name for item in dataclasses.fields(SAGERConfig)}
    unknown = sorted(set(raw["model"]) - valid)
    if unknown:
        raise ValueError(f"unknown model config keys: {unknown}")
    values = {key: value for key, value in raw["model"].items() if key in valid}
    for key in ("temporal_kernels", "path_local_scales"):
        if key in values:
            values[key] = tuple(values[key])
    values["num_classes"] = len(spec["class_names"])
    return values


def training_diagnostics(output: Any) -> dict[str, tuple[float, int]]:
    mask = output.utterance_mask
    active_rows = int(mask.sum().detach().cpu())
    if active_rows <= 0:
        raise RuntimeError("diagnostics received no active utterances")
    values = {
        "correction_gate_actual_mean": output.correction_strength,
        "relational_verifier_actual_mean": output.relational_verifier_probability,
        "audio_utility_mean": output.predicted_audio_utility,
    }
    out: dict[str, tuple[float, int]] = {}
    for key, value in values.items():
        tensor = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=mask.device)
        if tensor.ndim == 0:
            total = float(tensor.cpu()) * active_rows
            count = active_rows
        elif tuple(tensor.shape) == tuple(mask.shape):
            selected = tensor[mask]
            total = float(selected.to(torch.float64).sum().cpu())
            count = int(selected.numel())
        else:
            raise RuntimeError(f"diagnostic {key} shape does not align with utterance mask")
        if count <= 0 or not np.isfinite(total):
            raise RuntimeError(f"nonfinite diagnostic {key}")
        out[key] = (total, count)
    return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("iemocap_legacy7433", "meld_official"))
    parser.add_argument("--seed", type=int, default=42, help="Nonnegative experiment seed")
    parser.add_argument("--bundle", type=Path, default=None, help="Feature directory; default features/<dataset>")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Run directory; default runs/main/<dataset>/seed_<seed>",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the 30-epoch training budget",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from the last atomically saved epoch")
    parser.add_argument("--auto-test", action="store_true", help="Run the test split immediately after training completes")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.config is None:
        args.config = REPO / "configs" / (args.dataset + ".yaml")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    args.config = resolve_config_path(args.config, REPO)
    raw = load_experiment_config(args.config, REPO)
    spec = raw["datasets"][args.dataset]
    values = model_values(raw, spec)
    normalized_model_config = _normalized_model_config(values)
    model_config_sha256 = canonical_sha256(normalized_model_config)
    effective_sha256 = effective_config_sha256(raw)
    source_sha256 = _source_sha256()

    modalities = ["text", "audio"]
    binding = {
        "dataset_id": args.dataset,
        "seed": args.seed,
        "effective_config_sha256": effective_sha256,
        "model_config_sha256": model_config_sha256,
        "source_sha256": source_sha256,
        "modalities": modalities,
        "class_names": list(spec["class_names"]),
    }
    bundle = (args.bundle or (REPO / "features" / args.dataset)).expanduser()
    if not bundle.is_absolute():
        bundle = (REPO / bundle).resolve() if not bundle.exists() else bundle.resolve()
    else:
        bundle = bundle.resolve()
    default_output = REPO / "runs" / "main" / args.dataset / f"seed_{args.seed}"
    destination = (args.output or default_output).expanduser()
    if not destination.is_absolute():
        destination = (REPO / destination).resolve()
    else:
        destination = destination.resolve()
    # Validate required inputs before creating or replacing a run directory.
    train, dev, bundle_binding = D.load_bundle(bundle, spec)
    if destination.exists() and args.overwrite:
        if not _safe_to_replace(destination):
            raise ValueError(f"refusing to recursively replace unsafe output path: {destination}")
        import shutil

        shutil.rmtree(destination)
    elif destination.exists() and not args.resume:
        raise FileExistsError(f"{destination}; use --resume for an interrupted run or choose a new output")
    destination.mkdir(parents=True, exist_ok=True)
    run_lock = _acquire_run_lock(
        destination,
        dataset=args.dataset,
        seed=args.seed,
    )

    completed_receipt = destination / "run_receipt.json"
    if completed_receipt.is_file():
        _validate_completed_run(completed_receipt, destination / "checkpoint.pt", binding)
        print(json.dumps({"status": "SKIP", "reason": "training already complete", "output": _rel(destination)}, sort_keys=True), flush=True)
        return _run_auto_test(args, destination, binding) if args.auto_test else 0

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device(args.device)
    D.set_seed(args.seed)
    model = SAGERModel(SAGERConfig(**values)).to(device)
    import shutil
    shutil.copy2(args.config, destination / "effective_config.yaml")
    training = raw["training"]
    epochs = int(training["joint_epochs"] if args.epochs is None else args.epochs)
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    # The manuscript specifies one AdamW learning rate and weight decay.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    history: list[dict[str, Any]] = []
    best_state = best_metrics = best_logits = best_predictions = None
    selected = 0
    first_epoch = 1
    elapsed_before = 0.0
    resume_used = False
    resume_path = destination / "resume_state.pt"
    started = time.time()
    if resume_path.is_file():
        if not args.resume:
            raise RuntimeError("resume state exists but --resume was not specified")
        snapshot = _torch_load(resume_path)
        expected = {
            "schema_version": "sager_resume_v3",
            "model_id": "SAGER",
            "dataset_id": args.dataset,
            "seed": args.seed,
            "epochs": epochs,
            "effective_config_sha256": effective_sha256,
            "model_config_sha256": model_config_sha256,
            "source_sha256": source_sha256,
            "train_sha256": train.sha256,
            "dev_sha256": dev.sha256,
        }
        for key, value in expected.items():
            if snapshot.get(key) != value:
                raise ValueError(f"resume binding differs: {key}")
        try:
            snapshot_model_config_sha256 = canonical_sha256(_normalized_model_config(snapshot["model_config"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("resume model config is invalid") from exc
        _assert_binding(snapshot_model_config_sha256, model_config_sha256, "resume model config")
        last_completed = int(snapshot["last_completed_epoch"])
        history = list(snapshot["history"])
        if last_completed < 1 or last_completed > epochs or len(history) != last_completed:
            raise ValueError("resume epoch/history is inconsistent")
        model.load_state_dict(snapshot["model_state"], strict=True)
        optimizer.load_state_dict(snapshot["optimizer_state"])
        _optimizer_to_device(optimizer, device)
        best_state = snapshot["best_state"]
        best_metrics = snapshot["best_metrics"]
        best_logits = snapshot["best_logits"]
        best_predictions = snapshot["best_predictions"]
        selected = int(snapshot["selected_epoch"])
        elapsed_before = float(snapshot.get("elapsed_seconds", 0.0))
        first_epoch = last_completed + 1
        _restore_rng(snapshot, device)
        resume_used = True
        _atomic_history(destination / "history.jsonl", history)
        print(json.dumps({
            "dataset": args.dataset,
            "event": "RESUME_SESSION",
            "last_completed_epoch": last_completed,
            "next_epoch": first_epoch,
            "seed": args.seed,
            "status": "RESUME",
            "timestamp_unix": time.time(),
        }, sort_keys=True), flush=True)

    for epoch in range(first_epoch, epochs + 1):
        epoch_started = time.time()
        print(json.dumps({
            "dataset": args.dataset,
            "epoch": epoch,
            "event": "EPOCH_START",
            "seed": args.seed,
            "timestamp_unix": epoch_started,
        }, sort_keys=True), flush=True)
        model.train()
        losses: list[float] = []
        diagnostics: list[dict[str, tuple[float, int]]] = []
        rows_seen = 0
        visits = np.zeros(train.rows, np.int64)
        for dialogues in (
            D.groups(train, batch_dialogues=int(training["batch_dialogues"]), shuffle=True, seed=args.seed, epoch=epoch)
        ):
            batch = D.collate(train, dialogues, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch.inputs)
            parts = compute_sager_objective(output, batch.targets, batch.utility_targets, batch.utility_mask, raw["objective"])
            loss = parts["total"]
            if loss.ndim or not bool(torch.isfinite(loss).item()):
                raise RuntimeError("nonfinite training objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]), error_if_nonfinite=True)
            optimizer.step()
            project(model)
            losses.append(float(loss.detach().cpu()))
            diagnostics.append(training_diagnostics(output))
            rows_seen += sum(item.length for item in dialogues)
            visits[batch.row_index[batch.row_index >= 0]] += 1
        if rows_seen != train.rows or not np.all(visits == 1):
            raise RuntimeError("epoch coverage differs")
        print(json.dumps({
            "dataset": args.dataset,
            "epoch": epoch,
            "event": "DEV_START",
            "seed": args.seed,
            "timestamp_unix": time.time(),
            "train_rows_seen": rows_seen,
        }, sort_keys=True), flush=True)
        metrics, logits, predictions = D.evaluate(model, dev, int(training["batch_dialogues"]), device)
        diagnostic_counts = {key: sum(row[key][1] for row in diagnostics) for key in diagnostics[0]}
        if len(set(diagnostic_counts.values())) != 1 or next(iter(diagnostic_counts.values())) != rows_seen:
            raise RuntimeError("diagnostic utterance coverage differs")
        averaged = {
            key: float(sum(row[key][0] for row in diagnostics) / diagnostic_counts[key])
            for key in diagnostics[0]
        }
        mean_optimizer_loss = float(np.mean(losses))
        row = {
            "epoch": epoch,
            "learning_rate": float(training["learning_rate"]),
            "module_strengths": model.module_strengths.values(),
            "mean_optimizer_loss": mean_optimizer_loss,
            "train_rows_seen": rows_seen,
            "dev": metrics,
            "diagnostics": averaged,
        }
        key = (metrics["weighted_f1"], metrics["accuracy"], -epoch)
        old = None if best_metrics is None else (best_metrics["weighted_f1"], best_metrics["accuracy"], -selected)
        is_new_best = old is None or key > old
        if is_new_best:
            best_state = _cpu_state_dict(model)
            best_metrics = dict(metrics)
            best_logits = logits.copy()
            best_predictions = predictions.copy()
            selected = epoch
        completed_at = time.time()
        elapsed_seconds = elapsed_before + completed_at - started
        row.update({
            "completed_at_unix": completed_at,
            "dataset_id": args.dataset,
            "elapsed_seconds": elapsed_seconds,
            "epoch_duration_seconds": completed_at - epoch_started,
            "schema_version": "sager_epoch_history_v3",
            "seed": args.seed,
            "selection": {
                "best_dev": best_metrics,
                "is_new_best": is_new_best,
                "selected_epoch": selected,
            },
        })
        history.append(row)

        snapshot = {
            "schema_version": "sager_resume_v3",
            "model_id": "SAGER",
            "dataset_id": args.dataset,
            "seed": args.seed,
            "epochs": epochs,
            "last_completed_epoch": epoch,
            "next_epoch": epoch + 1,
            "effective_config_sha256": effective_sha256,
            "model_config_sha256": model_config_sha256,
            "source_sha256": source_sha256,
            "train_sha256": train.sha256,
            "dev_sha256": dev.sha256,
            "model_config": normalized_model_config,
            "model_state": _cpu_state_dict(model),
            "optimizer_state": optimizer.state_dict(),
            "best_state": best_state,
            "best_metrics": best_metrics,
            "best_logits": best_logits,
            "best_predictions": best_predictions,
            "selected_epoch": selected,
            "history": history,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "elapsed_seconds": elapsed_seconds,
        }
        _atomic_torch_save(resume_path, snapshot)
        _atomic_history(destination / "history.jsonl", history)
        print(json.dumps(row, sort_keys=True), flush=True)

    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state, strict=True)
    replay_metrics, replay_logits, replay_predictions = D.evaluate(model, dev, int(training["batch_dialogues"]), device)
    assert np.array_equal(replay_logits, best_logits), "selected dev replay differs"
    assert replay_metrics == best_metrics
    assert _source_sha256() == source_sha256, "source changed during run"
    checkpoint = {
        "schema_version": "sager_checkpoint_v3",
        "model_id": "SAGER",
        "dataset_id": args.dataset,
        "seed": args.seed,
        "effective_config_sha256": effective_sha256,
        "model_config_sha256": model_config_sha256,
        "source_sha256": source_sha256,
        "modalities": modalities,
        "selected_epoch": selected,
        "dev_metrics": best_metrics,
        "class_names": list(spec["class_names"]),
        "model_config": normalized_model_config,
        "train_sha256": train.sha256,
        "dev_sha256": dev.sha256,
        "state_dict": best_state,
        "module_strengths": model.module_strengths.values(),
    }
    checkpoint_path = destination / "checkpoint.pt"
    _atomic_torch_save(checkpoint_path, checkpoint)
    _atomic_npz(
        destination / "dev_predictions.npz",
        utterance_id=dev.utterance_id,
        labels=dev.labels,
        predictions=best_predictions,
        logits=best_logits,
        selected_epoch=np.asarray(selected),
    )
    _atomic_history(destination / "history.jsonl", history)
    _atomic_json(destination / "dev_metrics.json", {**best_metrics, "selected_epoch": selected})
    receipt = {
        "schema_version": "sager_training_receipt_v3",
        "status": "PASS",
        "model_id": "SAGER",
        "dataset_id": args.dataset,
        "seed": args.seed,
        "modalities": modalities,
        "bundle": bundle_binding,
        "config": {
            "path": _rel(Path(args.config)),
            "source_sha256": file_sha256(args.config),
            "effective_sha256": effective_sha256,
        },
        "model_config": {"sha256": model_config_sha256, "values": normalized_model_config},
        "train": {
            "rows": train.rows,
            "dialogues": len(train.dialogues),
            "sha256": train.sha256,
            "audio_observed_rows": int(np.asarray(train.audio_available, dtype=bool).sum()),
            "audio_missing_rows": int((~np.asarray(train.audio_available, dtype=bool)).sum()),
        },
        "dev": {
            "rows": dev.rows,
            "dialogues": len(dev.dialogues),
            "sha256": dev.sha256,
            "metrics": best_metrics,
            "audio_observed_rows": int(np.asarray(dev.audio_available, dtype=bool).sum()),
            "audio_missing_rows": int((~np.asarray(dev.audio_available, dtype=bool)).sum()),
        },
        "checkpoint": {"path": "checkpoint.pt", "sha256": file_sha256(checkpoint_path), "selected_epoch": selected},
        "history": {"path": "history.jsonl", "epochs": len(history), "sha256": file_sha256(destination / "history.jsonl")},
        "resume": {
            "supported": True,
            "used": resume_used,
            "last_completed_epoch": epochs,
            "snapshot_path": "resume_state.pt",
            "snapshot_sha256": file_sha256(resume_path),
        },
        "source_sha256": source_sha256,
        "duration_seconds": elapsed_before + time.time() - started,
        "device": str(device),
        "torch_version": torch.__version__,
        "module_strengths": model.module_strengths.values(),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "dev_replay_bitwise": True,
    }
    receipt["payload_sha256"] = canonical_sha256(receipt)
    _atomic_json(destination / "run_receipt.json", receipt)
    print(json.dumps({"status": "PASS", "output": _rel(destination), "selected_epoch": selected, "dev": best_metrics}, sort_keys=True), flush=True)
    if not args.auto_test:
        return 0
    del optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return _run_auto_test(args, destination, binding)


if __name__ == "__main__":
    raise SystemExit(main())
