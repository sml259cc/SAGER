#!/usr/bin/env python3
"""Load dialogue batches of frozen text and audio features."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from model.backbone import SAGERInputs
from utils.common import file_sha256


@dataclass(frozen=True)
class Dialogue:
    identifier: str
    start: int
    stop: int

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class Artifact:
    path: Path
    sha256: str
    split: str
    class_names: tuple[str, ...]
    utterance_id: np.ndarray
    dialogue_id: np.ndarray
    labels: np.ndarray
    speaker_index: np.ndarray
    temporal_position: np.ndarray
    text_cls: np.ndarray
    text_anchor_logits: np.ndarray
    audio_tokens: np.ndarray
    audio_quality: np.ndarray
    audio_available: np.ndarray
    router_utility_targets: np.ndarray
    router_utility_mask: np.ndarray
    dialogues: tuple[Dialogue, ...]

    @property
    def rows(self) -> int:
        return len(self.labels)


def dialogue_slices(dialogue_id: np.ndarray, temporal: np.ndarray, speakers: np.ndarray) -> tuple[Dialogue, ...]:
    boundaries = [0] + [i for i in range(1, len(dialogue_id)) if dialogue_id[i] != dialogue_id[i - 1]] + [len(dialogue_id)]
    out: list[Dialogue] = []
    seen: set[str] = set()
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        name = str(dialogue_id[start])
        if name in seen:
            raise ValueError(f"noncontiguous dialogue: {name}")
        seen.add(name)
        if not np.array_equal(temporal[start:stop], np.arange(stop - start)):
            raise ValueError(f"bad temporal positions: {name}")
        encountered: set[int] = set()
        for raw in speakers[start:stop].tolist():
            value = int(raw)
            if value not in encountered:
                if value != len(encountered):
                    raise ValueError(f"bad speaker indices: {name}")
                encountered.add(value)
        out.append(Dialogue(name, start, stop))
    return tuple(out)


def load_artifact(
    path: str | Path,
    *,
    split: str,
    rows: int | None = None,
    class_names: Sequence[str] | None = None,
    expected_sha: str | None = None,
) -> Artifact:
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"artifact must be a direct file: {source}")
    observed = file_sha256(source)
    if expected_sha is not None and observed != expected_sha:
        raise ValueError(f"{split} artifact hash differs")
    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    if "schema_version" in arrays:
        schema = str(np.asarray(arrays["schema_version"]).item())
        if schema != "sager_rows_v2":
            raise ValueError(f"{split} schema differs: {schema}")
    if "split" in arrays and str(np.asarray(arrays["split"]).item()) != split:
        raise ValueError(f"{split} identity differs")
    names = tuple(np.asarray(arrays["class_names"]).astype(str).tolist())
    if class_names is not None and names != tuple(class_names):
        raise ValueError(f"{split} class order differs")
    n = int(len(arrays["labels"]))
    if rows is not None and n != rows:
        raise ValueError(f"{split} row count differs")
    c = len(names)
    expected = {
        "text_cls": (n, 1024),
        "text_anchor_logits": (n, c),
        "audio_tokens": (n, 27, 768),
        "audio_quality": (n, 6),
        "audio_available": (n,),
    }
    for key, shape in expected.items():
        if arrays[key].shape != shape:
            raise ValueError(f"{split}.{key} shape differs: {arrays[key].shape}")
    if arrays["audio_available"].dtype != np.bool_:
        arrays["audio_available"] = arrays["audio_available"].astype(bool)
    utility_keys = ("router_utility_targets", "router_utility_mask")
    if any(key in arrays for key in utility_keys) and not all(key in arrays for key in utility_keys):
        raise ValueError(f"{split} requires both utility targets and their mask")
    if "router_utility_targets" not in arrays:
        if split == "train":
            raise ValueError("Training requires OOF utility targets; see docs/data.md")
        arrays["router_utility_targets"] = np.zeros(n, np.float32)
        arrays["router_utility_mask"] = np.zeros(n, bool)
    for key in utility_keys:
        if arrays[key].shape != (n,):
            raise ValueError(f"{split}.{key} must have shape [N]")
    utility_mask = arrays["router_utility_mask"]
    if not np.isin(utility_mask, [0, 1]).all():
        raise ValueError(f"{split} audio utility mask must be boolean or binary")
    utility_mask = utility_mask.astype(bool)
    utility_mask &= arrays["audio_available"]
    arrays["router_utility_mask"] = utility_mask
    active_targets = arrays["router_utility_targets"][utility_mask]
    if not np.isfinite(active_targets).all() or (np.abs(active_targets) > 1).any():
        raise ValueError(f"{split} active audio utility targets must be finite and within [-1, 1]")
    if split == "train" and not utility_mask.any():
        raise ValueError("Training requires valid OOF audio utility targets on available audio rows")
    dialogues = dialogue_slices(
        arrays["dialogue_id"].astype(str),
        arrays["temporal_position"],
        arrays["speaker_index"],
    )
    return Artifact(
        path=source,
        sha256=observed,
        split=split,
        class_names=names,
        utterance_id=arrays["utterance_id"],
        dialogue_id=arrays["dialogue_id"],
        labels=arrays["labels"].astype(np.int64),
        speaker_index=arrays["speaker_index"].astype(np.int64),
        temporal_position=arrays["temporal_position"].astype(np.int64),
        text_cls=np.asarray(arrays["text_cls"], np.float32),
        text_anchor_logits=np.asarray(arrays["text_anchor_logits"], np.float32),
        audio_tokens=np.asarray(arrays["audio_tokens"], np.float32),
        audio_quality=np.asarray(arrays["audio_quality"], np.float32),
        audio_available=np.asarray(arrays["audio_available"], bool),
        router_utility_targets=np.asarray(arrays["router_utility_targets"], np.float32),
        router_utility_mask=np.asarray(arrays["router_utility_mask"], bool),
        dialogues=dialogues,
    )


def load_bundle(bundle: str | Path, spec: Mapping[str, Any]) -> tuple[Artifact, Artifact, dict[str, Any]]:
    root = Path(bundle).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"bundle must be a directory: {root}")
    names = tuple(spec["class_names"])
    train = load_artifact(root / "train.rows.npz", split="train", rows=int(spec["train_rows"]), class_names=names)
    dev = load_artifact(root / "dev.rows.npz", split="dev", rows=int(spec["dev_rows"]), class_names=names)
    return train, dev, {"path": str(root), "train_sha256": train.sha256, "dev_sha256": dev.sha256}


def set_seed(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(4)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def groups(artifact: Artifact, *, batch_dialogues: int, shuffle: bool, seed: int, epoch: int) -> Iterable[tuple[Dialogue, ...]]:
    order = np.arange(len(artifact.dialogues))
    if shuffle:
        np.random.default_rng(np.random.SeedSequence([seed, epoch, 0x5A6E])).shuffle(order)
    for i in range(0, len(order), batch_dialogues):
        yield tuple(artifact.dialogues[int(j)] for j in order[i : i + batch_dialogues])


@dataclass(frozen=True)
class Batch:
    inputs: SAGERInputs
    targets: torch.Tensor
    utility_targets: torch.Tensor
    utility_mask: torch.Tensor
    row_index: np.ndarray


def collate(artifact: Artifact, dialogues: Sequence[Dialogue], device: torch.device) -> Batch:
    batch_size, turns, classes = len(dialogues), max(item.length for item in dialogues), len(artifact.class_names)
    row_index = np.full((batch_size, turns), -1, np.int64)
    mask = np.zeros((batch_size, turns), bool)
    audio_available = np.zeros((batch_size, turns), bool)
    speaker = np.full((batch_size, turns), -1, np.int64)
    temporal = np.zeros((batch_size, turns), np.int64)
    labels = np.full((batch_size, turns), -1, np.int64)
    text = np.zeros((batch_size, turns, 1024), np.float32)
    anchor = np.zeros((batch_size, turns, classes), np.float32)
    audio = np.zeros((batch_size, turns, 27, 768), np.float32)
    audio_quality = np.zeros((batch_size, turns, 6), np.float32)
    utility = np.zeros((batch_size, turns), np.float32)
    utility_mask = np.zeros((batch_size, turns), bool)
    for batch_index, dialogue in enumerate(dialogues):
        rows = np.arange(dialogue.start, dialogue.stop)
        length = dialogue.length
        target = np.s_[batch_index, :length]
        row_index[target] = rows
        mask[target] = True
        audio_available[target] = artifact.audio_available[rows]
        speaker[target] = artifact.speaker_index[rows]
        temporal[target] = artifact.temporal_position[rows]
        labels[target] = artifact.labels[rows]
        text[target] = artifact.text_cls[rows]
        anchor[target] = artifact.text_anchor_logits[rows]
        audio[target] = artifact.audio_tokens[rows]
        audio_quality[target] = artifact.audio_quality[rows]
        utility[target] = artifact.router_utility_targets[rows]
        utility_mask[target] = artifact.router_utility_mask[rows]
    mask_t = torch.as_tensor(mask, device=device)
    inputs = SAGERInputs(
        text_cls=torch.as_tensor(text, device=device),
        text_anchor_logits=torch.as_tensor(anchor, device=device),
        audio_tokens=torch.as_tensor(audio, device=device),
        audio_quality=torch.as_tensor(audio_quality, device=device),
        audio_available=torch.as_tensor(audio_available, device=device),
        utterance_mask=mask_t,
        speaker_index=torch.as_tensor(speaker, device=device),
        temporal_position=torch.as_tensor(temporal, device=device),
    )
    return Batch(
        inputs,
        torch.as_tensor(labels, device=device),
        torch.as_tensor(utility, device=device),
        torch.as_tensor(utility_mask, device=device),
        row_index,
    )


def metrics(labels: np.ndarray, predictions: np.ndarray, classes: int) -> dict[str, float]:
    y, pred = np.asarray(labels, np.int64), np.asarray(predictions, np.int64)
    per: list[float] = []
    support: list[int] = []
    for cls in range(classes):
        tp = int(np.sum((y == cls) & (pred == cls)))
        fp = int(np.sum((y != cls) & (pred == cls)))
        fn = int(np.sum((y == cls) & (pred != cls)))
        denom = 2 * tp + fp + fn
        per.append(0.0 if denom == 0 else 2.0 * tp / denom)
        support.append(int(np.sum(y == cls)))
    return {
        "accuracy": float(np.mean(y == pred)),
        "weighted_f1": float(np.dot(per, support) / len(y)),
        "macro_f1": float(np.mean(per)),
    }


@torch.no_grad()
def evaluate(model: torch.nn.Module, artifact: Artifact, batch_dialogues: int, device: torch.device) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    logits = np.zeros((artifact.rows, len(artifact.class_names)), np.float32)
    filled = np.zeros(artifact.rows, np.int64)
    for dialogues in groups(artifact, batch_dialogues=batch_dialogues, shuffle=False, seed=0, epoch=0):
        batch = collate(artifact, dialogues, device)
        output = model(batch.inputs)
        valid = batch.inputs.utterance_mask
        rows = batch.row_index[valid.cpu().numpy()]
        logits[rows] = output.logits[valid].detach().cpu().numpy()
        filled[rows] += 1
    if not np.all(filled == 1) or not np.isfinite(logits).all():
        raise RuntimeError("evaluation coverage/finite guard failed")
    predictions = logits.argmax(axis=1).astype(np.int64)
    return metrics(artifact.labels, predictions, len(artifact.class_names)), logits, predictions
