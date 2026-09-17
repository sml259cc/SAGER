#!/usr/bin/env python3
"""Generate five-fold dialogue-grouped OOF audio utility targets.

The paper does not specify ridge alpha or how audio tokens become ridge inputs.
Supply explicit 2D predictor matrices and record alpha in the output metadata.
This generator does not train or extract RoBERTa/data2vec representations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def utility_from_logits(text_logits, text_audio_logits, labels):
    """The manuscript's clipped loss-reduction plus correctness-change target."""
    text = np.asarray(text_logits, dtype=np.float64)
    joint = np.asarray(text_audio_logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if text.ndim != 2 or joint.shape != text.shape or labels.shape != (len(text),):
        raise ValueError("logits/labels do not align")
    if not np.isfinite(text).all() or not np.isfinite(joint).all():
        raise ValueError("ridge logits must be finite")
    if ((labels < 0) | (labels >= text.shape[1])).any():
        raise ValueError("label index out of range")

    def nll(logits):
        shifted = logits - logits.max(axis=1, keepdims=True)
        return np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(labels)), labels]

    improvement = np.clip(nll(text) - nll(joint), -2, 2)
    correctness = (joint.argmax(1) == labels).astype(float) - (text.argmax(1) == labels).astype(float)
    return np.clip(0.25 * improvement + 0.5 * correctness, -1, 1).astype(np.float32)


def cross_fit(text_features, audio_features, labels, dialogue_id, audio_available,
              *, num_classes, alpha=1.0):
    text = np.asarray(text_features, dtype=np.float64)
    audio = np.asarray(audio_features, dtype=np.float64)
    labels = np.asarray(labels)
    groups = np.asarray(dialogue_id).astype(str)
    available = np.asarray(audio_available, dtype=bool)
    n = len(labels)
    if text.ndim != 2 or audio.ndim != 2 or text.shape[0] != n or audio.shape[0] != n:
        raise ValueError("Provide aligned 2D text_features/audio_features")
    if groups.shape != (n,) or available.shape != (n,) or labels.shape != (n,):
        raise ValueError("row metadata does not align")
    if not np.issubdtype(labels.dtype, np.integer) or ((labels < 0) | (labels >= num_classes)).any():
        raise ValueError("invalid class indices")
    if len(np.unique(groups)) < 5:
        raise ValueError("Five dialogue-grouped folds require at least five training dialogues")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("ridge alpha must be positive and finite")
    if not np.isfinite(text).all() or not np.isfinite(audio[available]).all():
        raise ValueError("predictor features must be finite on available rows")
    audio = np.where(available[:, None], audio, 0.0)
    joint = np.concatenate((text, audio), axis=1)
    onehot = np.eye(num_classes)[labels]
    text_logits = np.empty((n, num_classes))
    joint_logits = np.empty_like(text_logits)
    folds = np.full(n, -1, dtype=np.int64)
    for fold, (fit, held) in enumerate(GroupKFold(n_splits=5).split(text, labels, groups)):
        for features, logits in ((text, text_logits), (joint, joint_logits)):
            scaler = StandardScaler().fit(features[fit])
            predictor = Ridge(alpha=alpha).fit(scaler.transform(features[fit]), onehot[fit])
            logits[held] = predictor.predict(scaler.transform(features[held]))
        folds[held] = fold
    utility = utility_from_logits(text_logits, joint_logits, labels)
    utility[~available] = 0.0
    return {"router_utility_targets": utility, "router_utility_mask": available,
            "text_oof_logits": text_logits, "text_audio_oof_logits": joint_logits,
            "oof_fold": folds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Original train.rows.npz")
    parser.add_argument("--predictors", type=Path, required=True,
                        help="NPZ with utterance_id, text_features [N,D_t], audio_features [N,D_a]")
    parser.add_argument("--ridge-alpha", type=float, default=1.0,
                        help="Explicit implementation choice; not specified in the manuscript")
    parser.add_argument("--output", type=Path, required=True, help="New training NPZ; cannot overwrite input")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".utility.json").exists():
        parser.error("Output already exists; choose a new path")
    with np.load(args.bundle, allow_pickle=False) as data:
        rows = {key: data[key] for key in data.files}
    split = str(rows.get("split", "train" if args.bundle.name == "train.rows.npz" else ""))
    if split != "train":
        parser.error("Utility targets must be generated from the training split only")
    with np.load(args.predictors, allow_pickle=False) as data:
        if not np.array_equal(data["utterance_id"].astype(str), rows["utterance_id"].astype(str)):
            parser.error("Predictor utterance order differs from the training bundle")
        result = cross_fit(data["text_features"], data["audio_features"], rows["labels"],
                           rows["dialogue_id"], rows["audio_available"],
                           num_classes=len(rows["class_names"]), alpha=args.ridge_alpha)
    rows.update(result)
    rows["split"] = np.asarray("train")
    rows["utility_target_protocol"] = np.asarray("sager_audio_utility_oof_v1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **rows)
    def sha(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    metadata = {"protocol": "sager_audio_utility_oof_v1", "folds": 5,
                "splitter": "GroupKFold", "group_by": "dialogue_id",
                "ridge_alpha": args.ridge_alpha, "ridge_alpha_status": "not_specified_in_manuscript",
                "predictor_layout": "explicit_caller_supplied_2d_matrices",
                "missing_audio": "zero_audio_features_and_mask_utility_loss",
                "bundle_sha256": sha(args.bundle), "predictors_sha256": sha(args.predictors),
                "output_sha256": sha(args.output)}
    args.output.with_suffix(".utility.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
