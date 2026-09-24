# Data preparation

## Required inputs

SAGER trains on frozen text/audio feature bundles. The bundles and upstream
RoBERTa/data2vec feature-extraction pipeline are not included in this package.

Raw IEMOCAP and MELD data are not included. Obtain them from their original
providers under the applicable access and distribution terms.
`utils/dataloader.py` reads prepared NPZ files; it does not process raw recordings.

```text
features/
├── iemocap_legacy7433/
│   ├── train.rows.npz
│   ├── dev.rows.npz
│   └── test.rows.npz
└── meld_official/
    ├── train.rows.npz
    ├── dev.rows.npz
    └── test.rows.npz
```

## Feature schema

`N` denotes the number of utterances in a split, and `C` the number of classes.
Use numeric or Unicode NumPy arrays; the loader does not allow pickled objects.

| Field | Shape and meaning |
|---|---|
| `class_names` | `[C]`, ordered class names |
| `utterance_id`, `dialogue_id` | `[N]`, identifiers |
| `labels` | `[N]`, integer class indices |
| `speaker_index` | `[N]`, speakers numbered by first occurrence, starting at zero within each dialogue |
| `temporal_position` | `[N]`, contiguous positions `0..length-1` within each dialogue |
| `text_cls` | `[N,1024]`, one frozen CLS vector per utterance |
| `text_anchor_logits` | `[N,C]`, frozen text predictions |
| `audio_tokens` | `[N,27,768]`, frozen audio features |
| `audio_quality` | `[N,6]` |
| `audio_available` | `[N]`, boolean availability |
| `router_utility_targets` | `[N]`, audio utility in `[-1,1]` |
| `router_utility_mask` | `[N]`, boolean/binary target validity |
| `schema_version` | Optional scalar: `sager_rows_v2` |
| `split` | Optional scalar: `train`, `dev`, or `test` |

Each dialogue must occupy contiguous rows. Class order, utterance order, split
membership, and feature/anchor alignment must be consistent. Class names and
expected row counts are listed in the dataset YAML files under `configs/`.

Both utility arrays are required for training; valid target entries must exist
on rows with available audio. Development and test files may omit both arrays.
The model's forward pass does not receive labels or utility targets. Each utterance has one audio utility prediction and, during training, one
corresponding supervision target.

## OOF audio utility targets

Prepare `predictors.npz` with `utterance_id`, `text_features` of shape `[N,D_t]`,
and `audio_features` of shape `[N,D_a]`. The row order must match the training
bundle exactly. These predictor matrices must be supplied explicitly; this script
does not define their upstream construction or the audio-token reduction.

```bash
python preprocessing/utility_targets.py \
  --bundle /path/to/input/train.rows.npz \
  --predictors /path/to/predictors.npz \
  --ridge-alpha 1.0 \
  --output /path/to/prepared/train.rows.npz
```

The generator uses five dialogue-grouped folds. Within each fold, feature
standardization and one-hot ridge regression are fitted on the other training
folds, separately for text and concatenated text/audio predictors. It computes
held-out logits and the utility target described in [Method implementation](method.md).
Unavailable audio features are replaced with zeros and their utility loss is masked.

`--ridge-alpha` controls ridge regularization and defaults to 1.0. This is an
implementation default, not a hyperparameter specified in the manuscript.
The output contains targets, masks, fold assignments, and OOF logits. A JSON
sidecar records the predictor/input/output hashes and generator settings.
The command writes a new bundle and does not overwrite existing files.

Only training data are used for target generation. Do not combine development or
test examples with training folds. Upstream feature/anchor training and extraction
must also be documented separately; downstream OOF target generation alone does
not establish how the upstream predictions were obtained.

## Evaluation inputs

Evaluation requires a checkpoint produced with matching code, configuration, and
feature inputs. No pretrained checkpoints are supplied. Runtime records include
observed input and source hashes for identifying the evaluated experiment.
