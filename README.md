# VL-JEPA — DeepFake Video Classifier

A Vision-Language Joint Embedding Predictive Architecture (VL-JEPA) adapted for **binary video classification**: predicting whether a video is **FAKE** or **REAL**.

## Task

Given a video clip and the fixed text query `"it is fake or not?"`, the model predicts a single binary label.

- **Input**: short video clip (up to 16 frames @ 224×224) + a tokenized query.
- **Output**: a pooled embedding that is classified against two label prototypes — `FAKE` / `REAL` — using cosine similarity.
- **Training objective**: symmetric InfoNCE (image-to-text + text-to-image) between the predicted query-conditioned video embedding and the label-token embedding produced by the label encoder.

## Architecture

```
video frames ──► PatchEmbed3D ──► [pos] ──► VisionTransformer (x_encoder)
                                                   │
text query tokens ──► Embedding ──► [text_pos] ────┴──► Predictor (cross-attention)
                                                              │
                                                              ▼
                                                   pooled prediction embedding
                                              
```

| Component | File | Role |
| --- | --- | --- |
| `DeepFake` | `model.py` | Top-level wrapper. Wires encoders, predictor, embeddings, positional params. Computes symmetric InfoNCE loss during training; returns pooled prediction embedding at inference. |
| `VisionTransformer` | `x_encoder.py` | Video context encoder (ViT over 3D tubelet patches). |
| `Transformer` | `y_encoder.py` | Label-token encoder (vanilla ViT blocks, no masking). |
| `Predictor` | `predictor.py` | Stack of `CrossAttentionBlock`s that fuses query tokens with video tokens. |
| `PatchEmbed3D` | `utils/patch_embed.py` | 3D tubelet patch embedding for video. |
| `Block`, `Attention`, `CrossAttention`, `MLP` | `utils/modules.py` | Transformer building blocks. |
| `apply_masks`, `get_complement_masks` | `mask/utils.py` | Mask utilities (kept from the JEPA template for compatibility). |
| `DeepFakeDataset`, `collate_fn`, `QUERY` | `prepare.py` | Video dataset + batching + the fixed classification query. |

### Key design choices

- **Prototype-based classification.** Class assignment comes from cosine similarity between the pooled prediction and two prototypes — the mean label-token embedding of `FAKE` and `REAL` computed on a few sampled batches (`get_label_prototypes` in `train.py`).
- **Fixed, classification-only query.** `QUERY = "it is fake or not?"` is the only textual input the model ever sees; it is tokenized once with the `tiktoken` GPT-2 BPE and reused.
- **Symmetric contrastive loss.** The predictor output and the label-encoder output are both mean-pooled; their similarity matrix is used with `cross_entropy` in both directions.

## Repository layout

```
vl-jepa/
├── model.py          # DeepFake model
├── train.py          # Training loop + prototype-based evaluation
├── prepare.py        # DeepFakeDataset, collate_fn, fixed QUERY
├── predictor.py      # Cross-attention predictor
├── x_encoder.py      # Vision Transformer (video)
├── y_encoder.py      # Transformer (label tokens)
├── mask/utils.py     # Mask helpers
└── utils/
    ├── modules.py        # Block, Attention, CrossAttention, MLP
    ├── patch_embed.py    # PatchEmbed3D
    ├── multimask.py
    └── pos_embs.py
```

## Dataset format

`prepare.py` reads video files referenced from CSVs in `--csv_dir`. The loader resolves each row's `file_path` against `--videos_dir`, supporting `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`. Frames are decoded with OpenCV, uniformly subsampled to `MAX_FRAMES = 16`, resized to `224×224`, and normalized with ImageNet statistics. Labels are tokenized with the same GPT-2 BPE used for the query.

## Training

```bash
python train.py \
    --train_videos /path/to/videos \
    --train_csv    /path/to/csvs \
    [--val_videos /path/to/val_videos --val_csv /path/to/val_csvs] \
    [--val_split 0.2] \
    --epochs 100 --batch_size 4 --grad_accum 4 \
    --lr 3e-4 --weight_decay 0.05 --warmup_epochs 5 \
    --save_dir checkpoints
```

If no separate validation set is supplied, the loader deterministically splits the training set 80/20 (seed `42`) and uses the split for validation. `--small_model` switches the architecture to a 4-block / 256-dim variant for quick runs.

Training writes:
- `checkpoints/best.pt` — lowest validation loss.
- `checkpoints/epoch_N.pt` — every `--save_every` epochs.
- `checkpoints/final.pt` — at the end of training.
- `checkpoints/metrics.json` and `metrics.csv` — per-epoch `train_loss`, `val_loss`, `val_accuracy`, `val_precision`, `val_recall`, `val_f1`, `lr`, `elapsed_s`.
- `checkpoints/best_metrics.json` and `best_f1_metrics.json` — best snapshot metadata.

AMP is enabled by default on CUDA (`--no_amp` to disable). Gradient clipping is fixed at `1.0`.

## Evaluation

Each validation pass pools the predictor output, computes cosine similarity to the FAKE/REAL prototypes (re-estimated from up to 8 batches via `get_label_prototypes`), and reports accuracy, precision, recall, and F1 against the `FAKE` = 1 / `REAL` = 0 convention.

## Requirements

- PyTorch (CUDA recommended for AMP)
- `torchvision`
- `tiktoken`
- `opencv-python`
- `Pillow`
- `numpy`

## Notes

- This repository is a **classification-only** adaptation of a JEPA-style architecture; the mask utilities are preserved but the current training/evaluation path does not apply masking to the encoder.
- The model has no learned classification head — class assignment is purely prototype-based at inference time, which means prototypes must be (re-)computed from data that matches the deployment distribution.
