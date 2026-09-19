import csv
import torch
import random
import cv2
import tiktoken
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple


QUERY = "it is fake or not?"
IMG_SIZE = 224
MAX_FRAMES = 16


class DeepFakeDataset(Dataset):
    def __init__(
        self,
        videos_dir: str,
        csv_dir: str,
        img_size: int = IMG_SIZE,
        max_frames: int = MAX_FRAMES,
        max_query_len: int = 32,
        max_label_len: int = 8,
        is_train: bool = True,
    ):
        self.videos_dir = Path(videos_dir)
        self.csv_dir = Path(csv_dir)
        self.img_size = img_size
        self.max_frames = max_frames
        self.max_query_len = max_query_len
        self.max_label_len = max_label_len
        self.is_train = is_train

        self.enc = tiktoken.get_encoding("gpt2")
        self.query_tokens = torch.tensor(
            self.enc.encode(QUERY), dtype=torch.long
        )

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.samples = self._load_csv_data()
        print(f"Loaded {len(self.samples)} samples from {csv_dir}")

    def _resolve_video_path(
        self, csv_stem: str, file_path: str
    ) -> Optional[Path]:
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        raw_path = file_path.strip().replace("\\", "/")
        rel = Path(raw_path)
        video_stem = rel.stem
        folder = self.videos_dir / csv_stem
        candidates: List[Path] = []

        # CSV paths are commonly relative to the dataset folder or to the
        # folder represented by the CSV filename.
        if rel.is_absolute():
            candidates.append(rel)
        else:
            candidates.extend([
                folder / rel,
                self.videos_dir / rel,
                folder / rel.name,
                self.videos_dir / rel.name,
            ])

        for candidate in candidates:
            if candidate.is_file():
                return candidate
            if candidate.suffix.lower() not in video_exts:
                for ext in video_exts:
                    with_ext = candidate.with_suffix(ext)
                    if with_ext.is_file():
                        return with_ext

        # Fall back to a stem search, but prefer the CSV's matching folder so
        # duplicate filenames from different dataset parts are not mixed.
        for ext in video_exts:
            matches = list(folder.glob(f"**/{video_stem}{ext}")) if folder.is_dir() else []
            if matches:
                matches.sort()
                return matches[0]

        for ext in video_exts:
            matches = list(self.videos_dir.glob(f"**/{video_stem}{ext}"))
            if matches:
                matches.sort()
                return matches[0]

        return None

    def _load_csv_data(self) -> List[Dict]:
        samples = []

        for csv_file in sorted(self.csv_dir.glob("*.csv")):
            csv_stem = csv_file.stem
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    file_path = (
                        row.get("File Path")
                        or row.get("file_path")
                        or row.get("filename")
                    )
                    label = row.get("Label") or row.get("label") or row.get("class")

                    if file_path is None or label is None:
                        continue

                    label = label.strip().upper()

                    if label in ("FAKE", "1", "TRUE", "YES", "1.0", "T"):
                        label_id = 1
                    elif label in ("REAL", "0", "FALSE", "NO", "0.0", "F", "ORIGINAL", "PRISTINE"):
                        label_id = 0
                    else:
                        print(
                            f"Warning: unknown label '{label}' in '{csv_file.name}', skipping"
                        )
                        continue

                    video_path = self._resolve_video_path(csv_stem, file_path)
                    if video_path is None:
                        print(
                            f"Warning: no video found for '{file_path}' "
                            f"(csv '{csv_file.name}'), skipping"
                        )
                        continue

                    samples.append({
                        "video_path": str(video_path),
                        "label": label,
                        "label_id": label_id,
                    })

        return samples

    def _read_video_frames(self, video_path: str) -> List[torch.Tensor]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: cannot open '{video_path}'")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []

        if total_frames <= self.max_frames:
            indices = list(range(total_frames))
        elif self.is_train:
            indices = sorted(random.sample(range(total_frames), self.max_frames))
        else:
            step = total_frames / self.max_frames
            indices = [int(i * step) for i in range(self.max_frames)]

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = self.transform(frame)
                frames.append(frame)

        cap.release()
        return frames

    def _tokenize_label(self, label: str) -> torch.Tensor:
        label_norm = label.strip().upper()
        if label_norm in ("FAKE", "1", "TRUE", "YES", "1.0", "T"):
            text = "fake"
        elif label_norm in ("REAL", "0", "FALSE", "NO", "0.0", "F", "ORIGINAL", "PRISTINE"):
            text = "not fake"
        else:
            raise ValueError(f"Unsupported label: {label}")
        tokens = self.enc.encode(text)
        tokens = tokens[: self.max_label_len]
        return torch.tensor(tokens, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        frames = self._read_video_frames(sample["video_path"])
        if not frames:
            frames = [torch.zeros(3, self.img_size, self.img_size)]

        x = torch.stack(frames, dim=1)  # [C, T, H, W]

        query = self.query_tokens[: self.max_query_len].clone()
        label_tokens = self._tokenize_label(sample["label"])

        return {
            "x": x,
            "query": query,
            "y": label_tokens,
            "label": torch.tensor(sample["label_id"], dtype=torch.long),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    max_q = max(item["query"].shape[0] for item in batch)
    max_t = max(item["x"].shape[1] for item in batch)
    # PatchEmbed3D uses a 2-frame tubelet and 16x16 spatial patches.
    # The model predicts one target embedding per resulting visual token.
    max_t = max(2, ((max_t + 1) // 2) * 2)
    patch_tokens = (max_t // 2) * (batch[0]["x"].shape[2] // 16) * (
        batch[0]["x"].shape[3] // 16
    )

    queries = []
    labels = []
    label_ids = []
    label_lengths = []
    images = []

    for item in batch:
        q = item["query"]
        y = item["y"]
        x = item["x"]

        q_padded = torch.zeros(max_q, dtype=torch.long)
        q_padded[: q.shape[0]] = q

        y_padded = y.repeat((patch_tokens + y.shape[0] - 1) // y.shape[0])[:patch_tokens]

        C, T_i, H, W = x.shape
        if T_i < max_t:
            pad = torch.zeros(C, max_t - T_i, H, W)
            x = torch.cat([x, pad], dim=1)

        queries.append(q_padded)
        labels.append(y_padded)
        label_ids.append(item["label"])
        label_lengths.append(torch.tensor(patch_tokens, dtype=torch.long))
        images.append(x)

    return {
        "x": torch.stack(images),
        "query": torch.stack(queries),
        "y": torch.stack(labels),
        "label": torch.stack(label_ids),
        "y_length": torch.stack(label_lengths),
    }


def create_dataloader(
    videos_dir: str,
    csv_dir: str,
    batch_size: int = 16,
    img_size: int = IMG_SIZE,
    max_frames: int = MAX_FRAMES,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    max_query_len: int = 32,
    max_label_len: int = 8,
    is_train: bool = True,
) -> Tuple[DataLoader, DeepFakeDataset]:
    dataset = DeepFakeDataset(
        videos_dir=videos_dir,
        csv_dir=csv_dir,
        img_size=img_size,
        max_frames=max_frames,
        max_query_len=max_query_len,
        max_label_len=max_label_len,
        is_train=is_train,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
    )

    return dataloader, dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare DeepFake dataset")
    parser.add_argument("--videos_dir", type=str, required=True, help="Path to videos folder")
    parser.add_argument("--csv_dir", type=str, required=True, help="Path to CSV folder")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    dataloader, dataset = create_dataloader(
        videos_dir=args.videos_dir,
        csv_dir=args.csv_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        max_frames=args.max_frames,
        num_workers=args.num_workers,
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"Query: \"{QUERY}\" -> tokens: {dataset.query_tokens.tolist()}")
    print(f"Input format: (B, C, T, H, W) = (B, 3, <=16, 224, 224)")

    for batch in dataloader:
        print(f"\nBatch shapes:")
        print(f"  x:     {batch['x'].shape}  # (B, C, T, H, W)")
        print(f"  query: {batch['query'].shape}")
        print(f"  y:     {batch['y'].shape}")
        break
