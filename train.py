import os
import sys
import time
import math
import json
import csv
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from model import DeepFake
from prepare import DeepFakeDataset, collate_fn, QUERY


def parse_args():
    parser = argparse.ArgumentParser(description="Train VL-JEPA DeepFake model")

    # data
    parser.add_argument("--train_videos", type=str, required=True)
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_videos", type=str, default=None)
    parser.add_argument("--val_csv", type=str, default=None)
    parser.add_argument("--val_split", type=float, default=0.2)

    # model
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--vocab_size", type=int, default=50244)

    # training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    parser.add_argument("--small_model", action="store_true", default=False)

    # data params
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--max_frames", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)

    # checkpointing
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)

    # logging
    parser.add_argument("--print_every", type=int, default=1)

    return parser.parse_args()


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def get_label_prototypes(model, device, dataset, max_batches=8):
    model.eval()

    fake_embs = []
    real_embs = []
    loader = DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=0,
        collate_fn=collate_fn,
    )
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        labels = batch["label"]
        y_emb = model.embed(y) + model.pos[:, :y.shape[1], :]
        y_out = model.y_encoder(y_emb)
        for j in range(y_out.shape[0]):
            y_out_j = y_out[j, : batch["y_length"][j].item()]
            if labels[j].item() == 1:
                fake_embs.append(y_out_j.mean(0))
            else:
                real_embs.append(y_out_j.mean(0))

    if not fake_embs or not real_embs:
        return None, None

    fake_proto = torch.stack(fake_embs, 0).mean(0)
    real_proto = torch.stack(real_embs, 0).mean(0)
    fake_proto = F.normalize(fake_proto, dim=0)
    real_proto = F.normalize(real_proto, dim=0)
    return fake_proto, real_proto


@torch.no_grad()
def predict_labels(y_pred, fake_proto, real_proto):
    if fake_proto is None:
        return None

    y_pred_pooled = y_pred.mean(dim=1) if y_pred.dim() == 3 else y_pred
    y_pred_norm = F.normalize(y_pred_pooled, dim=-1)

    sim_fake = y_pred_norm @ fake_proto
    sim_real = y_pred_norm @ real_proto
    preds = (sim_fake > sim_real).long().cpu()
    return preds


@torch.no_grad()
def evaluate(model, dataloader, device, dataset, max_batches=None):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []

    fake_proto, real_proto = get_label_prototypes(
        model, device, dataset, max_batches=4
    )
    model.eval()

    try:
        n_total = len(dataloader) if not max_batches else min(len(dataloader), max_batches)
    except TypeError:
        n_total = "?"
    # print(f"Validating on {n_total} batches...", flush=True)

    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break
        t_b = time.time()

        x = batch["x"].to(device)
        query = batch["query"].to(device)
        y = batch["y"].to(device)

        y_pred, loss = model(x, query, y, train=True)
        total_loss += loss.item()
        n_batches += 1

        preds = predict_labels(y_pred, fake_proto, real_proto)
        if preds is not None:
            labels = batch["label"].long()
            all_preds.append(preds)
            all_labels.append(labels)

        # print(f"val batch {i + 1}/{n_total} completed, took {time.time() - t_b:.2f} sec", flush=True)

    avg_loss = total_loss / max(n_batches, 1)
    metrics = {"loss": avg_loss, "accuracy": 0.0, "precision": 0.0,
               "recall": 0.0, "f1": 0.0, "n_samples": 0}

    if all_preds:
        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        tn = ((preds == 0) & (labels == 0)).sum().item()
        n = tp + fp + fn + tn
        acc = (tp + tn) / max(n, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        metrics.update({
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "n_samples": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    return metrics


def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, args):
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()
    batch_start = time.time()

    for i, batch in enumerate(dataloader):
        x = batch["x"].to(device)
        query = batch["query"].to(device)
        y = batch["y"].to(device)

        if args.amp:
            with autocast(device_type="cuda"):
                _, loss = model(x, query, y, train=True)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
        else:
            _, loss = model(x, query, y, train=True)
            loss = loss / args.grad_accum
            loss.backward()

        if (i + 1) % args.grad_accum == 0 or (i + 1) == len(dataloader):
            if args.amp:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * args.grad_accum
        n_batches += 1

        now = time.time()
        # print(f"batch {i + 1} completed, took {now - batch_start:.2f} sec")
        batch_start = now

    return total_loss / max(n_batches, 1)


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, loss, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "loss": loss,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler):
    ckpt = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"load_checkpoint: missing keys (using init values): {missing}")
    if unexpected:
        print(f"load_checkpoint: unexpected keys (ignored): {unexpected}")
    if missing or unexpected:
        # model architecture changed since this checkpoint was saved (e.g. new
        # params) -- the old optimizer/scheduler state won't line up with the
        # new param groups, so start those fresh rather than crashing.
        print("load_checkpoint: model shape changed, skipping optimizer/scheduler state (starting fresh)")
    else:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt["epoch"], ckpt["loss"]


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Query: \"{QUERY}\"")

    # -- data --
    use_split = not (args.val_videos and args.val_csv)

    if use_split:
        full_dataset = DeepFakeDataset(
            videos_dir=args.train_videos,
            csv_dir=args.train_csv,
            img_size=args.img_size,
            max_frames=args.max_frames,
            is_train=True,
        )
        n = len(full_dataset)
        n_val = int(n * args.val_split)
        n_train = n - n_val

        indices = list(range(n))
        random.seed(42)
        random.shuffle(indices)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
        print(f"Train: {n_train} samples | Val: {n_val} samples (split={args.val_split})")
    else:
        full_dataset = DeepFakeDataset(
            videos_dir=args.train_videos,
            csv_dir=args.train_csv,
            img_size=args.img_size,
            max_frames=args.max_frames,
            is_train=True,
        )
        train_loader = DataLoader(
            full_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )

        val_dataset = DeepFakeDataset(
            videos_dir=args.val_videos,
            csv_dir=args.val_csv,
            img_size=args.img_size,
            max_frames=args.max_frames,
            is_train=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
        print(f"Train: {len(full_dataset)} samples | Val: {len(val_dataset)} samples (separate)")

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # -- model --
    if args.small_model:
        model = DeepFake(
            embed_dim=256,
            depth=4,
            num_heads=8,
            pred_depth=2,
            pred_heads=8,
            vocab_size=args.vocab_size,
        ).to(device)
    else:
        model = DeepFake(
            embed_dim=args.embed_dim,
            vocab_size=args.vocab_size,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {n_params:.2f}M")

    # -- optimizer --
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    if len(train_loader) == 0:
        raise ValueError(
            "No training batches were created. Reduce --batch_size or provide more data."
        )
    updates_per_epoch = math.ceil(len(train_loader) / max(args.grad_accum, 1))
    total_steps = updates_per_epoch * args.epochs
    warmup_steps = updates_per_epoch * args.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    args.amp = args.amp and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=True) if args.amp else None

    # -- resume --
    start_epoch = 0
    best_val_loss = float("inf")
    best_val_f1 = -1.0
    if args.resume and os.path.exists(args.resume):
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        start_epoch += 1
        print(f"Resumed from {args.resume}, epoch {start_epoch}")

    # -- save dir --
    os.makedirs(args.save_dir, exist_ok=True)

    metrics_path_json = os.path.join(args.save_dir, "metrics.json")
    metrics_path_csv = os.path.join(args.save_dir, "metrics.csv")
    history = []
    if args.resume and os.path.exists(metrics_path_json):
        try:
            with open(metrics_path_json, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    if history:
        best_val_loss = min(row.get("val_loss", float("inf")) for row in history)
        best_val_f1 = max(row.get("val_f1", -1.0) for row in history)

    def save_metrics():
        with open(metrics_path_json, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        if history:
            keys = list(history[0].keys())
            with open(metrics_path_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for row in history:
                    w.writerow(row)

    # -- training loop --
    print(f"\nStarting training for {args.epochs} epochs")
    print("-" * 60)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, args
        )

        val_metrics = evaluate(
            model, val_loader, device, full_dataset if use_split else val_dataset
        )
        val_loss = val_metrics["loss"]

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        log_row = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_metrics.get("accuracy", 0.0)),
            "val_precision": float(val_metrics.get("precision", 0.0)),
            "val_recall": float(val_metrics.get("recall", 0.0)),
            "val_f1": float(val_metrics.get("f1", 0.0)),
            "val_n_samples": int(val_metrics.get("n_samples", 0)),
            "lr": float(lr_now),
            "elapsed_s": float(elapsed),
        }
        history.append(log_row)
        save_metrics()

        if (epoch + 1) % args.print_every == 0:
            msg = (
                f"Epoch {epoch+1}/{args.epochs} | "
                f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
                f"val_acc: {val_metrics.get('accuracy', 0.0):.4f} | "
                f"val_prec: {val_metrics.get('precision', 0.0):.4f} | "
                f"val_rec: {val_metrics.get('recall', 0.0):.4f} | "
                f"val_f1: {val_metrics.get('f1', 0.0):.4f} | "
                f"lr: {lr_now:.2e} | {elapsed:.1f}s"
            )
            print(msg)

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch+1}.pt")
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, train_loss, ckpt_path
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.save_dir, "best.pt")
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, val_loss, best_path
            )
            with open(os.path.join(args.save_dir, "best_metrics.json"), "w") as f:
                json.dump({**log_row, **val_metrics}, f, indent=2)

        if val_metrics.get("f1", 0.0) > best_val_f1:
            best_val_f1 = val_metrics.get("f1", 0.0)
            with open(os.path.join(args.save_dir, "best_f1_metrics.json"), "w") as f:
                json.dump({**log_row, **val_metrics}, f, indent=2)

    save_metrics()

    # -- final save --
    final_path = os.path.join(args.save_dir, "final.pt")
    save_checkpoint(model, optimizer, scheduler, scaler, args.epochs - 1, train_loss, final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
