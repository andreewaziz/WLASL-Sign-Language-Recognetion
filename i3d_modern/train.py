from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .cache import default_cache_dir
from .checkpoint import load_checkpoint, load_model_state, save_checkpoint
from .config import config_to_dict, load_config
from .data import WLASLVideoDataset
from .metrics import AverageMeter, append_jsonl, topk_correct
from .model import build_i3d
from .runtime import dataloader_worker_count, get_device, gpu_memory_mb, make_output_dir, set_seed


def make_targets(labels: torch.Tensor, num_classes: int, frames: int) -> torch.Tensor:
    one_hot = F.one_hot(labels, num_classes=num_classes).float()
    return one_hot.unsqueeze(-1).expand(-1, -1, frames)


def compute_i3d_loss(per_frame_logits: torch.Tensor, labels: torch.Tensor, num_classes: int, frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    per_frame_logits = F.interpolate(per_frame_logits, size=frames, mode="linear", align_corners=False)
    temporal_targets = make_targets(labels, num_classes, frames)
    clip_targets = temporal_targets[:, :, 0]
    loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, temporal_targets)
    cls_logits = per_frame_logits.max(dim=2).values
    cls_loss = F.binary_cross_entropy_with_logits(cls_logits, clip_targets)
    return 0.5 * loc_loss + 0.5 * cls_loss, cls_logits


def make_loader(dataset: WLASLVideoDataset, batch_size: int, num_workers: int, train: bool, device: torch.device) -> DataLoader:
    workers = dataloader_worker_count(num_workers)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def make_scaler(device: torch.device, enabled: bool):
    amp_enabled = enabled and device.type == "cuda"
    try:
        return torch.amp.GradScaler(device.type, enabled=amp_enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=amp_enabled)


def train_one_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    num_classes: int,
    amp: bool,
    grad_accum_steps: int,
    global_step: int,
    max_steps: int,
    limit_batches: int | None,
) -> tuple[dict, int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = AverageMeter()
    totals = {"top1": 0, "top5": 0, "top10": 0, "count": 0}
    amp_enabled = amp and device.type == "cuda"

    iterator = enumerate(tqdm(loader, desc="train", leave=False), start=1)
    for batch_idx, (inputs, labels, _) in iterator:
        if limit_batches is not None and batch_idx > limit_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        frames = inputs.size(2)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            per_frame_logits = model(inputs)
            loss, cls_logits = compute_i3d_loss(per_frame_logits, labels, num_classes, frames)
            scaled_loss = loss / grad_accum_steps

        scaler.scale(scaled_loss).backward()
        if batch_idx % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        batch_size = labels.size(0)
        losses.update(loss.item(), batch_size)
        for key, value in topk_correct(cls_logits.detach(), labels).items():
            totals[key] += value
        totals["count"] += batch_size

        if global_step >= max_steps:
            break

    metrics = {"loss": losses.avg}
    for key in ("top1", "top5", "top10"):
        metrics[key] = totals[key] / max(1, totals["count"])
    return metrics, global_step


@torch.no_grad()
def validate(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    amp: bool,
    limit_batches: int | None,
) -> dict:
    model.eval()
    losses = AverageMeter()
    totals = {"top1": 0, "top5": 0, "top10": 0, "count": 0}
    amp_enabled = amp and device.type == "cuda"

    for batch_idx, (inputs, labels, _) in enumerate(tqdm(loader, desc="val", leave=False), start=1):
        if limit_batches is not None and batch_idx > limit_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        frames = inputs.size(2)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            per_frame_logits = model(inputs)
            loss, cls_logits = compute_i3d_loss(per_frame_logits, labels, num_classes, frames)

        batch_size = labels.size(0)
        losses.update(loss.item(), batch_size)
        for key, value in topk_correct(cls_logits, labels).items():
            totals[key] += value
        totals["count"] += batch_size

    metrics = {"loss": losses.avg}
    for key in ("top1", "top5", "top10"):
        metrics[key] = totals[key] / max(1, totals["count"])
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train modern RGB I3D on WLASL")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--pretrained")
    parser.add_argument("--resume")
    parser.add_argument("--output", default="runs/i3d")
    parser.add_argument("--device")
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--limit-val-batches", type=int)
    parser.add_argument("--refresh-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    split_file = config.resolved_split_file(args.config)
    output_dir = make_output_dir(args.output)
    cache_dir = Path(config.cache_dir) if config.cache_dir else default_cache_dir(output_dir, args.data_root)
    device = get_device(args.device)
    set_seed(config.seed)

    train_dataset = WLASLVideoDataset(
        split_file=split_file,
        data_root=args.data_root,
        splits=config.train_splits,
        clip_frames=config.clip_frames,
        image_size=config.image_size,
        resize_size=config.resize_size,
        train=True,
        cache_mode=config.cache_mode,
        cache_dir=cache_dir,
        refresh_cache=args.refresh_cache,
    )
    val_dataset = WLASLVideoDataset(
        split_file=split_file,
        data_root=args.data_root,
        splits=config.val_splits,
        clip_frames=config.clip_frames,
        image_size=config.image_size,
        resize_size=config.resize_size,
        train=False,
        cache_mode=config.cache_mode,
        cache_dir=cache_dir,
        refresh_cache=args.refresh_cache,
    )

    missing = sorted(set(train_dataset.missing_video_ids + val_dataset.missing_video_ids))
    if missing:
        missing_path = output_dir / "missing_videos.txt"
        missing_path.write_text("\n".join(missing) + "\n", encoding="utf-8")
        print(f"Missing videos: {len(missing)} written to {missing_path}")
    if not train_dataset or not val_dataset:
        raise RuntimeError(f"Empty dataset. train={len(train_dataset)} val={len(val_dataset)} data_root={args.data_root}")

    train_loader = make_loader(train_dataset, config.batch_size, config.num_workers, True, device)
    val_loader = make_loader(val_dataset, config.batch_size, config.num_workers, False, device)

    model = build_i3d(config.num_classes, device=device, pretrained=args.pretrained if not args.resume else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=config.adam_eps, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.3)
    scaler = make_scaler(device, config.amp)

    start_epoch = 0
    global_step = 0
    best_metric = 0.0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, device)
        load_model_state(model, checkpoint)
        if checkpoint.get("optimizer"):
            optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_metric = float(checkpoint.get("best_metric", 0.0))

    config_dict = config_to_dict(config)
    (output_dir / "config.json").write_text(json.dumps(config_dict, indent=2, sort_keys=True), encoding="utf-8")

    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        train_metrics, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            num_classes=config.num_classes,
            amp=config.amp,
            grad_accum_steps=max(1, config.grad_accum_steps),
            global_step=global_step,
            max_steps=config.max_steps,
            limit_batches=args.limit_train_batches,
        )

        should_eval = (epoch + 1) % config.eval_interval == 0
        val_metrics = validate(
            model=model,
            loader=val_loader,
            device=device,
            num_classes=config.num_classes,
            amp=config.amp,
            limit_batches=args.limit_val_batches,
        ) if should_eval else {"loss": float("nan"), "top1": 0.0, "top5": 0.0, "top10": 0.0}
        scheduler.step(val_metrics["loss"])

        row = {
            "epoch": epoch,
            "global_step": global_step,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - epoch_start,
            "gpu_memory_mb": gpu_memory_mb(device),
            "train": train_metrics,
            "val": val_metrics,
        }
        append_jsonl(output_dir / "metrics.jsonl", row)
        print(json.dumps(row, indent=2, sort_keys=True))

        if (epoch + 1) % config.checkpoint_interval == 0:
            save_checkpoint(
                output_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config_dict,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
            )

        if val_metrics["top1"] > best_metric:
            best_metric = val_metrics["top1"]
            save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config=config_dict,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
            )

        if global_step >= config.max_steps:
            break


if __name__ == "__main__":
    main()
