#!/usr/bin/env python3
"""
Training script for Experiment 7.

This script fine-tunes the differential CLIP model on DIOR / mini-DIOR datasets
using RemoteCLIP weights as initialization. Only the lambda parameters and the
query projections are trained by default, matching the experimental protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

from dataset import DIORDataset, build_default_transform, build_train_transform
from diff_clip import (
    DiffCLIP_VITB16,
    DifferentialMultiheadAttention,
)
from diff_attention import DiffAttention
from tokenizer import SimpleTokenizer

try:
    from argparse import BooleanOptionalAction
except ImportError:  # pragma: no cover
    class BooleanOptionalAction(argparse.Action):
        def __init__(self, option_strings, dest, default=None, **kwargs):
            opts = []
            for option in option_strings:
                opts.append(option)
                if option.startswith("--"):
                    opts.append(option.replace("--", "--no-"))
            super().__init__(opts, dest, nargs=0, default=default, **kwargs)

        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, not option_string.startswith("--no-"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DiffCLIP on DIOR / mini-DIOR.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="datasets/DIOR",
        help="Path to the DIOR dataset root.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/RemoteCLIP-ViT-B-32.pt",
        help="Path to the RemoteCLIP checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiment7/outputs",
        help="Directory to save checkpoints and logs.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--tune-lambda",
        action=BooleanOptionalAction,
        default=True,
        help="Fine-tune the lambda parameters of differential attention (default: enabled).",
    )
    parser.add_argument(
        "--tune-q",
        action=BooleanOptionalAction,
        default=True,
        help="Fine-tune the query projection weights and biases (default: enabled).",
    )
    parser.add_argument(
        "--tune-projection",
        action=BooleanOptionalAction,
        default=False,
        help="Fine-tune the final image/text projection layers.",
    )
    parser.add_argument(
        "--tune-logit-scale",
        action=BooleanOptionalAction,
        default=False,
        help="Fine-tune the logit scale parameter.",
    )
    parser.add_argument(
        "--evaluate-every",
        type=int,
        default=1,
        help="Run validation after this many epochs.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save a checkpoint after this many epochs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. cuda:0). Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=0.5,
        help="Gradient clipping threshold.",
    )
    parser.add_argument(
        "--scheduler",
        choices=("cosine", "plateau", "none"),
        default="plateau",
        help="Learning rate scheduler strategy.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience based on validation loss.",
    )
    args = parser.parse_args()
    if not (args.tune_lambda or args.tune_q or args.tune_projection or args.tune_logit_scale):
        parser.error(
            "At least one of --tune-lambda/--tune-q/--tune-projection/--tune-logit-scale must be set."
        )
    return args


def setup_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def freeze_all_parameters(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def configure_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    tune_lambda: bool,
    tune_q: bool,
    tune_projection: bool,
    tune_logit_scale: bool,
) -> Optimizer:
    param_groups: List[Dict[str, object]] = []

    if tune_lambda:
        lambda_params: List[nn.Parameter] = []
        for module in model.modules():
            if isinstance(module, DifferentialMultiheadAttention) or isinstance(module, DiffAttention):
                for name in ("lambda_q1", "lambda_k1", "lambda_q2", "lambda_k2"):
                    param = getattr(module, name, None)
                    if param is not None:
                        param.requires_grad = True
                        lambda_params.append(param)
        if lambda_params:
            param_groups.append({"params": lambda_params, "lr": lr, "weight_decay": 0.0})

    if tune_q:
        q_params: List[nn.Parameter] = []
        for module in model.modules():
            if isinstance(module, DifferentialMultiheadAttention) or isinstance(module, DiffAttention):
                if hasattr(module, "q_proj"):
                    module.q_proj.weight.requires_grad = True
                    q_params.append(module.q_proj.weight)
                    if module.q_proj.bias is not None:
                        module.q_proj.bias.requires_grad = True
                        q_params.append(module.q_proj.bias)
        if q_params:
            param_groups.append({"params": q_params, "lr": lr, "weight_decay": weight_decay})

    if tune_projection:
        proj_params: List[nn.Parameter] = []
        for name in ("image_projection", "text_projection"):
            param = getattr(model, name, None)
            if param is not None:
                param.requires_grad = True
                proj_params.append(param)
        if proj_params:
            param_groups.append({"params": proj_params, "lr": lr, "weight_decay": weight_decay})

    if tune_logit_scale and hasattr(model, "logit_scale"):
        model.logit_scale.requires_grad = True
        param_groups.append({"params": [model.logit_scale], "lr": lr, "weight_decay": 0.0})

    if not param_groups:
        raise RuntimeError("No parameters were selected for optimization.")

    return AdamW(param_groups)


def build_dataloaders(
    data_root: str,
    batch_size: int,
    num_workers: int,
    tokenizer: SimpleTokenizer,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    train_dataset = DIORDataset(
        root=data_root,
        split="train",
        transform=build_train_transform(),
        augment=True,
    )
    val_dataset = DIORDataset(
        root=data_root,
        split="val",
        transform=build_default_transform(),
        augment=False,
        text_templates=("a satellite photo of {}",),
    )

    def collate_fn(batch):
        images, texts, metas = zip(*batch)
        image_tensor = torch.stack(images)
        text_tensor = torch.stack([tokenizer(text) for text in texts])
        return image_tensor, text_tensor, metas

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, train_dataset.classes


def clip_contrastive_loss(
    image_embed: torch.Tensor, text_embed: torch.Tensor, logit_scale: torch.Tensor
) -> torch.Tensor:
    image_embed = F.normalize(image_embed, dim=-1)
    text_embed = F.normalize(text_embed, dim=-1)
    logits_per_image = logit_scale * image_embed @ text_embed.t()
    logits_per_text = logits_per_image.t()
    batch_size = image_embed.size(0)
    labels = torch.arange(batch_size, device=image_embed.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return 0.5 * (loss_i + loss_t)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    grad_clip: float = 0.5,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    start_time = time.time()
    trainable_params = [
        param for group in optimizer.param_groups for param in group["params"] if param.requires_grad
    ]

    for step, (images, text_tokens, _) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        text_tokens = text_tokens.to(device, non_blocking=True)

        outputs = model(images, text_tokens)
        loss = clip_contrastive_loss(
            outputs["image_embed"], outputs["text_embed"], outputs["logit_scale"]
        )

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(trainable_params, grad_clip)
        optimizer.step()

        with torch.no_grad():
            image_embed = F.normalize(outputs["image_embed"], dim=-1)
            text_embed = F.normalize(outputs["text_embed"], dim=-1)
            logits = outputs["logit_scale"] * image_embed @ text_embed.t()
            preds = logits.argmax(dim=-1)
            labels = torch.arange(images.size(0), device=device)
            total_correct += (preds == labels).sum().item()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    elapsed = time.time() - start_time
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "time": elapsed,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, text_tokens, _ in dataloader:
        images = images.to(device, non_blocking=True)
        text_tokens = text_tokens.to(device, non_blocking=True)
        outputs = model(images, text_tokens)
        loss = clip_contrastive_loss(
            outputs["image_embed"], outputs["text_embed"], outputs["logit_scale"]
        )

        image_embed = F.normalize(outputs["image_embed"], dim=-1)
        text_embed = F.normalize(outputs["text_embed"], dim=-1)
        logits = outputs["logit_scale"] * image_embed @ text_embed.t()
        preds = logits.argmax(dim=-1)
        labels = torch.arange(images.size(0), device=device)
        total_correct += (preds == labels).sum().item()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best: Optional[float] = None
        self.counter = 0

    def step(self, value: float) -> bool:
        if self.best is None or value < self.best - self.min_delta:
            self.best = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    output_dir: Path,
    classes: List[str],
) -> None:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "classes": classes,
    }
    ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
    torch.save(state, ckpt_path)


def main() -> None:
    args = parse_args()
    device = setup_device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    tokenizer = SimpleTokenizer()
    train_loader, val_loader, classes = build_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tokenizer=tokenizer,
    )

    model = DiffCLIP_VITB16()
    model.load_remoteclip_weights(args.checkpoint)
    model.to(device)

    freeze_all_parameters(model)
    optimizer = configure_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        tune_lambda=args.tune_lambda,
        tune_q=args.tune_q,
        tune_projection=args.tune_projection,
        tune_logit_scale=args.tune_logit_scale,
    )

    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=1e-7,
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-7,
        )
    else:
        scheduler = None

    history: List[Dict[str, float]] = []
    early_stopping = EarlyStopping(patience=args.patience)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
        )
        log_entry: Dict[str, float] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_time": train_metrics["time"],
        }
        val_metrics = None
        if epoch % args.evaluate_every == 0:
            val_metrics = evaluate(model, val_loader, device=device)
            log_entry.update({
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            })
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics["loss"])
                else:
                    scheduler.step()
        elif scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                log_entry["lr"] = optimizer.param_groups[0]["lr"]
            else:
                log_entry["lr"] = scheduler.get_last_lr()[0]
        else:
            log_entry["lr"] = optimizer.param_groups[0]["lr"]
        history.append(log_entry)

        print(
            f"[Epoch {epoch:03d}] "
            f"Loss: {log_entry['train_loss']:.4f} "
            f"Acc: {log_entry['train_accuracy']:.4f} "
            f"ValLoss: {log_entry.get('val_loss', float('nan')):.4f} "
            f"ValAcc: {log_entry.get('val_accuracy', float('nan')):.4f} "
            f"LR: {log_entry['lr']:.2e} "
            f"Time: {train_metrics['time']:.1f}s"
        )

        if val_metrics is not None and val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(model, optimizer, epoch, output_dir, classes)
            with open(output_dir / "best_epoch.txt", "w", encoding="utf-8") as bf:
                bf.write(str(epoch))
        elif epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch, output_dir, classes)

        with open(output_dir / "training_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        if val_metrics is not None and early_stopping.step(val_metrics["loss"]):
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    # Save final state for convenience.
    final_epoch = history[-1]["epoch"] if history else 0
    save_checkpoint(model, optimizer, final_epoch, output_dir, classes)


if __name__ == "__main__":
    main()

