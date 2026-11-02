#!/usr/bin/env python3
"""Train SG-MoE on paired segmentation and medical text examples.

Example:
python train.py --train_jsonl examples/train.jsonl --image_root examples/data \
  --model_id Qwen/Qwen2.5-VL-3B-Instruct --output_dir runs/demo --max_steps 10
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from sgmoe.data import MedicalTaskRecords
from sgmoe.model import load_grounded_model
from sgmoe.optimization import (
    apply_shared_pcgrad, overlap_penalty, private_parameters, shared_parameters,
    task_grads, update_conflict_ema,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--image_root", default=".")
    p.add_argument("--model_id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--output_dir", default="runs/sgmoe")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=0, help="0 = no limit")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mask_size", type=int, default=224)
    p.add_argument("--last_n_layers", type=int, default=4)
    p.add_argument("--shared_rank", type=int, default=4)
    p.add_argument("--private_rank", type=int, default=4)
    p.add_argument("--num_private", type=int, default=3)
    p.add_argument("--top_k", type=int, default=1)
    p.add_argument("--alpha", type=float, default=8.0)
    p.add_argument("--init", choices=("random", "spectral"), default="spectral")
    p.add_argument("--seg_weight", type=float, default=1.0)
    p.add_argument("--text_weight", type=float, default=1.0)
    p.add_argument("--conflict_lambda", type=float, default=0.1)
    p.add_argument("--pcgrad", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--grad_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save_every", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. For CPU, pass --device cpu --dtype fp32")
    if args.device == "cpu" and args.dtype != "fp32":
        raise ValueError("Use --dtype fp32 for CPU training")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    ds = MedicalTaskRecords(args.train_jsonl, args.image_root)
    model, processor = load_grounded_model(
        args.model_id, args.device, args.dtype, mask_size=args.mask_size,
        shared_rank=args.shared_rank, private_rank=args.private_rank,
        num_private=args.num_private, top_k=args.top_k, alpha=args.alpha,
        last_n_layers=args.last_n_layers, init=args.init,
    )
    model.train()
    if args.grad_checkpointing:
        # autograd.grad below requires non-reentrant checkpointing.
        model.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    shared = shared_parameters(model.adapters)
    private = private_parameters(model.adapters)
    print(f"trainable={sum(p.numel() for p in trainable):,} / total={sum(p.numel() for p in model.parameters()):,}")
    print(f"adapters={len(model.adapters)}; seg_examples={len(ds.seg)}; text_examples={len(ds.text)}")

    global_step = 0
    log_file = out / "train_log.jsonl"
    for epoch in range(args.epochs):
        for seg_row, text_row in ds.pairs(args.seed, epoch):
            optimizer.zero_grad(set_to_none=True)
            seg_out = model.segmentation(
                processor, ds.image(seg_row), seg_row["prompt"],
                ds.mask(seg_row, args.mask_size))
            text_out = model.text_task(
                processor, ds.image(text_row), text_row["task"],
                text_row["prompt"], text_row["answer"])
            seg_loss = seg_out["loss"]
            text_loss = text_out["loss"]
            if not torch.isfinite(seg_loss + text_loss):
                raise FloatingPointError("Non-finite task loss")

            # Obtain per-task gradients separately (retaining both forward graphs).
            # Expensive but necessary for gradient conflict measurement and PCGrad.
            ga = task_grads(seg_loss, shared + private)
            gb = task_grads(text_loss, shared + private)
            sg_a, pg_a = ga[:len(shared)], ga[len(shared):]
            sg_b, pg_b = gb[:len(shared)], gb[len(shared):]
            conflict_values = update_conflict_ema(model.adapters, pg_a, pg_b)
            overlap = overlap_penalty(model.adapters, seg_out["routes"], text_out["routes"])
            loss = (args.seg_weight * seg_loss + args.text_weight * text_loss
                    + args.conflict_lambda * overlap)
            loss.backward()
            conflict_cosine = None
            if args.pcgrad:
                conflict_cosine = apply_shared_pcgrad(
                    shared, sg_a, sg_b, args.seg_weight, args.text_weight)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            for adapter in model.adapters.values():
                adapter.last_route_probs = None

            global_step += 1
            stats = {
                "step": global_step, "epoch": epoch,
                "seg_loss": float(seg_loss.detach()),
                "text_loss": float(text_loss.detach()),
                "overlap_penalty": float(overlap.detach()),
                "mean_conflict_ema": sum(conflict_values.values()) / max(1, len(conflict_values)),
                "shared_grad_cosine": conflict_cosine,
                "text_task": text_row["task"],
            }
            print(json.dumps(stats), flush=True)
            with log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(stats) + "\n")
            if args.save_every and global_step % args.save_every == 0:
                model.save_adapter(str(out / f"adapter_step_{global_step}.pt"), vars(args))
            if args.max_steps and global_step >= args.max_steps:
                break
        if args.max_steps and global_step >= args.max_steps:
            break
    model.save_adapter(str(out / "adapter_final.pt"), vars(args))
    print("Saved:", out / "adapter_final.pt")


if __name__ == "__main__":
    main()
