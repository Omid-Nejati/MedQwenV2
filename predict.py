#!/usr/bin/env python3
"""Generate a binary mask or text answer using a trained SG-MoE adapter."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sgmoe.model import load_grounded_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--task", choices=("seg", "vqa", "report", "classification"), default="seg")
    p.add_argument("--model_id", default=None, help="defaults to checkpoint training model ID")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    p.add_argument("--output", default="prediction_mask.png")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max_new_tokens", type=int, default=128)
    args = p.parse_args()
    checkpoint = torch.load(args.adapter, map_location="cpu", weights_only=True)
    meta = checkpoint.get("metadata", {})
    kwargs = {k: meta[k] for k in (
        "mask_size", "last_n_layers", "shared_rank", "private_rank", "num_private",
        "top_k", "alpha", "init") if k in meta}
    model, processor = load_grounded_model(
        args.model_id or meta.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct"),
        args.device, args.dtype, **kwargs)
    model.load_adapter(args.adapter)
    model.eval()
    image = Image.open(args.image).convert("RGB")
    with torch.no_grad():
        if args.task == "seg":
            result = model.segmentation(processor, image, args.prompt)
            prob = result["mask_logits"].sigmoid()[0, 0].float().cpu().numpy()
            im = Image.fromarray(np.uint8(prob > args.threshold) * 255)
            im = im.resize(image.size, Image.Resampling.NEAREST)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            im.save(args.output)
            print("Saved binary mask:", args.output)
        else:
            model.set_task(args.task)
            inputs = model.prepare_prompt(processor, args.task, args.prompt, image)
            ids = model.backbone.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                          do_sample=False, use_cache=True)
            new_ids = ids[0][inputs["input_ids"].shape[1]:]
            print(processor.tokenizer.decode(new_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
