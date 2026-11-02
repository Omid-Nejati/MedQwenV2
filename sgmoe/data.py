"""Single-image medical segmentation + text-task JSONL dataset utilities."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .adapters import TASKS


class MedicalTaskRecords:
    def __init__(self, path: str | Path, image_root: str | Path = "."):
        self.path = Path(path)
        self.image_root = Path(image_root)
        self.seg: list[dict] = []
        self.text: list[dict] = []
        with self.path.open(encoding="utf-8") as file:
            for n, line in enumerate(file, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                task = row.get("task")
                if task not in TASKS:
                    raise ValueError(f"Line {n}: task must be one of {TASKS}")
                for field in ("image", "prompt"):
                    if not row.get(field):
                        raise ValueError(f"Line {n}: missing '{field}'")
                if task == "seg":
                    if not row.get("mask"):
                        raise ValueError(f"Line {n}: segmentation requires a 'mask' path")
                    self.seg.append(row)
                else:
                    if not isinstance(row.get("answer"), str) or not row["answer"]:
                        raise ValueError(f"Line {n}: text task requires 'answer'")
                    self.text.append(row)
        if not self.seg or not self.text:
            raise ValueError("Need at least one segmentation and one text-task example for paired training")

    def pairs(self, seed: int, epoch: int):
        rng = random.Random(seed + epoch)
        s = list(self.seg)
        t = list(self.text)
        rng.shuffle(s)
        rng.shuffle(t)
        for i in range(max(len(s), len(t))):
            yield s[i % len(s)], t[i % len(t)]

    def image(self, row: dict) -> Image.Image:
        return Image.open(self.image_root / row["image"]).convert("RGB")

    def mask(self, row: dict, size: int) -> torch.Tensor:
        # Binary foreground mask. Multiclass labels require conversion to a
        # binary target for the named organ/pathology before training.
        img = Image.open(self.image_root / row["mask"]).convert("L")
        img = img.resize((size, size), resample=Image.Resampling.NEAREST)
        target = (np.array(img, dtype=np.uint8) > 0).astype(np.float32)
        return torch.from_numpy(target)[None, None]  # [1,1,H,W]
