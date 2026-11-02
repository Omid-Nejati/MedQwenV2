"""Generate a tiny synthetic JSONL dataset for pipeline testing only.

These geometric images are NOT medical training data and are not suitable for
reporting clinical or segmentation research performance.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

root = Path(__file__).parent
folder = root / "data"
folder.mkdir(exist_ok=True)
records = []
for i in range(4):
    img = Image.new("RGB", (224, 224), (30 + i * 10, 30 + i * 10, 30 + i * 10))
    mask = Image.new("L", (224, 224), 0)
    x0 = 60 + i * 5
    ImageDraw.Draw(img).ellipse((x0, 70, x0 + 75, 145), fill=(180, 180, 180))
    ImageDraw.Draw(mask).ellipse((x0, 70, x0 + 75, 145), fill=255)
    img.save(folder / f"im{i}.png")
    mask.save(folder / f"mask{i}.png")
    records.append({"task": "seg", "image": f"im{i}.png", "mask": f"mask{i}.png",
                    "prompt": "Segment the bright circular object in this image."})
    records.append({"task": "vqa", "image": f"im{i}.png",
                    "prompt": "Is there a bright circular object in the image?",
                    "answer": "Yes."})
    records.append({"task": "classification", "image": f"im{i}.png",
                    "prompt": "Classify whether a bright circular object is present. Reply yes or no.",
                    "answer": "yes"})
    records.append({"task": "report", "image": f"im{i}.png",
                    "prompt": "Describe the geometric image in one sentence.",
                    "answer": "A bright circular shape is visible on a dark background."})
with (root / "train.jsonl").open("w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record) + "\n")
print("Generated", len(records), "synthetic examples in", folder)
