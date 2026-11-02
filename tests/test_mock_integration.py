"""Integrated end-to-end task tests using a tiny fake Qwen interface, no downloads."""
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image

from sgmoe.model import GroundedSpectralQwen
from sgmoe.optimization import apply_shared_pcgrad, private_parameters, shared_parameters, task_grads, update_conflict_ema, overlap_penalty


class TinyQwenBody(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(128, 16)
        self.layers = nn.ModuleList([nn.ModuleDict({
            "q_proj": nn.Linear(16, 16), "v_proj": nn.Linear(16, 16)})])

    def forward(self, input_ids, **kwargs):
        h = self.embedding(input_ids)
        for block in self.layers:
            h = h + block["q_proj"](h) + block["v_proj"](h)
        return SimpleNamespace(last_hidden_state=h)


class TinyQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyQwenBody()
        self.lm_head = nn.Linear(16, 128)
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=16),
            vision_config=SimpleNamespace(spatial_merge_size=2), image_token_id=99)

    def forward(self, input_ids, labels, **kwargs):
        h = self.model(input_ids=input_ids).last_hidden_state
        logits = self.lm_head(h)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 128),
                               labels[:, 1:].reshape(-1), ignore_index=-100)
        return SimpleNamespace(loss=loss)


class TinyProcessor:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = messages[0]["content"][0]["text"].replace(" ", "_")
        prefix = f"START {text} [IMG] ASSISTANT "
        if len(messages) > 1:
            prefix += messages[1]["content"][0]["text"].replace(" ", "_") + " END"
        return prefix

    def __call__(self, text, images, return_tensors="pt"):
        tokens = []
        for word in text[0].split():
            if word == "[IMG]":
                tokens.extend([99] * 16)
            else:
                tokens.append((sum(word.encode()) % 90) + 1)
        return {"input_ids": torch.tensor([tokens]),
                "attention_mask": torch.ones((1, len(tokens)), dtype=torch.long),
                "pixel_values": torch.zeros((1, 3, 4, 4)),
                "image_grid_thw": torch.tensor([[1, 8, 8]])}


def test_integrated_seg_text_pcgrad_checkpoint(tmp_path):
    torch.manual_seed(7)
    model = GroundedSpectralQwen(TinyQwen(), mask_size=32, decoder_hidden=16,
                                 shared_rank=2, private_rank=2, num_private=3,
                                 top_k=1, init="spectral", last_n_layers=1)
    processor = TinyProcessor()
    image = Image.new("RGB", (32, 32), (10, 20, 30))
    target = torch.ones(1, 1, 32, 32)
    seg = model.segmentation(processor, image, "Segment the organ", target)
    txt = model.text_task(processor, image, "vqa", "Is organ visible", "Yes")
    assert seg["mask_logits"].shape == target.shape
    assert torch.isfinite(seg["loss"] + txt["loss"])
    shared = shared_parameters(model.adapters)
    private = private_parameters(model.adapters)
    ga = task_grads(seg["loss"], shared + private)
    gb = task_grads(txt["loss"], shared + private)
    update_conflict_ema(model.adapters, ga[len(shared):], gb[len(shared):])
    overlap = overlap_penalty(model.adapters, seg["routes"], txt["routes"])
    (seg["loss"] + txt["loss"] + 0.1 * overlap).backward()
    cosine = apply_shared_pcgrad(shared, ga[:len(shared)], gb[:len(shared)])
    assert -1.001 <= cosine <= 1.001
    assert any(p.grad is not None for p in shared)
    assert any(p.grad is not None for p in private)
    assert any(p.grad is not None for p in model.mask_decoder.parameters())
    path = tmp_path / "adapter.pt"
    model.save_adapter(str(path), {"model_id": "tiny"})
    model.load_adapter(str(path))
    assert path.exists()
