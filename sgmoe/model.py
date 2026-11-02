"""Qwen2.5-VL with conflict-aware spectral LoRA and text-conditioned masks.

This prototype uses a lightweight conv mask decoder, NOT SAM2. All image token
features are taken from the frozen-vision/Qwen language backbone; updates flow
through the injected LoRA experts in the language layers.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from PIL import Image

from .adapters import inject_spectral_adapters, route_snapshot, set_adapters_task, trainable_state
from .optimization import dice_bce_loss


class MaskDecoder(nn.Module):
    def __init__(self, text_hidden_size: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(text_hidden_size, hidden, kernel_size=1),
            nn.GroupNorm(8 if hidden % 8 == 0 else 1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, spatial_tokens: Tensor, height: int, width: int, mask_size: int) -> Tensor:
        if spatial_tokens.ndim != 2 or spatial_tokens.shape[0] != height * width:
            raise ValueError("Spatial token count does not match reconstructed grid")
        fmap = spatial_tokens.T.reshape(1, spatial_tokens.shape[1], height, width).float()
        low_res = self.net(fmap)
        return F.interpolate(low_res, size=(mask_size, mask_size), mode="bilinear", align_corners=False)


class GroundedSpectralQwen(nn.Module):
    def __init__(self, backbone: nn.Module, *, mask_size: int = 224, decoder_hidden: int = 128,
                 shared_rank: int = 4, private_rank: int = 4, num_private: int = 3,
                 top_k: int = 1, alpha: float = 8., last_n_layers: int = 4,
                 init: str = "spectral"):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.adapters = inject_spectral_adapters(
            self.backbone, last_n_layers=last_n_layers, shared_rank=shared_rank,
            private_rank=private_rank, num_private=num_private, top_k=top_k,
            alpha=alpha, init=init,
        )
        config = backbone.config
        hidden = (config.text_config.hidden_size if hasattr(config, "text_config")
                  else config.hidden_size)
        self.mask_decoder = MaskDecoder(hidden, decoder_hidden).to(next(self.backbone.parameters()).device)
        self.mask_size = mask_size
        self.merge_size = int(config.vision_config.spatial_merge_size)
        self.image_token_id = int(config.image_token_id)

    @property
    def device(self):
        return next(self.backbone.parameters()).device

    def set_task(self, task: str) -> None:
        set_adapters_task(self.adapters, task)

    def _move_inputs(self, batch):
        return {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}

    def _conversation(self, task: str, prompt: str, *, answer: str | None = None):
        # Text goes BEFORE image so autoregressive image-token states can
        # attend to the region/task instruction (important for segmentation).
        user = {"role": "user", "content": [
            {"type": "text", "text": prompt}, {"type": "image"},
        ]}
        messages = [user]
        if answer is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        return messages

    def prepare_prompt(self, processor, task: str, prompt: str, image: Image.Image) -> dict:
        conversation = self._conversation(task, prompt)
        text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        return self._move_inputs(processor(text=[text], images=[image], return_tensors="pt"))

    def prepare_text_training(self, processor, task: str, prompt: str,
                              answer: str, image: Image.Image) -> tuple[dict, Tensor]:
        prefix = processor.apply_chat_template(
            self._conversation(task, prompt), tokenize=False, add_generation_prompt=True)
        full = processor.apply_chat_template(
            self._conversation(task, prompt, answer=answer), tokenize=False,
            add_generation_prompt=False)
        # Qwen expands <|image_pad|> to a variable number of visual tokens.
        # Processing the prefix independently ensures the correct label boundary.
        pref = processor(text=[prefix], images=[image], return_tensors="pt")
        batch = processor(text=[full], images=[image], return_tensors="pt")
        boundary = pref["input_ids"].shape[1]
        if not torch.equal(batch["input_ids"][0, :boundary], pref["input_ids"][0]):
            raise RuntimeError("Prompt is not an exact token prefix; check tokenizer/chat template")
        labels = batch["input_ids"].clone()
        labels[:, :boundary] = -100
        labels[labels == self.image_token_id] = -100
        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = -100
        if (labels != -100).sum() == 0:
            raise RuntimeError("No supervised assistant tokens found")
        return self._move_inputs(batch), labels.to(self.device)

    def _spatial_tokens(self, input_ids: Tensor, image_grid_thw: Tensor, hidden: Tensor):
        if input_ids.shape[0] != 1 or image_grid_thw.shape[0] != 1:
            raise ValueError("Prototype supports a single still image per forward pass")
        t, h, w = (int(z) for z in image_grid_thw[0].tolist())
        if t != 1 or h % self.merge_size or w % self.merge_size:
            raise ValueError(f"Expected one image with grid divisible by {self.merge_size}, got {(t,h,w)}")
        fh, fw = h // self.merge_size, w // self.merge_size
        visual = hidden[0][input_ids[0] == self.image_token_id]
        if visual.shape[0] != fh * fw:
            raise RuntimeError(f"Visual tokens: got {visual.shape[0]}, expected {fh * fw}; grid={h}x{w}")
        return visual, fh, fw

    def segmentation(self, processor, image: Image.Image, prompt: str,
                     target: Tensor | None = None) -> dict:
        self.set_task("seg")
        inputs = self.prepare_prompt(processor, "seg", prompt, image)
        # Run the bare backbone to avoid the expensive vocabulary projection.
        outputs = self.backbone.model(**inputs, use_cache=False, return_dict=True)
        tokens, fh, fw = self._spatial_tokens(inputs["input_ids"], inputs["image_grid_thw"], outputs.last_hidden_state)
        logits = self.mask_decoder(tokens, fh, fw, self.mask_size)
        res = {"mask_logits": logits, "routes": route_snapshot(self.adapters)}
        if target is not None:
            res["loss"] = dice_bce_loss(logits, target.to(self.device))
        return res

    def text_task(self, processor, image: Image.Image, task: str,
                  prompt: str, answer: str) -> dict:
        if task == "seg":
            raise ValueError("For masks, use segmentation()")
        self.set_task(task)
        inputs, labels = self.prepare_text_training(processor, task, prompt, answer, image)
        outputs = self.backbone(**inputs, labels=labels, use_cache=False, return_dict=True)
        return {"loss": outputs.loss, "routes": route_snapshot(self.adapters)}

    def load_adapter(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        selected = checkpoint["state"]
        current = self.state_dict()
        unexpected = set(selected) - set(current)
        if unexpected:
            raise RuntimeError(f"Unexpected adapter keys: {sorted(unexpected)[:5]}")
        for k, v in selected.items():
            if current[k].shape != v.shape:
                raise RuntimeError(f"Shape mismatch for {k}: {current[k].shape} != {v.shape}")
        self.load_state_dict(selected, strict=False)

    def save_adapter(self, path: str, metadata: dict | None = None) -> None:
        torch.save({"state": trainable_state(self), "metadata": metadata or {}}, path)


def load_grounded_model(model_id: str, device: str, dtype: str = "bf16", **model_kwargs):
    # Lazy import allows the CPU unit tests to run without Transformers.
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch_dtype, attn_implementation="sdpa",
        device_map=None, low_cpu_mem_usage=True,
    ).to(device)
    backbone.config.use_cache = False
    processor = AutoProcessor.from_pretrained(model_id)
    model = GroundedSpectralQwen(backbone, **model_kwargs)
    return model, processor
