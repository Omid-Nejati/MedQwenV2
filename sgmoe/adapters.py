"""SVD-initialized sparse LoRA experts with a mandatory shared adapter.

SVD is used for *initialization*, not as an orthogonality guarantee after
training. Conflict handling occurs in sgmoe.optimization.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

TASKS = ("seg", "vqa", "report", "classification")
TASK_IDS = {name: i for i, name in enumerate(TASKS)}


class SpectralRouterLinear(nn.Module):
    """Frozen base linear + shared LoRA + sparse task-conditioned private LoRAs.

    Private selection is *per input sequence*, not per token. Gate outputs use
    straight-through top-k support (gradients reach the selected logits).
    The base model output is preserved exactly at initialization: B = 0.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        shared_rank: int = 4,
        private_rank: int = 4,
        num_private: int = 3,
        top_k: int = 1,
        alpha: float = 8.0,
        init: str = "spectral",
        num_tasks: int = len(TASKS),
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("Expected torch.nn.Linear")
        if shared_rank <= 0 or private_rank <= 0:
            raise ValueError("Ranks must be positive")
        if not 1 <= top_k <= num_private:
            raise ValueError("top_k must be between 1 and num_private")
        if init not in ("spectral", "random"):
            raise ValueError("init must be 'spectral' or 'random'")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        din, dout = base.in_features, base.out_features
        self.num_private = num_private
        self.top_k = top_k
        self.scaling = alpha / float(private_rank)
        self.shared_scaling = alpha / float(shared_rank)
        self.task_id = 0
        device = base.weight.device
        self.shared_A = nn.Parameter(torch.empty(shared_rank, din, device=device, dtype=torch.float32))
        self.shared_B = nn.Parameter(torch.zeros(dout, shared_rank, device=device, dtype=torch.float32))
        self.private_A = nn.Parameter(torch.empty(num_private, private_rank, din, device=device, dtype=torch.float32))
        self.private_B = nn.Parameter(torch.zeros(num_private, dout, private_rank, device=device, dtype=torch.float32))
        self.task_embedding = nn.Embedding(num_tasks, din, device=device, dtype=torch.float32)
        self.gate = nn.Linear(din, num_private, bias=True, device=device, dtype=torch.float32)
        nn.init.normal_(self.task_embedding.weight, std=0.02)
        nn.init.zeros_(self.gate.bias)
        nn.init.normal_(self.gate.weight, std=0.002)
        self.register_buffer("conflict_ema", torch.zeros(num_private, device=device, dtype=torch.float32))
        self.last_route_probs: Tensor | None = None

        if init == "spectral":
            self._spectral_initialize(base.weight.detach())
        else:
            nn.init.kaiming_uniform_(self.shared_A, a=math.sqrt(5))
            for i in range(num_private):
                nn.init.kaiming_uniform_(self.private_A[i], a=math.sqrt(5))

    @torch.no_grad()
    def _spectral_initialize(self, weight: Tensor) -> None:
        """Initialize distinct right singular directions via randomized SVD.

        This is a low-rank approximation, not a full dense SVD. If a layer is
        narrower than the requested total rank, spectral directions wrap.
        """
        total = self.shared_A.shape[0] + self.num_private * self.private_A.shape[1]
        min_dim = min(weight.shape)
        q = min(min_dim, max(2, total + 4))
        # Randomized SVD on the same device as the layer. The result is converted
        # to float32 and B remains zero to avoid changing pretrained outputs.
        _, _, right = torch.svd_lowrank(weight.float(), q=q, niter=2)
        vectors = right.T.float()  # [q, in_features]
        if total > q:
            vectors = vectors[torch.arange(total, device=weight.device) % q]
        else:
            vectors = vectors[:total]
        s = self.shared_A.shape[0]
        r = self.private_A.shape[1]
        self.shared_A.copy_(vectors[:s])
        self.private_A.copy_(vectors[s:s + self.num_private * r].reshape(self.num_private, r, -1))

    def set_task(self, task: str) -> None:
        if task not in TASK_IDS:
            raise ValueError(f"Unknown task {task}; expected one of {TASKS}")
        self.task_id = TASK_IDS[task]

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base(x)
        squeezed = x.ndim == 2
        if squeezed:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise ValueError(f"Expected [B,S,D] or [S,D], got {x.shape}")
        xf = x.float()
        task = torch.tensor(self.task_id, device=xf.device)
        context = xf.mean(dim=1) + self.task_embedding(task)
        logits = self.gate(context) / math.sqrt(float(self.gate.in_features))
        probs = logits.softmax(dim=-1)
        self.last_route_probs = probs.mean(dim=0)
        _, selected = logits.topk(self.top_k, dim=-1)
        sparse = torch.full_like(logits, float("-inf"))
        sparse.scatter_(-1, selected, logits.gather(-1, selected))
        hard_weights = sparse.softmax(dim=-1)
        # Forward: actual Top-K sparse routing. Backward: dense-softmax
        # straight-through surrogate, so the router can learn even for K=1.
        weights = hard_weights.detach() + probs - probs.detach()

        shared = F.linear(F.linear(xf, self.shared_A), self.shared_B) * self.shared_scaling
        private_out = torch.zeros_like(shared)
        # This loop is deliberately explicit and easy to modify for new routing.
        for i in range(self.num_private):
            projected = F.linear(F.linear(xf, self.private_A[i]), self.private_B[i])
            private_out = private_out + weights[:, i, None, None] * projected
        adapter = shared + private_out * self.scaling
        if squeezed:
            adapter = adapter.squeeze(0)
        return base_out + adapter.to(base_out.dtype)


def inject_spectral_adapters(
    model: nn.Module,
    *,
    targets: tuple[str, ...] = ("q_proj", "v_proj"),
    last_n_layers: int = 4,
    shared_rank: int = 4,
    private_rank: int = 4,
    num_private: int = 3,
    top_k: int = 1,
    alpha: float = 8.0,
    init: str = "spectral",
) -> Dict[str, SpectralRouterLinear]:
    """Replace only language-layer projections (never the vision encoder)."""
    import re

    found: dict[str, tuple[nn.Module, str, nn.Linear, int]] = {}
    for name, module in model.named_modules():
        # compatible with qwen2_5_vl model.model.language_model.layers.X...
        # and older model.model.layers.X... layout
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        if not match or not any(name.endswith("." + t) for t in targets):
            continue
        if not isinstance(module, nn.Linear):
            continue
        parent_path, leaf = name.rsplit(".", 1)
        parent = model.get_submodule(parent_path)
        found[name] = (parent, leaf, module, int(match.group(1)))
    if not found:
        raise RuntimeError("No target projections found. Inspect backbone named_modules().")
    layers = sorted({record[3] for record in found.values()})
    keep = set(layers[-last_n_layers:]) if last_n_layers > 0 else set(layers)
    wrapped: Dict[str, SpectralRouterLinear] = {}
    for name, (parent, leaf, original, layer_num) in found.items():
        if layer_num not in keep:
            continue
        adapter = SpectralRouterLinear(
            original, shared_rank=shared_rank, private_rank=private_rank,
            num_private=num_private, top_k=top_k, alpha=alpha, init=init,
        )
        setattr(parent, leaf, adapter)
        wrapped[name] = adapter
    return wrapped


def set_adapters_task(adapters: Mapping[str, SpectralRouterLinear], task: str) -> None:
    for module in adapters.values():
        module.set_task(task)


def route_snapshot(adapters: Mapping[str, SpectralRouterLinear]) -> dict[str, Tensor]:
    """Keep the differentiable gate probabilities from the last forward."""
    return {name: module.last_route_probs for name, module in adapters.items()
            if module.last_route_probs is not None}


def trainable_state(model: nn.Module) -> dict[str, Tensor]:
    """Only save trainable tensors and conflict EMA buffers, not the frozen LLM."""
    selected = {name for name, p in model.named_parameters() if p.requires_grad}
    selected.update(name for name, _ in model.named_buffers() if name.endswith("conflict_ema"))
    state = model.state_dict()
    return {name: state[name].detach().cpu() for name in sorted(selected)}
