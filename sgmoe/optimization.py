"""Conflict-aware multitask regularization and shared-parameter PCGrad."""
from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .adapters import SpectralRouterLinear


def dice_bce_loss(logits: Tensor, targets: Tensor, smooth: float = 1.0) -> Tensor:
    if logits.shape != targets.shape:
        raise ValueError(f"Shape mismatch logits={logits.shape}, targets={targets.shape}")
    bce = F.binary_cross_entropy_with_logits(logits.float(), targets.float())
    p = logits.float().sigmoid().flatten(1)
    t = targets.float().flatten(1)
    dice = 1 - ((2 * (p * t).sum(-1) + smooth) / (p.sum(-1) + t.sum(-1) + smooth)).mean()
    return bce + dice


def shared_parameters(adapters: Mapping[str, SpectralRouterLinear]) -> list[nn.Parameter]:
    return [p for m in adapters.values() for p in (m.shared_A, m.shared_B)]


def private_parameters(adapters: Mapping[str, SpectralRouterLinear]) -> list[nn.Parameter]:
    return [p for m in adapters.values() for p in (m.private_A, m.private_B)]


def task_grads(loss: Tensor, parameters: Sequence[nn.Parameter]) -> list[Tensor | None]:
    return list(torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True))


def _dot(a: Tensor | None, b: Tensor | None) -> Tensor:
    if a is None or b is None:
        return torch.tensor(0.0)
    return (a.detach().float() * b.detach().float()).sum()


@torch.no_grad()
def update_conflict_ema(
    adapters: Mapping[str, SpectralRouterLinear],
    task_a_private_grads: Sequence[Tensor | None],
    task_b_private_grads: Sequence[Tensor | None],
    *,
    decay: float = 0.9,
) -> dict[str, float]:
    """Estimate per-expert negative gradient cosine, smoothly over steps.

    Input gradients must be ordered: private_A, private_B for each adapter.
    """
    if len(task_a_private_grads) != 2 * len(adapters):
        raise ValueError("Gradient list must have two entries per adapter")
    diagnostics = {}
    for i, (name, adapter) in enumerate(adapters.items()):
        ga = task_a_private_grads[2 * i:2 * i + 2]
        gb = task_b_private_grads[2 * i:2 * i + 2]
        severities = torch.zeros_like(adapter.conflict_ema)
        for e in range(adapter.num_private):
            va = torch.cat([g[e].detach().flatten().float() if g is not None else
                            torch.zeros_like(p[e]).flatten().float()
                            for g, p in zip(ga, (adapter.private_A, adapter.private_B))])
            vb = torch.cat([g[e].detach().flatten().float() if g is not None else
                            torch.zeros_like(p[e]).flatten().float()
                            for g, p in zip(gb, (adapter.private_A, adapter.private_B))])
            na, nb = torch.linalg.vector_norm(va), torch.linalg.vector_norm(vb)
            if na > 1e-12 and nb > 1e-12:
                cosine = torch.dot(va, vb) / (na * nb + 1e-12)
                severities[e] = (-cosine).clamp(min=0, max=1)
        adapter.conflict_ema.mul_(decay).add_(severities * (1 - decay))
        diagnostics[name] = adapter.conflict_ema.mean().item()
    return diagnostics


def overlap_penalty(
    adapters: Mapping[str, SpectralRouterLinear],
    route_a: Mapping[str, Tensor],
    route_b: Mapping[str, Tensor],
) -> Tensor:
    terms = []
    for name, module in adapters.items():
        if name in route_a and name in route_b:
            # Gradient conflict is measured without passing grads through it.
            terms.append((module.conflict_ema.detach() * route_a[name] * route_b[name]).sum())
    if not terms:
        raise RuntimeError("No router probabilities captured. Check forward passes.")
    return torch.stack(terms).mean()


@torch.no_grad()
def apply_shared_pcgrad(
    shared: Sequence[nn.Parameter],
    grad_a: Sequence[Tensor | None],
    grad_b: Sequence[Tensor | None],
    weight_a: float = 1.0,
    weight_b: float = 1.0,
) -> float:
    """Replace task-loss gradients on shared params with symmetric PCGrad.

    Preserves any gradients already in p.grad from router regularization.
    PCGrad applies only to shared LoRA tensors, not the private ones.
    """
    if not (len(shared) == len(grad_a) == len(grad_b)):
        raise ValueError("Mismatched parameter / gradient counts")
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(grad_a, grad_b):
        if a is None or b is None:
            continue
        aa, bb = a.detach().float(), b.detach().float()
        dot += (aa * bb).sum().item()
        norm_a += aa.square().sum().item()
        norm_b += bb.square().sum().item()
    cosine = dot / (max(norm_a * norm_b, 1e-20) ** 0.5)
    for p, a, b in zip(shared, grad_a, grad_b):
        if a is None and b is None:
            continue
        aa = torch.zeros_like(p) if a is None else a.detach()
        bb = torch.zeros_like(p) if b is None else b.detach()
        orig = weight_a * aa + weight_b * bb
        if dot < 0 and norm_a > 0 and norm_b > 0:
            projected_a = aa - (dot / (norm_b + 1e-12)) * bb
            projected_b = bb - (dot / (norm_a + 1e-12)) * aa
            updated = weight_a * projected_a + weight_b * projected_b
        else:
            updated = orig
        # Subtract the old task component but retain other regularizer gradients.
        if p.grad is None:
            p.grad = (updated - orig).clone()
        else:
            p.grad.add_(updated - orig)
    return cosine
