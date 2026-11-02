import torch
from torch import nn

from sgmoe.adapters import SpectralRouterLinear, inject_spectral_adapters, trainable_state
from sgmoe.model import MaskDecoder
from sgmoe.optimization import (apply_shared_pcgrad, dice_bce_loss, overlap_penalty,
                                private_parameters, shared_parameters, task_grads,
                                update_conflict_ema)


def test_base_equivalence_and_spectral_init():
    torch.manual_seed(2)
    linear = nn.Linear(12, 10)
    layer = SpectralRouterLinear(linear, shared_rank=2, private_rank=2,
                                num_private=3, init="spectral", top_k=2)
    x = torch.randn(2, 4, 12)
    assert torch.allclose(layer(x), linear(x), atol=1e-6)
    # SVD right singular bases should start approximately orthonormal.
    a = torch.cat([layer.shared_A, layer.private_A.flatten(0, 1)], dim=0)
    assert torch.allclose(a @ a.T, torch.eye(8), atol=5e-3)


def test_training_masks_and_task_routing():
    torch.manual_seed(0)
    layer = SpectralRouterLinear(nn.Linear(8, 8), shared_rank=2,
                                private_rank=2, num_private=2, top_k=1, init="random")
    head = MaskDecoder(8, hidden=16)
    x = torch.randn(1, 16, 8)
    layer.set_task("seg")
    y = layer(x)
    logits = head(y[0], 4, 4, 32)
    loss = dice_bce_loss(logits, torch.ones_like(logits))
    loss.backward()
    assert layer.shared_B.grad.abs().sum() > 0
    assert layer.private_B.grad.abs().sum() > 0
    assert layer.gate.weight.grad is not None


def test_conflict_ema_penalty():
    layer = SpectralRouterLinear(nn.Linear(8, 8), shared_rank=2,
                                private_rank=2, num_private=2, top_k=2, init="random")
    mods = {"layer": layer}
    x = torch.randn(1, 5, 8)
    layer.set_task("seg")
    loss_a = layer(x).square().mean()
    route_a = {"layer": layer.last_route_probs}
    layer.set_task("vqa")
    loss_b = (-layer(x)).mean()
    route_b = {"layer": layer.last_route_probs}
    params = private_parameters(mods)
    ga, gb = task_grads(loss_a, params), task_grads(loss_b, params)
    update_conflict_ema(mods, ga, gb, decay=0.)
    penalty = overlap_penalty(mods, route_a, route_b)
    assert penalty.ndim == 0 and torch.isfinite(penalty)


def test_pcgrad_cancels_opposing_gradients():
    p = nn.Parameter(torch.tensor([1.0, 1.0]))
    ga = [torch.tensor([1.0, 0.0])]
    gb = [torch.tensor([-1.0, 1.0])]
    p.grad = ga[0] + gb[0]
    cosine = apply_shared_pcgrad([p], ga, gb)
    assert cosine < 0
    assert not torch.allclose(p.grad, ga[0] + gb[0])
    assert torch.isfinite(p.grad).all()


def test_injection_on_fake_qwen_layers():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(8, 8)
            self.v_proj = nn.Linear(8, 8)
    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block() for _ in range(3)])
    fake = Fake()
    adapters = inject_spectral_adapters(fake, last_n_layers=1, shared_rank=1,
                                        private_rank=1, num_private=2, init="random")
    assert len(adapters) == 2
    assert isinstance(fake.layers[-1].q_proj, SpectralRouterLinear)
    assert isinstance(fake.layers[0].q_proj, nn.Linear)
    assert all(k.startswith("layers.2.") for k in adapters)
    saved = trainable_state(fake)
    assert any(k.endswith("conflict_ema") for k in saved)
