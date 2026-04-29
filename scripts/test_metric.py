"""Numeric correctness test for src.metric.RiemannianMetric.

Stubs Score_Distillation with linear score functions whose Jacobians are known
in closed form. Verifies that:

1. The 'jvp' backend computes ‖J v‖² + G_ε(x)·‖v‖² to fp32 tolerance against
   the closed form, for both a diagonal and a channel-mixing score.
2. The 'fd' backend agrees with 'jvp' to within O(ε²) at ε=1e-3.
3. Chunking gives the same answer as no chunking.
4. With a zero score, kinetic energy reduces to G_ε(x)·‖v‖².

CPU-only; no GPU or SD UNet required. Run with:

    python scripts/test_metric.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metric import AnnulusStats, RiemannianMetric


# -----------------------------------------------------------------------------
# Stubs that quack like Score_Distillation.grad_compute(latent, embed_cond).
# -----------------------------------------------------------------------------

class DiagStubScore:
    """s(x) = a ⊙ x  (elementwise). Then J = diag(a) and J v = a ⊙ v."""

    def __init__(self, a: torch.Tensor):
        self.a = a  # broadcastable to latent shape

    def grad_compute(self, latent, embed_cond):
        return self.a * latent


class ChannelMixStubScore:
    """s(x)_c = Σ_{c'} M[c, c'] x_{c'}.  J v applies the same channel mixing."""

    def __init__(self, M: torch.Tensor):
        self.M = M  # (C, C)

    def grad_compute(self, latent, embed_cond):
        return torch.einsum('ij,njhw->nihw', self.M, latent)


def _stub_embed():
    """Tiny dummy embedding (1, T, D) — stubs ignore it but the constructor checks shape."""
    return torch.zeros(1, 4, 8)


def _annulus_closed_form(x, mu, sigma):
    return ((x.flatten(2).norm(p=2, dim=-1) - mu) / sigma).pow(2)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_jvp_diagonal():
    torch.manual_seed(0)
    a = 0.5 + 0.1 * torch.randn(1, 4, 64, 64)  # bounded scaling
    score = DiagStubScore(a)
    annulus = AnnulusStats(mu=125.0, sigma=2.4)
    metric = RiemannianMetric(score, annulus, _stub_embed(), jvp_backend='jvp')

    x = torch.randn(2, 3, 4, 64, 64) * 100
    v = torch.randn(2, 3, 4, 64, 64)
    ke = metric.kinetic(x, v)

    Jv = a * v
    expected = (Jv.flatten(2).pow(2).sum(-1)
                + _annulus_closed_form(x, 125.0, 2.4) * v.flatten(2).pow(2).sum(-1))
    rel = (ke - expected).abs().max().item() / expected.abs().max().item()
    assert rel < 1e-5, f"diag/jvp rel_err = {rel:.2e}"
    print(f"OK  test_jvp_diagonal           rel_err={rel:.2e}")


def test_jvp_channel_mix():
    torch.manual_seed(1)
    M = 0.1 * torch.randn(4, 4)
    score = ChannelMixStubScore(M)
    annulus = AnnulusStats(mu=0.0, sigma=1.0)  # so G_ε(x) = ||x||²
    metric = RiemannianMetric(score, annulus, _stub_embed(), jvp_backend='jvp')

    x = torch.randn(2, 3, 4, 64, 64)
    v = torch.randn(2, 3, 4, 64, 64)
    ke = metric.kinetic(x, v)

    Jv = torch.einsum('ij,bkjhw->bkihw', M, v)
    expected = (Jv.flatten(2).pow(2).sum(-1)
                + _annulus_closed_form(x, 0.0, 1.0) * v.flatten(2).pow(2).sum(-1))
    rel = (ke - expected).abs().max().item() / expected.abs().max().item()
    assert rel < 1e-5, f"chmix/jvp rel_err = {rel:.2e}"
    print(f"OK  test_jvp_channel_mix        rel_err={rel:.2e}")


def test_fd_matches_jvp():
    torch.manual_seed(2)
    M = 0.1 * torch.randn(4, 4)
    score = ChannelMixStubScore(M)
    annulus = AnnulusStats(mu=125.0, sigma=2.4)
    embed = _stub_embed()

    metric_jvp = RiemannianMetric(score, annulus, embed, jvp_backend='jvp')
    metric_fd = RiemannianMetric(score, annulus, embed, jvp_backend='fd', fd_eps=1e-3)

    x = torch.randn(2, 3, 4, 64, 64) * 100
    v = torch.randn(2, 3, 4, 64, 64)
    ke_jvp = metric_jvp.kinetic(x, v)
    ke_fd = metric_fd.kinetic(x, v)
    rel = (ke_jvp - ke_fd).abs().max().item() / ke_jvp.abs().max().item()
    assert rel < 1e-3, f"jvp/fd rel_err = {rel:.2e}"
    print(f"OK  test_fd_matches_jvp         rel_err={rel:.2e}")


def test_chunking_matches_no_chunking():
    torch.manual_seed(3)
    M = 0.1 * torch.randn(4, 4)
    score = ChannelMixStubScore(M)
    annulus = AnnulusStats(mu=125.0, sigma=2.4)
    embed = _stub_embed()

    metric_full = RiemannianMetric(score, annulus, embed, jvp_backend='jvp', chunk_size=None)
    metric_chunk = RiemannianMetric(score, annulus, embed, jvp_backend='jvp', chunk_size=2)

    x = torch.randn(3, 5, 4, 64, 64)
    v = torch.randn(3, 5, 4, 64, 64)
    ke_full = metric_full.kinetic(x, v)
    ke_chunk = metric_chunk.kinetic(x, v)
    err = (ke_full - ke_chunk).abs().max().item()
    assert err < 1e-5, f"chunking abs_err = {err:.2e}"
    print(f"OK  test_chunking               abs_err={err:.2e}")


def test_zero_score_recovers_annulus():
    score = DiagStubScore(torch.zeros(1, 4, 64, 64))
    annulus = AnnulusStats(mu=125.0, sigma=2.4)
    metric = RiemannianMetric(score, annulus, _stub_embed(), jvp_backend='jvp')

    torch.manual_seed(4)
    x = torch.randn(2, 3, 4, 64, 64) * 100
    v = torch.randn(2, 3, 4, 64, 64)
    ke = metric.kinetic(x, v)
    expected = _annulus_closed_form(x, 125.0, 2.4) * v.flatten(2).pow(2).sum(-1)
    rel = (ke - expected).abs().max().item() / expected.abs().max().item()
    assert rel < 1e-5, f"zero-score rel_err = {rel:.2e}"
    print(f"OK  test_zero_score             rel_err={rel:.2e}")


def test_kinetic_differentiable_through_v():
    """Score-Jacobian and annulus terms should both push gradients back to v.

    Important because in training, v == ż_t depends on φ_{t,η}; if any term
    silently detaches we lose part of the loss.
    """
    torch.manual_seed(5)
    a = 0.5 * torch.ones(1, 4, 64, 64)
    score = DiagStubScore(a)
    annulus = AnnulusStats(mu=125.0, sigma=2.4)
    metric = RiemannianMetric(score, annulus, _stub_embed(), jvp_backend='jvp')

    x = torch.randn(2, 3, 4, 64, 64, requires_grad=True) * 100
    v = torch.randn(2, 3, 4, 64, 64, requires_grad=True)
    ke = metric.kinetic(x, v).sum()
    ke.backward()
    assert v.grad is not None and v.grad.abs().max() > 0, "no gradient flowed to v"
    assert x.grad is not None and x.grad.abs().max() > 0, "no gradient flowed to x"
    print(f"OK  test_kinetic_differentiable v.grad.max={v.grad.abs().max().item():.2e} "
          f"x.grad.max={x.grad.abs().max().item():.2e}")


if __name__ == "__main__":
    test_jvp_diagonal()
    test_jvp_channel_mix()
    test_fd_matches_jvp()
    test_chunking_matches_no_chunking()
    test_zero_score_recovers_annulus()
    test_kinetic_differentiable_through_v()
    print("\nAll metric tests passed.")
