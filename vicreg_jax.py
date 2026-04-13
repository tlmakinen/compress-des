"""
VICReg (Variance-Invariance-Covariance Regularization) in pure JAX.

Reference: Bardes, Ponce, LeCun, "VICReg: Variance-Invariance-Covariance
Regularization for Self-Supervised Learning", NeurIPS 2021.

Typical total loss:
    L = λ * invariance + μ * variance + ν * covariance

For compression / single-view settings, set λ=0 and use only variance + covariance
on batch embeddings z of shape (N, D).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Any

Array = Any


def vicreg_invariance_loss(z1: Array, z2: Array) -> Array:
    """Mean squared distance between two embedding views, same sample order.

    Args:
        z1, z2: (N, D) embeddings for N samples (e.g. two augmentations).

    Returns:
        Scalar: mean over batch of ||z1 - z2||^2.
    """
    return jnp.mean(jnp.sum((z1 - z2) ** 2, axis=-1))


def vicreg_variance_loss(
    z: Array,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> Array:
    """Hinge on per-dimension standard deviation (anti-collapse).

    For each dimension j: ReLU(gamma - std_j), averaged over j.

    Args:
        z: (N, D) batch of embeddings.
        gamma: target minimum standard deviation per dimension.
        eps: small constant inside sqrt for stability.

    Returns:
        Scalar variance regularizer.
    """
    # Population std over batch axis (ddof=0); match common implementations
    std = jnp.sqrt(jnp.var(z, axis=0, ddof=0) + eps)
    return jnp.mean(jax.nn.relu(gamma - std))


def vicreg_covariance_loss(z: Array, eps: float = 1e-8) -> Array:
    """Penalize off-diagonal entries of the batch covariance (decorrelation).

    Args:
        z: (N, D) batch of embeddings.
        eps: added to denominator for N=1 edge case.

    Returns:
        Scalar: (1/D) * sum_{i != j} Cov_ij^2 (Bardes et al. normalization).
    """
    z = z - jnp.mean(z, axis=0, keepdims=True)
    n = z.shape[0]
    denom = jnp.maximum(n - 1, 1.0) + eps
    cov = (z.T @ z) / denom  # (D, D)
    d = z.shape[1]
    cov_off = cov * (1.0 - jnp.eye(d, dtype=z.dtype))
    return jnp.sum(cov_off**2) / d


def vicreg_loss(
    z1: Array,
    z2: Array | None = None,
    *,
    sim_coeff: float = 25.0,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    gamma: float = 1.0,
    eps_var: float = 1e-4,
    eps_cov: float = 1e-8,
) -> Array:
    """Full VICReg loss with paper-style default coefficients.

    If z2 is None, the invariance term is omitted (single-view / no augmentation).

    Args:
        z1: (N, D) primary embeddings (required).
        z2: (N, D) second view; if None, invariance term is 0.
        sim_coeff, std_coeff, cov_coeff: weights (λ, μ, ν).
        gamma: variance hinge target std per dimension.

    Returns:
        Scalar total VICReg loss.
    """
    inv = (
        vicreg_invariance_loss(z1, z2)
        if z2 is not None
        else jnp.array(0.0, dtype=z1.dtype)
    )
    var = vicreg_variance_loss(z1, gamma=gamma, eps=eps_var)
    cov = vicreg_covariance_loss(z1, eps=eps_cov)
    return sim_coeff * inv + std_coeff * var + cov_coeff * cov


__all__ = [
    "vicreg_invariance_loss",
    "vicreg_variance_loss",
    "vicreg_covariance_loss",
    "vicreg_loss",
]
