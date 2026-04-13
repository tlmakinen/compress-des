"""
Standalone diagonal Gaussian mixture density network (MDN) in JAX / Flax.
No TensorFlow Probability — numerics match the usual MixtureSameFamily +
MultivariateNormalDiag setup used in des-hybrid ``network.new_epe_code.MDN``.

Public API
----------
- ``mixture_diag_log_prob``: pure JAX — log p(theta | logits, mu, sigma).
- ``MDNJax``: Flax module with the same parameterization as ``MDN`` there
  (MLP trunk, logits head, shared head for mus and raw scales, softplus scale).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import flax.linen as nn

Array = Any


def mixture_diag_log_prob(
    logits: Array,
    mu: Array,
    sigma: Array,
    theta: Array,
) -> Array:
    """Log-density of a Gaussian mixture with diagonal covariances.

    Args:
        logits: mixture logits, shape (K,) or (B, K).
        mu: component means, shape (K, D) or (B, K, D).
        sigma: component std devs (positive), same shape as mu.
        theta: evaluation point, shape (D,) or (B, D).

    Returns:
        Scalar log p(theta) if unbatched, else shape (B,).
    """
    single = theta.ndim == 1
    if single:
        logits = logits[None, ...]
        mu = mu[None, ...]
        sigma = sigma[None, ...]
        theta = theta[None, ...]

    # log pi_k
    log_w = logits - logsumexp(logits, axis=-1, keepdims=True)
    # log N(theta | mu_k, diag(sigma_k^2)) for each k
    inv_sig = 1.0 / sigma
    z = (theta[:, None, :] - mu) * inv_sig
    d = mu.shape[-1]
    log_comp = (
        -jnp.sum(jnp.log(sigma), axis=-1)
        -0.5 * jnp.sum(z * z, axis=-1)
        -0.5 * d * jnp.log(2.0 * jnp.pi)
    )
    out = logsumexp(log_w + log_comp, axis=-1)
    return out[0] if single else out


class MLP(nn.Module):
    """Small MLP matching ``new_epe_code.MLP``."""

    features: Sequence[int]
    act: Callable = nn.relu
    activate_final: bool = False

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for feat in self.features[:-1]:
            x = self.act(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        if self.activate_final:
            x = self.act(x)
        return x


class MDNJax(nn.Module):
    """MDN conditioned on embedding ``x``; predicts p(theta | x).

    Same layout as ``network.new_epe_code.MDN`` (TFP version):
    MLP -> logits (K) and (mu, raw_scale) with K*D*2 outputs, softplus(scale).
    Optional ``theta_star`` is subtracted from theta (default zeros).
    """

    hidden_channels: Sequence[int]
    n_components: int
    n_dimension: int
    act: Callable = nn.relu
    theta_star: Array | None = None

    def setup(self) -> None:
        self.net = MLP(
            self.hidden_channels, act=self.act, activate_final=True
        )
        self.logits_net = nn.Dense(self.n_components)
        self.mu_sigma_net = nn.Dense(self.n_components * self.n_dimension * 2)

    def __call__(self, x: Array, theta: Array) -> Array:
        ts = (
            jnp.zeros((self.n_dimension,))
            if self.theta_star is None
            else self.theta_star
        )
        theta = theta - ts

        h = self.net(x)
        logits = self.logits_net(h)
        mu_sigma = self.mu_sigma_net(h)
        mu_raw, sig_raw = jnp.split(mu_sigma, 2, axis=-1)
        k, d = self.n_components, self.n_dimension
        mu = mu_raw.reshape(k, d)
        sigma = nn.softplus(sig_raw.reshape(k, d))
        return mixture_diag_log_prob(logits, mu, sigma, theta)


__all__ = ["mixture_diag_log_prob", "MLP", "MDNJax"]
