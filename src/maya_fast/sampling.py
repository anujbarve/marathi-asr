from __future__ import annotations

import numpy as np


def gumbel(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """
    Produce float32 Gumbel noise: -log(-log(U)) with U in [1e-10, 1 - 1e-7].
    """
    u = rng.uniform(low=1e-10, high=1.0 - 1e-7, size=shape).astype(np.float32)
    return (-np.log(-np.log(u))).astype(np.float32)


def topk_gumbel_argmax(
    logits: np.ndarray,
    temp: float = 0.7,
    top_k: int = 64,
    noise: np.ndarray | None = None,
) -> int:
    """
    Numpy reference of the graph's rule:
    argmax(where(rank < top_k, logits / temp + noise, -inf))
    """
    logits = np.asarray(logits, dtype=np.float32)
    if noise is None:
        noise = np.zeros_like(logits)
    else:
        noise = np.asarray(noise, dtype=np.float32)

    k = min(top_k, logits.shape[-1])
    topk_idx = np.argpartition(logits, -k)[-k:]
    masked_perturbed = np.full_like(logits, -np.inf)
    masked_perturbed[topk_idx] = logits[topk_idx] / temp + noise[topk_idx]
    return int(np.argmax(masked_perturbed))
