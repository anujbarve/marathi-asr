"""Single depth graph: levels 1..15, teacher-forced, weights stored in fp16 and computed in fp32.

Same computation as tts/model/depth_v2.py for levels 1..15: each level's prefix (_prefix), the level's weight group
(GROUP_OF_LEVEL: groups 0..7 private, levels 9..15 share group 8), the 4 shared cross-attention blocks, LayerNorm and
the level head. Group 8 is one set of weights used by seven levels, so it is stored once in the graph.

Storage: every weight is an fp16 initializer; each use is a Cast to fp32 (parametrization in torch, Cast node in ONNX).
Built one tensor at a time from the checkpoint, so the fp32 copy is never resident.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tts.model.depth_v2 import D, GROUP_OF_LEVEL, LAYERS, RANK, _CrossBlock, _SelfBlock
from tts.tokens import N_CODES, N_Q

LEVELS = 15  # serving q_serve = 16 -> levels 1..15
TEMP = 0.7
TOP_K = 64
N_PRIVATE_GROUPS = 9  # groups 0..8 are used by levels 1..15


class _UpcastFp32(nn.Module):
    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return w.float()


class _Fp16Embedding(nn.Embedding):
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(idx, self.weight).float()


class DepthAll(nn.Module):
    def __init__(self, n_codes: int = N_CODES, d_backbone: int = 2048) -> None:
        super().__init__()
        self.n_codes = n_codes
        self.in_proj = nn.Linear(d_backbone, D, bias=False)
        self.code_lo = nn.Embedding((N_Q - 1) * n_codes, RANK)
        self.code_up = nn.Linear(RANK, D, bias=False)
        self.level_emb = nn.Embedding(N_Q, D)
        self.groups = nn.ModuleList(
            nn.ModuleList(_SelfBlock() for _ in range(LAYERS)) for _ in range(N_PRIVATE_GROUPS))
        self.cross = nn.ModuleList(_CrossBlock() for _ in range(LAYERS))
        self.norm = nn.LayerNorm(D)
        self.heads = nn.ModuleList(nn.Linear(D, n_codes, bias=False) for _ in range(LEVELS))

    def _prefix(self, h: torch.Tensor, codes: torch.Tensor, level: int) -> torch.Tensor:
        ks = torch.arange(level, device=h.device)
        flat = codes[:, :level] + ks * self.n_codes
        e = self.code_up(self.code_lo(flat)) + self.level_emb(ks + 1)[None]
        return torch.cat([self.in_proj(h)[:, None] + self.level_emb.weight[:1], e], 1)

    def _level_logits(self, h: torch.Tensor, codes_prev: torch.Tensor, level: int, mem: torch.Tensor) -> torch.Tensor:
        """Logits of one level given the codes of the levels before it (codes_prev has `level` columns)."""
        x = self._prefix(h, codes_prev, level)
        T = x.shape[1]
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()[None, None]
        group = self.groups[GROUP_OF_LEVEL[level]]
        for li in range(LAYERS):
            x = group[li](x, mask)
            x = self.cross[li](x, mem, None)
        return self.heads[level - 1](self.norm(x[:, -1]))

    def forward(self, h: torch.Tensor, codes: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        """Teacher-forced: h (B, 2048); codes (B, Q_SERVE=16) int64; mem (B, 1, 1024). Returns logits (B, 15, 2048)."""
        return torch.stack([self._level_logits(h, codes[:, :level], level, mem) for level in range(1, LEVELS + 1)], 1)

    def sample(self, h: torch.Tensor, cb0: torch.Tensor, mem: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Sampled frame. h (B, 2048); cb0 (B,) int64; mem (B, 1, 1024); noise (B, 15, 2048) Gumbel.
        Level L: argmax(where(rank < top_k, logits / TEMP + noise, -inf)) over logits of the previous levels' codes.
        Returns codes (B, 16) int64, cb0 first. Gumbel-max over the top-k softmax, same distribution as upstream
        depth_generate (torch.multinomial over the top-k softmax of logits / TEMP)."""
        codes = [cb0]
        for level in range(1, LEVELS + 1):
            prev = torch.stack(codes, 1)  # (B, level)
            scaled = self._level_logits(h, prev, level, mem) / TEMP
            kth = scaled.topk(TOP_K, -1).values[:, -1:]
            masked = scaled.masked_fill(scaled < kth, float("-inf"))
            codes.append((masked + noise[:, level - 1]).argmax(-1))
        return torch.stack(codes, 1)


class DepthSample(nn.Module):
    """Export wrapper: the sampled-frame graph (inputs h, cb0, mem, noise; output codes)."""

    def __init__(self, depth: DepthAll) -> None:
        super().__init__()
        self.depth = depth

    def forward(self, h, cb0, mem, noise):
        return self.depth.sample(h, cb0, mem, noise)


def load_fp16(model: DepthAll, get) -> None:
    """Load every parameter from `get(ckpt_name) -> tensor` as fp16, and mark it for an fp32 upcast on use.
    `get` returns the bf16/fp32 checkpoint tensor; mapping from module names to checkpoint names is in export."""
    names = {}
    for name, p in model.named_parameters():
        names[name] = get(name)
    for name, t in names.items():
        mod_name, _, leaf = name.rpartition(".")
        mod = model.get_submodule(mod_name)
        mod._parameters[leaf] = nn.Parameter(t.to(torch.float16), requires_grad=False)
    for mod_name, mod in list(model.named_modules()):
        for leaf, p in list(mod._parameters.items()):
            if p is None or p.dtype != torch.float16:
                continue
            if isinstance(mod, nn.Embedding):
                mod.__class__ = _Fp16Embedding
                continue
            torch.nn.utils.parametrize.register_parametrization(mod, leaf, _UpcastFp32(), unsafe=True)
