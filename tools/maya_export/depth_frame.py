"""Per-level teacher-forced depth graph (one ONNX graph per level 1..15).

Each graph is the upstream DepthV2 computation for one level, with the same classes (_SelfBlock, _CrossBlock) and the
same prefix (_prefix) and last-position read-out (_run) as tts/model/depth_v2.py, mask None (the upstream kmask is
all True for named voices, so padding is never masked). It holds only the weights that level needs: its weight group
(4 self-attention blocks), the 4 shared cross-attention blocks, the code embeddings, the input projection and the
level's head. The 15 graphs are chained by the runtime; the group's weights are not shared between levels 9..15 here
(each level graph carries group 8), which costs bandwidth but keeps every part small.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from tts.model.depth_v2 import D, GROUP_OF_LEVEL, LAYERS, RANK, _CrossBlock, _SelfBlock
from tts.tokens import N_CODES, N_Q

Q_SERVE = 16  # serving codebooks per frame (cb0..cb15); levels 1..15 read columns 0..L-1 of these


class DepthLevel(nn.Module):
    def __init__(self, level: int, n_codes: int = N_CODES, d_backbone: int = 2048) -> None:
        super().__init__()
        if not 1 <= level <= N_Q - 1:
            raise ValueError(f"level {level} outside 1..{N_Q - 1}")
        self.level = level
        self.group = GROUP_OF_LEVEL[level]
        self.n_codes = n_codes
        self.in_proj = nn.Linear(d_backbone, D, bias=False)
        self.code_lo = nn.Embedding((N_Q - 1) * n_codes, RANK)
        self.code_up = nn.Linear(RANK, D, bias=False)
        self.level_emb = nn.Embedding(N_Q, D)
        self.trunk = nn.ModuleList(_SelfBlock() for _ in range(LAYERS))
        self.cross = nn.ModuleList(_CrossBlock() for _ in range(LAYERS))
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, n_codes, bias=False)

    def forward(self, h: torch.Tensor, codes: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        """h (B, 2048) fp32 backbone hidden; codes (B, Q_SERVE) int64 with columns cb0..cb_{L-1} used; mem (B, 1, 1024).
        Returns the level's logits (B, 2048)."""
        L = self.level
        ks = torch.arange(L, device=h.device)
        flat = codes[:, :L] + ks * self.n_codes
        e = self.code_up(self.code_lo(flat)) + self.level_emb(ks + 1)[None]
        x = torch.cat([self.in_proj(h)[:, None] + self.level_emb.weight[:1], e], 1)  # (B, L+1, D)
        T = x.shape[1]
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()[None, None]
        for li in range(LAYERS):
            x = self.trunk[li](x, mask)
            x = self.cross[li](x, mem, None)
        return self.head(self.norm(x[:, -1]))


def level_group(level: int) -> int:
    return GROUP_OF_LEVEL[level]
