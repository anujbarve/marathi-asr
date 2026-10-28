"""Reference backbone that streams one decoder layer at a time from the checkpoint.

The upstream Qwen3 backbone is 4.7 GiB in bf16. Holding it resident breaks the 4 GB RAM limit, so this runs the
same transformers Qwen3DecoderLayer / Qwen3RMSNorm / Qwen3RotaryEmbedding code, one layer at a time: read the layer's
weights from safetensors, upcast to fp32, run it, move on. The KV cache is a transformers DynamicCache, so the
prefill / one-token-per-step path is the upstream one. Costs: one read of ~2.6 GiB of layer weights per call of
`forward` (per generated token), so it is slow; it is used only for the reference corpus.
"""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3RotaryEmbedding

LAYER_PREFIX = "backbone.lm.model.layers.{i}."
NORM_KEY = "backbone.lm.model.norm.weight"


class _Weights:
    """Name -> tensor lookup over the checkpoint shards, reading one tensor at a time."""

    def __init__(self, shard_files: list[str], model_prefix: str = "model.") -> None:
        self.index: dict[str, str] = {}
        for f in shard_files:
            with safe_open(f, "pt") as fh:
                for k in fh.keys():
                    self.index[k] = f
        self.model_prefix = model_prefix

    def get(self, name: str) -> torch.Tensor:
        """name: checkpoint key without the leading `model.` (e.g. backbone.lm.model.norm.weight)."""
        key = name if name in self.index else self.model_prefix + name
        with safe_open(self.index[key], "pt") as fh:
            return fh.get_tensor(key)


class StreamedBackbone:
    def __init__(self, backbone_cfg_path: str, shard_files: list[str], n_layers: int | None = None) -> None:
        self.cfg = AutoConfig.from_pretrained(backbone_cfg_path)
        self.cfg._attn_implementation = "sdpa"  # sdpa applies the causal mask when mask is None and T > 1
        self.n_layers = int(n_layers if n_layers is not None else self.cfg.num_hidden_layers)
        self.weights = _Weights(shard_files)
        self.rotary = Qwen3RotaryEmbedding(self.cfg)
        self.norm = Qwen3RMSNorm(self.cfg.hidden_size, eps=self.cfg.rms_norm_eps)
        self.norm.weight.data.copy_(self.weights.get(NORM_KEY).float())
        self.layer = Qwen3DecoderLayer(self.cfg, layer_idx=0).eval()
        self.cache = DynamicCache(config=self.cfg)
        self.past_len = 0

    def reset(self) -> None:
        self.cache = DynamicCache(config=self.cfg)
        self.past_len = 0

    def _load_layer(self, i: int) -> None:
        prefix = LAYER_PREFIX.format(i=i)
        state = {}
        for name, _ in self.layer.named_parameters():
            state[name] = self.weights.get(prefix + name).float()
        with torch.no_grad():
            for name, param in self.layer.named_parameters():
                param.data.copy_(state[name])
        del state
        self.layer.self_attn.layer_idx = i

    @torch.inference_mode()
    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        """embeds: (1, T, hidden) fp32, the next T positions after the cached ones. Returns (1, T, hidden) after the
        final norm (the upstream `last_hidden_state`)."""
        t = embeds.shape[1]
        position_ids = torch.arange(self.past_len, self.past_len + t)[None]
        position_embeddings = self.rotary(embeds, position_ids)
        h = embeds
        for i in range(self.n_layers):
            self._load_layer(i)
            h = self.layer(
                h,
                attention_mask=None,  # sdpa: causal when T > 1 and no cache, none needed for one-token steps
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_values=self.cache,
                use_cache=True,
            )
        self.past_len += t
        return self.norm(h)


def shard_files(ckpt_dir: str | Path) -> list[str]:
    return sorted(str(p) for p in Path(ckpt_dir).glob("*.safetensors"))
