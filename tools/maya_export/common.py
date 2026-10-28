from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Iterator
import torch
from safetensors import safe_open


def work_dir() -> Path:
    env = os.environ.get("MAYA_FAST_WORK")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "maya_fast"


def ckpt_paths() -> dict:
    from maya3 import hub
    return hub.resolve("maya-research/maya2-native")


def iter_ckpt_tensors(
    skip_prefixes: tuple[str, ...] = (),
    files: list[str] | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield one tensor at a time from checkpoint safetensors, stripping model. prefix.
    If files is None, uses ckpt_paths()["ckpt"].
    """
    from tts.model import load_s3 as L

    if files is None:
        paths = ckpt_paths()
        src = os.path.abspath(str(paths["ckpt"]))
        _, files, _ = L._safetensors_files(src)

    seen = set()
    for f in files:
        with safe_open(f, "pt") as fh:
            for k in fh.keys():
                name = k[len(L.MODEL_PREFIX):] if k.startswith(L.MODEL_PREFIX) else k
                if any(name.startswith(p) for p in skip_prefixes):
                    continue
                if name in seen:
                    continue
                seen.add(name)
                t = fh.get_tensor(k)
                yield name, t
                del t


def _stub_lm(path: str, dtype: torch.dtype, attn: str):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(path)
    cfg.num_hidden_layers = 0
    cfg.layer_types = []
    with torch.device("meta"):
        lm = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32, attn_implementation="sdpa")
    for m in lm.modules():
        if hasattr(m, "inv_freq") and hasattr(m, "rope_type"):
            from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
            fn = m.rope_init_fn if getattr(m, "rope_init_fn", None) is not None else (
                m.compute_default_rope_parameters if m.rope_type == "default" else ROPE_INIT_FUNCTIONS[m.rope_type])
            with torch.device("cpu"):
                inv, scale = fn(m.config, torch.device("cpu"))
            m.inv_freq = inv
            if hasattr(m, "original_inv_freq"):
                m.original_inv_freq = inv
            if hasattr(m, "attention_scaling"):
                m.attention_scaling = scale
    lm.tie_weights()
    return lm


def build_light_model():
    """Build light TTSModelV2 on meta device with 0 backbone decoder layers,
    and populate non-backbone layer weights via iter_ckpt_tensors."""
    import tts.model.backbone as BB
    import tts.model.load_s3 as L

    orig_stub = BB._lm_from_config
    try:
        BB._lm_from_config = _stub_lm
        paths = ckpt_paths()
        src = str(paths["ckpt"])
        model_cfg, cfg_path = L.resolve_model_cfg(src)
        model_cfg["attn"] = "sdpa"

        anchor_cfg = model_cfg.get("speaker_anchor")
        depth_grouped = bool(model_cfg.get("depth_grouped", True))

        with torch.device("meta"):
            model = L.build_eval_model(
                model_cfg,
                anchor_cfg=anchor_cfg,
                amortize_frac=1.0,
                depth_grouped=depth_grouped,
            )

        mods = dict(model.named_modules())
        for name, t in iter_ckpt_tensors(skip_prefixes=("backbone.lm.model.layers.",)):
            parent_name, _, leaf = name.rpartition(".")
            parent = mods.get(parent_name)
            if parent is None or (leaf not in parent._parameters and leaf not in parent._buffers):
                continue
            is_param = leaf in parent._parameters
            t_float = t.float()
            if is_param:
                parent._parameters[leaf] = torch.nn.Parameter(t_float, requires_grad=False)
            else:
                parent._buffers[leaf] = t_float
            del t_float

        model.backbone.lm.tie_weights()
        gc.collect()
        return model.eval()
    finally:
        BB._lm_from_config = orig_stub
